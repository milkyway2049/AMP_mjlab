"""XQ180 source dynamics, inertia repair, task wiring and motion mapping checks."""

import unittest
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from src import SRC_PATH
from src.assets.robots.xq180.xq180_constants import (
  XQ180_AMP_BODY_NAMES, XQ180_ARTICULATION, XQ180_FOOT_SITE_NAMES,
  XQ180_JOINT_NAMES, XQ180_TOTAL_MASS, XQ180_ACTION_SCALE,
  XQ180_URDF, XQ180_PD_PROFILE, get_xq180_robot_cfg,
  g1_joint_name,
)
from src.assets.robots.unitree_g1.g1_constants import G1_ARTICULATION
import re
from src.tasks.amp_loco.config.xq180.motion import validate_motion_files
from src.tasks.amp_loco.config.xq180.motion import XQ180_LOCOMOTION_DIR
from scripts.derive_xq180_pd import derive, joint_group
from scripts.retarget_g1_to_xq180 import G1ToXQ180
import src.tasks  # noqa: F401 - registration


class XQ180Test(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.converter = G1ToXQ180()
    cls.robot = cls.converter.target
    cls.model = cls.converter.target_model
    cls.urdf = ET.parse(XQ180_URDF).getroot()

  def test_source_dynamics_and_joint_frames(self):
    m = self.model
    self.assertEqual((m.nq, m.nv, m.nu, m.nbody), (26, 25, 19, 21))
    self.assertAlmostEqual(float(m.body_mass.sum()), 66.609902)
    for link in self.urdf.findall("link"):
      body = m.body(link.get("name"))
      inertia = link.find("inertial")
      self.assertAlmostEqual(float(body.mass[0]), float(inertia.find("mass").get("value")))
      np.testing.assert_allclose(body.ipos, np.fromstring(inertia.find("origin").get("xyz"), sep=" "))
      values = inertia.find("inertia").attrib
      matrix = np.array([[float(values["ixx"]), float(values["ixy"]), float(values["ixz"])],
                         [float(values["ixy"]), float(values["iyy"]), float(values["iyz"])],
                         [float(values["ixz"]), float(values["iyz"]), float(values["izz"])]])
      rest = Rotation.from_euler("xyz", np.fromstring(inertia.find("origin").get("rpy"), sep=" ")).as_matrix()
      actual = Rotation.from_quat(body.iquat[[1, 2, 3, 0]]).as_matrix()
      expected = rest @ matrix @ rest.T
      if link.get("name") == "left_shoulder_pitch_Link":
        values, axes = np.linalg.eigh(expected)
        self.assertGreater(values[2], values[0] + values[1])
        values[2] = (values[0] + values[1]) * 0.999
        expected = axes @ np.diag(values) @ axes.T
      np.testing.assert_allclose(actual @ np.diag(body.inertia) @ actual.T, expected, atol=1e-8)
    for i, joint in enumerate(self.urdf.findall("joint")):
      name = joint.get("name"); body = m.body(joint.find("child").get("link"))
      np.testing.assert_allclose(body.pos, np.fromstring(joint.find("origin").get("xyz"), sep=" "), atol=1e-9)
      expected = Rotation.from_euler("xyz", np.fromstring(joint.find("origin").get("rpy"), sep=" ")).as_matrix()
      np.testing.assert_allclose(Rotation.from_quat(body.quat[[1, 2, 3, 0]]).as_matrix(), expected, atol=1e-9)
      np.testing.assert_allclose(m.joint(name).range, [float(joint.find("limit").get(k)) for k in ("lower", "upper")])
      np.testing.assert_allclose(m.joint(name).axis, np.fromstring(joint.find("axis").get("xyz"), sep=" "))
      self.assertEqual(int(m.actuator_trnid[i, 0]), m.joint(name).id)
      effort = float(joint.find("limit").get("effort"))
      np.testing.assert_allclose(m.actuator_forcerange[i], [-effort, effort])

  def test_user_pd_v5_and_bounded_action_scale(self):
    self.assertAlmostEqual(XQ180_TOTAL_MASS, 66.609902)
    report = derive()
    expected_groups = {
      "hip_yaw": (200, 10), "hip_roll": (400, 20), "hip_pitch": (300, 15),
      "knee": (300, 15), "ankle": (100, 5), "waist": (100, 5),
      "shoulder_pitch": (50, 2), "shoulder_roll": (50, 2),
      "shoulder_yaw": (25, 1), "elbow": (25, 1),
    }
    io = json.loads((SRC_PATH.parent / "docs/xq180_policy_io.json").read_text())
    self.assertEqual(io["pd_profile"], XQ180_PD_PROFILE)
    self.assertEqual(XQ180_PD_PROFILE, "user_pd_v5")
    self.assertEqual([j["name"] for j in io["joints"]], list(XQ180_JOINT_NAMES))
    deployment = {j["name"]: j for j in io["joints"]}
    self.assertEqual(report["clips"], 17)
    self.assertEqual(report["sampled_poses"], 544)
    yaw = next(row for row in report["joint_inertia_ratios"] if row["joint"] == "left_hip_yaw_joint")
    self.assertEqual(yaw["g1_chain"]["root_to_joint"],
      ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"])
    self.assertEqual(yaw["xq180_chain"]["root_to_joint"], ["left_hip_yaw_joint"])
    for actuator in XQ180_ARTICULATION.actuators:
      name = actuator.target_names_expr[0]
      reference = next(a for a in G1_ARTICULATION.actuators
        if any(re.fullmatch(p, g1_joint_name(name)) for p in a.target_names_expr))
      expected = expected_groups[joint_group(name)]
      self.assertEqual((actuator.stiffness, actuator.damping), expected)
      self.assertEqual((deployment[name]["kp"], deployment[name]["kd"]), expected)
      self.assertAlmostEqual(deployment[name]["action_scale"], XQ180_ACTION_SCALE[name])
      self.assertGreaterEqual(actuator.damping, 1)
      self.assertLessEqual(XQ180_ACTION_SCALE[name] * actuator.stiffness,
                           0.25 * actuator.effort_limit + 1e-9)
      self.assertLessEqual(XQ180_ACTION_SCALE[name],
                           0.25 * reference.effort_limit / reference.stiffness + 1e-9)
      self.assertEqual(actuator.armature, 0.01)
    self.assertEqual(g1_joint_name("waist_point"), "waist_yaw_joint")
    self.assertEqual(g1_joint_name("left_ankle_joint"), "left_ankle_pitch_joint")
    self.assertEqual(g1_joint_name("rightelbow_joint"), "right_elbow_joint")

  def test_nominal_pose_original_feet_and_config_isolation(self):
    m = self.model; data = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, data, 0); mujoco.mj_forward(m, data)
    self.assertEqual(data.ncon, 0)
    feet = data.site_xpos[[m.site(n).id for n in XQ180_FOOT_SITE_NAMES]]
    self.assertTrue(np.all((feet[:, 2] >= 0) & (feet[:, 2] < 0.04)))
    for side in ("left", "right"):
      geom = m.geom(f"{side}_foot_collision")
      self.assertEqual(int(geom.type[0]), mujoco.mjtGeom.mjGEOM_MESH)
      self.assertEqual(int(geom.dataid[0]), m.mesh(f"{side}_ankle_Link").id)
    first, second = get_xq180_robot_cfg(), get_xq180_robot_cfg()
    first.init_state.joint_pos[".*_knee_joint"] = 1
    first.articulation.actuators[0].stiffness = 1
    self.assertEqual(second.init_state.joint_pos[".*_knee_joint"], 0.5)
    self.assertNotEqual(second.articulation.actuators[0].stiffness, 1)

  def test_registered_task_references(self):
    self.assertEqual(len(XQ180_JOINT_NAMES), 19)
    self.assertEqual(len(XQ180_AMP_BODY_NAMES), 11)
    for task in ("XQ180-Flat", "XQ180-Rough", "XQ180-AMP-Flat", "XQ180-AMP-Rough"):
      for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        self.assertEqual(len(cfg.actions["joint_pos"].scale), 19)
        self.assertEqual("height_scan" in cfg.observations["actor"].terms, task.endswith("Rough"))
        if "AMP" in task:
          self.assertIn("motions/xq180/amp_ik", cfg.events["reset_from_motion"].params["motion_dir"])
          self.assertEqual(cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names, XQ180_AMP_BODY_NAMES)
          self.assertEqual(len(load_rl_cfg(task).min_normalized_std), 19)
          self.assertEqual(Path(load_rl_cfg(task).amp_motion_files),
            Path(cfg.events["reset_from_motion"].params["motion_dir"]))

  def test_locomotion_only_training_and_play(self):
    paths = sorted(XQ180_LOCOMOTION_DIR.rglob("*.npz"))
    self.assertEqual(len(paths), 17)
    self.assertTrue(all("Recovery" not in p.parts for p in paths))
    for task in ("XQ180-AMP-Flat", "XQ180-AMP-Rough"):
      for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        init = cfg.events["init_motion_loader"].params
        self.assertEqual(Path(init["motion_dir"]), XQ180_LOCOMOTION_DIR)
        self.assertIsNone(init["recovery_dir"])
        self.assertEqual(init["delay_reset_env_ratio"], 0)
        self.assertEqual(init["max_delay_steps"], 0)
        self.assertEqual(cfg.terminations["bad_base_height"].params["minimum_height"], .75)
        self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (-.3, .8))
        runner = load_rl_cfg(task)
        self.assertEqual(Path(runner.amp_motion_files), XQ180_LOCOMOTION_DIR)
        self.assertEqual(runner.clip_actions, 3.)
        self.assertEqual(runner.algorithm.schedule, "fixed")

  def test_elbow_offsets_and_actual_retargeted_states(self):
    converter = self.converter
    self.assertEqual(converter.joint_offsets, {"leftelbow_joint": -np.pi / 2, "rightelbow_joint": -np.pi / 2})
    root = SRC_PATH / "assets/motions/xq180/amp"
    paths = sorted(root.rglob("*.npz"))
    self.assertEqual(len(paths), 18)
    validate_motion_files(paths, self.robot.joint_names, self.robot.body_names)
    total_frames = 0
    for path in paths:
      with np.load(path) as target, np.load(SRC_PATH / "assets/motions/g1/amp" / path.relative_to(root)) as source:
        total_frames += len(target["joint_pos"])
        indices = [converter.source.joint_names.index(converter.joint_mapping[n]) for n in self.robot.joint_names]
        offsets = np.array([converter.joint_offsets.get(n, 0.) for n in self.robot.joint_names])
        expected = np.clip(source["joint_pos"][:, indices] + offsets, converter.limits[:, 0], converter.limits[:, 1])
        np.testing.assert_allclose(target["joint_pos"], expected, atol=2e-7)
        self.assertEqual(target["joint_pos"].shape, (len(source["joint_pos"]), 19))
        self.assertEqual(target["body_pos_w"].shape, (len(source["joint_pos"]), 20, 3))
        for frame in (0, len(target["joint_pos"]) // 2, len(target["joint_pos"]) - 1):
          data = converter.target_data
          data.qpos[:] = np.r_[target["body_pos_w"][frame, 0], target["body_quat_w"][frame, 0], target["joint_pos"][frame]]
          mujoco.mj_kinematics(self.model, data)
          np.testing.assert_allclose(data.xpos[converter.body_ids], target["body_pos_w"][frame], atol=1e-6)
          self.assertAlmostEqual(converter.target_ground.minimum_z(data), float(target["ground_clearance"][frame]), places=6)
    self.assertEqual(total_frames, 9992)


if __name__ == "__main__":
  unittest.main()
