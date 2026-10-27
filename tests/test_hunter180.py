"""Hunter180 URDF fidelity, requested PD scaling and retargeted motion checks."""

import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from src import SRC_PATH
from src.assets.robots.hunter180.hunter180_constants import (
  HUNTER180_AMP_BODY_NAMES, HUNTER180_ARTICULATION, HUNTER180_FOOT_SITE_NAMES,
  HUNTER180_JOINT_NAMES, HUNTER180_PD_SCALE, HUNTER180_MASS_RATIO, HUNTER180_TOTAL_MASS,
  HUNTER130_REFERENCE_MASS, HUNTER180_URDF, get_hunter180_robot_cfg,
)
from src.tasks.amp_loco.config.hunter180.motion import validate_motion_files
from scripts.retarget_g1_to_hunter180 import G1ToHunter180
import src.tasks  # noqa: F401 - registration


class Hunter180Test(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.converter = G1ToHunter180()
    cls.robot = cls.converter.target
    cls.model = cls.converter.target_model
    cls.urdf = ET.parse(HUNTER180_URDF).getroot()

  def test_source_dynamics_and_joint_frames(self):
    m = self.model
    self.assertEqual((m.nq, m.nv, m.nu, m.nbody), (30, 29, 23, 25))
    self.assertAlmostEqual(float(m.body_mass.sum()), 64.34554445)
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
      np.testing.assert_allclose(actual @ np.diag(body.inertia) @ actual.T, rest @ matrix @ rest.T, atol=1e-8)
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

  def test_pd_three_times_historical_hunter130(self):
    self.assertAlmostEqual(HUNTER180_MASS_RATIO, HUNTER180_TOTAL_MASS / HUNTER130_REFERENCE_MASS)
    self.assertEqual(HUNTER180_PD_SCALE, 3.0)
    for actuator in HUNTER180_ARTICULATION.actuators:
      name = actuator.target_names_expr[0]
      if "hip_pitch" in name or "hip_roll" in name or "knee" in name:
        expected = (300.0, 18.9, 0.0251)
      elif "hip_yaw" in name or "waist_yaw" in name:
        expected = (120.0, 7.8, 0.0102)
      elif "ankle" in name or "waist" in name:
        expected = (85.5, 5.4, 0.0072)
      else:
        expected = (42.75, 2.7, 0.0036)
      self.assertEqual((actuator.stiffness, actuator.damping, actuator.armature), expected)

  def test_nominal_pose_original_feet_and_config_isolation(self):
    m = self.model; data = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, data, 0); mujoco.mj_forward(m, data)
    self.assertEqual(data.ncon, 0)
    feet = data.site_xpos[[m.site(n).id for n in HUNTER180_FOOT_SITE_NAMES]]
    self.assertTrue(np.all((feet[:, 2] >= 0) & (feet[:, 2] < 0.02)))
    for side in ("left", "right"):
      geom = m.geom(f"{side}_foot_collision")
      self.assertEqual(int(geom.type[0]), mujoco.mjtGeom.mjGEOM_MESH)
      self.assertEqual(int(geom.dataid[0]), m.mesh(f"{side}_ankle_roll_link").id)
    first, second = get_hunter180_robot_cfg(), get_hunter180_robot_cfg()
    first.init_state.joint_pos[".*_knee_joint"] = 1
    first.articulation.actuators[0].stiffness = 1
    self.assertEqual(second.init_state.joint_pos[".*_knee_joint"], 0.5)
    self.assertNotEqual(second.articulation.actuators[0].stiffness, 1)

  def test_registered_task_references(self):
    self.assertEqual(len(HUNTER180_JOINT_NAMES), 23)
    self.assertEqual(len(HUNTER180_AMP_BODY_NAMES), 11)
    for task in ("Hunter180-Flat", "Hunter180-Rough", "Hunter180-AMP-Flat", "Hunter180-AMP-Rough"):
      for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        self.assertEqual(len(cfg.actions["joint_pos"].scale), 23)
        self.assertEqual("height_scan" in cfg.observations["actor"].terms, task.endswith("Rough"))
        if "AMP" in task:
          self.assertIn("motions/hunter180/amp", cfg.events["reset_from_motion"].params["motion_dir"])
          self.assertEqual(cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names, HUNTER180_AMP_BODY_NAMES)
          self.assertEqual(len(load_rl_cfg(task).min_normalized_std), 23)

  def test_elbow_offsets_and_actual_retargeted_states(self):
    converter = self.converter
    self.assertEqual(converter.joint_offsets, {"left_elbow_joint": -1.57, "right_elbow_joint": -1.57})
    root = SRC_PATH / "assets/motions/hunter180/amp"
    paths = sorted(root.rglob("*.npz"))
    self.assertEqual(len(paths), 18)
    validate_motion_files(paths, self.robot.joint_names, self.robot.body_names)
    total_frames = 0
    for path in paths:
      with np.load(path) as target, np.load(SRC_PATH / "assets/motions/g1/amp" / path.relative_to(root)) as source:
        total_frames += len(target["joint_pos"])
        indices = [converter.source.joint_names.index(n) for n in self.robot.joint_names]
        offsets = np.array([converter.joint_offsets.get(n, 0.) for n in self.robot.joint_names])
        expected = np.clip(source["joint_pos"][:, indices] + offsets, converter.limits[:, 0], converter.limits[:, 1])
        np.testing.assert_allclose(target["joint_pos"], expected, atol=2e-7)
        self.assertEqual(target["joint_pos"].shape, (len(source["joint_pos"]), 23))
        self.assertEqual(target["body_pos_w"].shape, (len(source["joint_pos"]), 24, 3))
        for frame in (0, len(target["joint_pos"]) // 2, len(target["joint_pos"]) - 1):
          data = converter.target_data
          data.qpos[:] = np.r_[target["body_pos_w"][frame, 0], target["body_quat_w"][frame, 0], target["joint_pos"][frame]]
          mujoco.mj_kinematics(self.model, data)
          np.testing.assert_allclose(data.xpos[converter.body_ids], target["body_pos_w"][frame], atol=1e-6)
          self.assertAlmostEqual(converter.target_ground.minimum_z(data), float(target["ground_clearance"][frame]), places=6)
    self.assertEqual(total_frames, 9992)


if __name__ == "__main__":
  unittest.main()
