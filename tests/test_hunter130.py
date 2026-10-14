"""Regression checks for URDF dynamics, task wiring and motion compatibility."""

import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from src.assets.robots.hunter130.hunter130_constants import (
  HUNTER130_ACTION_SCALE, HUNTER130_AMP_BODY_NAMES, HUNTER130_EFFORT_LIMITS,
  HUNTER130_FOOT_SITE_NAMES, HUNTER130_JOINT_NAMES, HUNTER130_URDF,
  get_hunter130_robot_cfg,
)
from src.tasks.amp_loco.config.hunter130.motion import validate_motion_files
from src.assets.robots.unitree_g1.g1_constants import G1_ARTICULATION
import src.tasks  # noqa: F401 - task registration


class Hunter130Test(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.robot = Entity(get_hunter130_robot_cfg())
    cls.model = cls.robot.spec.compile()
    cls.urdf = ET.parse(HUNTER130_URDF).getroot()

  def test_urdf_dynamics(self):
    m = self.model
    self.assertEqual((m.nq, m.nv, m.nu, m.nbody), (32, 31, 25, 27))
    for link in self.urdf.findall("link"):
      body = m.body(link.attrib["name"])
      inertial = link.find("inertial")
      self.assertAlmostEqual(float(body.mass[0]), float(inertial.find("mass").attrib["value"]))
      np.testing.assert_allclose(body.ipos, np.fromstring(inertial.find("origin").attrib["xyz"], sep=" "))
      source = inertial.find("inertia").attrib
      expected = np.array([[float(source["ixx"]), float(source["ixy"]), float(source["ixz"])],
                           [float(source["ixy"]), float(source["iyy"]), float(source["iyz"])],
                           [float(source["ixz"]), float(source["iyz"]), float(source["izz"])]])
      rot = Rotation.from_euler("xyz", np.fromstring(inertial.find("origin").attrib["rpy"], sep=" ")).as_matrix()
      expected = rot @ expected @ rot.T
      actual_rot = Rotation.from_quat(body.iquat[[1, 2, 3, 0]]).as_matrix()
      # MuJoCo's fullinertia eigensolver has a 1e-8 absolute stopping tolerance.
      np.testing.assert_allclose(actual_rot @ np.diag(body.inertia) @ actual_rot.T, expected, atol=1e-8)

  def test_joint_transforms_limits_and_actuator_order(self):
    m = self.model
    self.assertEqual(tuple(m.joint(i).name for i in range(1, m.njnt)), HUNTER130_JOINT_NAMES)
    for i, joint in enumerate(self.urdf.findall("joint")):
      name = joint.attrib["name"]
      body = m.body(joint.find("child").attrib["link"])
      node = joint.find("origin")
      np.testing.assert_allclose(body.pos, np.fromstring(node.attrib["xyz"], sep=" "), atol=1e-9)
      expected = Rotation.from_euler("xyz", np.fromstring(node.attrib["rpy"], sep=" ")).as_matrix()
      np.testing.assert_allclose(Rotation.from_quat(body.quat[[1, 2, 3, 0]]).as_matrix(), expected, atol=1e-9)
      limit = joint.find("limit").attrib
      np.testing.assert_allclose(m.joint(name).range, [float(limit["lower"]), float(limit["upper"])])
      np.testing.assert_allclose(m.joint(name).axis, np.fromstring(joint.find("axis").attrib["xyz"], sep=" "))
      self.assertEqual(int(m.actuator_trnid[i, 0]), m.joint(name).id)
      np.testing.assert_allclose(m.actuator_forcerange[i], [-float(limit["effort"]), float(limit["effort"])])
      self.assertLessEqual(HUNTER130_ACTION_SCALE[name] * m.actuator_gainprm[i, 0], HUNTER130_EFFORT_LIMITS[name])

  def test_nominal_pose_feet_and_sensors(self):
    m = self.model
    data = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, data, 0)
    mujoco.mj_forward(m, data)
    self.assertEqual(data.ncon, 0)
    feet = data.site_xpos[[m.site(n).id for n in HUNTER130_FOOT_SITE_NAMES]]
    self.assertTrue(np.all((feet[:, 2] >= 0) & (feet[:, 2] < 0.02)))
    for name in ("imu_ang_vel", "imu_lin_vel", "imu_lin_acc", "root_angmom"):
      self.assertEqual(int(m.sensor(name).dim[0]), 3)
    for side in ("left", "right"):
      geom = m.geom(f"{side}_foot_collision")
      self.assertEqual(int(geom.type[0]), mujoco.mjtGeom.mjGEOM_MESH)
      self.assertEqual(int(geom.dataid[0]), m.mesh(f"{side}_ankle_roll_link").id)
    for i in range(1, m.njnt):
      self.assertTrue(m.jnt_range[i, 0] <= data.qpos[m.jnt_qposadr[i]] <= m.jnt_range[i, 1])

  def test_configs_are_independent(self):
    first, second = get_hunter130_robot_cfg(), get_hunter130_robot_cfg()
    first.init_state.joint_pos[".*_knee_joint"] = 1.0
    original_stiffness = second.articulation.actuators[0].stiffness
    first.articulation.actuators[0].stiffness = 1.0
    self.assertEqual(second.init_state.joint_pos[".*_knee_joint"], 0.3)
    self.assertEqual(second.articulation.actuators[0].stiffness, original_stiffness)

  def test_pd_matches_g1(self):
    for actuator in get_hunter130_robot_cfg().articulation.actuators:
      name = actuator.target_names_expr[0]
      reference = next(a for a in G1_ARTICULATION.actuators if any(re.fullmatch(p, name) for p in a.target_names_expr))
      self.assertEqual(actuator.stiffness, reference.stiffness)
      self.assertEqual(actuator.damping, reference.damping)

  def test_task_configs(self):
    for task in ("Hunter130-Flat", "Hunter130-Rough", "Hunter130-AMP-Flat", "Hunter130-AMP-Rough"):
      for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        self.assertEqual(cfg.viewer.body_name, "torso_link")
        self.assertEqual(len(cfg.actions["joint_pos"].scale), 25)
        self.assertEqual("height_scan" in cfg.observations["actor"].terms, task.endswith("Rough"))
        for name in cfg.events["foot_friction"].params["asset_cfg"].geom_names:
          self.assertEqual(int(self.model.geom(name).condim[0]), 3)
        if "AMP" in task:
          self.assertEqual(cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names, HUNTER130_AMP_BODY_NAMES)
          runner = load_rl_cfg(task)
          self.assertEqual(len(runner.min_normalized_std), 25)
          self.assertEqual(runner.amp_body_names, HUNTER130_AMP_BODY_NAMES)
          self.assertIn("motions/hunter130/amp", runner.amp_motion_files)
          self.assertIn("motions/hunter130/amp", cfg.events["reset_from_motion"].params["motion_dir"])

  def test_motion_order_validation(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "clip.npz"
      joints, bodies = self.robot.joint_names, self.robot.body_names
      data = {
        "fps": [50.0], "joint_names": joints, "body_names": bodies,
        "joint_pos": np.zeros((2, 25)), "joint_vel": np.zeros((2, 25)),
        "body_pos_w": np.zeros((2, 26, 3)), "body_quat_w": np.tile([1., 0., 0., 0.], (2, 26, 1)),
        "body_lin_vel_w": np.zeros((2, 26, 3)), "body_ang_vel_w": np.zeros((2, 26, 3)),
      }
      np.savez(path, **data)
      validate_motion_files([path], joints, bodies)
      data["joint_names"] = tuple(reversed(joints))
      np.savez(path, **data)
      with self.assertRaisesRegex(ValueError, "must match Hunter130"):
        validate_motion_files([path], joints, bodies)
      data["joint_names"] = joints
      data["joint_pos"] = np.zeros((2, 29))
      np.savez(path, **data)
      with self.assertRaisesRegex(ValueError, "shape"):
        validate_motion_files([path], joints, bodies)


if __name__ == "__main__":
  unittest.main()
