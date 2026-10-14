"""Validate retargeted motion geometry and velocities against MuJoCo."""

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.retarget_g1_to_hunter130 import G1ToHunter130, angular_velocity_w
from src import SRC_PATH
from src.tasks.amp_loco.config.hunter130.motion import validate_motion_files


class RetargetTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.converter = G1ToHunter130()

  def test_rotation_sign_and_two_frames(self):
    quaternions = np.array([[1., 0., 0., 0.], [-1., 0., 0., 0.]])
    np.testing.assert_allclose(angular_velocity_w(quaternions, 50), 0)
    angle = 0.04
    quaternions[1] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    np.testing.assert_allclose(angular_velocity_w(quaternions, 50), [[0, 0, 2], [0, 0, 2]])

  def test_converted_states_and_velocities(self):
    converter = self.converter
    model = converter.target_model
    with tempfile.TemporaryDirectory() as directory:
      source_path, target_path = Path(directory) / "g1.npz", Path(directory) / "hunter130.npz"
      # Nonstationary real frames exercise both translation and rotation.
      source_clip = SRC_PATH / "assets/motions/g1/amp/WalkandRun/arc_walk_left_loop_001__A029.npz"
      with np.load(source_clip) as source:
        values = {key: source[key][:9] if key != "fps" else source[key] for key in source.files}
      np.savez(source_path, **values)
      converter.convert(source_path, target_path)
      validate_motion_files([target_path], converter.target.joint_names, converter.target.body_names)
      with np.load(target_path) as target:
        self.assertEqual(target["joint_pos"].shape, (9, 25))
        self.assertEqual(target["body_pos_w"].shape, (9, 26, 3))
        np.testing.assert_allclose(target["body_pos_w"][:, 0, :2], values["body_pos_w"][:, 0, :2])
        indices = [converter.source.joint_names.index(name) for name in converter.target.joint_names]
        expected = np.clip(values["joint_pos"][:, indices], converter.limits[:, 0], converter.limits[:, 1])
        np.testing.assert_allclose(target["joint_pos"], expected, atol=1e-7)
        frame = 4
        data = mujoco.MjData(model)
        data.qpos[:] = np.r_[target["body_pos_w"][frame, 0], target["body_quat_w"][frame, 0], target["joint_pos"][frame]]
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(data.xpos[converter.body_ids], target["body_pos_w"][frame], atol=1e-6)
        self.assertAlmostEqual(converter.target_ground.minimum_z(data), float(target["ground_clearance"][frame]), places=6)
        rotation = Rotation.from_quat(target["body_quat_w"][frame, 0, [1, 2, 3, 0]])
        data.qvel[:] = np.r_[target["body_lin_vel_w"][frame, 0], rotation.inv().apply(target["body_ang_vel_w"][frame, 0]), target["joint_vel"][frame]]
        original = data.qpos.copy()
        epsilon = 1e-6
        poses = []
        quaternions = []
        for sign in (-1, 1):
          data.qpos[:] = original
          mujoco.mj_integratePos(model, data.qpos, data.qvel, sign * epsilon)
          mujoco.mj_kinematics(model, data)
          poses.append(data.xpos[converter.body_ids].copy())
          quaternions.append(data.xquat[converter.body_ids][:, [1, 2, 3, 0]].copy())
        np.testing.assert_allclose((poses[1] - poses[0]) / (2 * epsilon), target["body_lin_vel_w"][frame], atol=2e-6)
        angular = (Rotation.from_quat(quaternions[1]) * Rotation.from_quat(quaternions[0]).inv()).as_rotvec() / (2 * epsilon)
        np.testing.assert_allclose(angular, target["body_ang_vel_w"][frame], atol=2e-6)

  def test_mesh_support_uses_original_extrema(self):
    converter = self.converter
    model = converter.target_model
    directions = Rotation.from_euler("xyz", [[0, 0, 0], [0.8, -0.7, 1], [2, 0.4, -0.6]]).as_matrix()[:, 2]
    for mesh_id, hull_vertices in converter.target_ground.mesh_vertices.items():
      start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
      original = model.mesh_vert[start:start + count].astype(np.float64)
      np.testing.assert_allclose((original @ directions.T).min(axis=0), (hull_vertices @ directions.T).min(axis=0), atol=1e-9)


if __name__ == "__main__":
  unittest.main()
