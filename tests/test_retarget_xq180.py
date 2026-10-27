"""Contact IK data must agree with XQ180 geometry and reset constraints."""

import unittest
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.xq180_retarget_ik import G1ToXQ180IK
from src import SRC_PATH
from src.tasks.amp_loco.config.xq180.motion import validate_motion_files


class XQ180IKTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.converter = G1ToXQ180IK()
    cls.root = SRC_PATH / 'assets/motions/xq180/amp_ik'
    cls.paths = sorted(cls.root.rglob('*.npz'))

  def test_all_clips_limits_timing_and_forward_kinematics(self):
    c = self.converter; m = c.target_model; d = c.target_data
    self.assertEqual(len(self.paths), 18)
    validate_motion_files(self.paths, c.target.joint_names, c.target.body_names)
    total = 0
    for path in self.paths:
      with self.subTest(clip=path.name), np.load(path) as target, np.load(
          SRC_PATH/'assets/motions/g1/amp'/path.relative_to(self.root)) as source:
        q = target['joint_pos']; total += len(q)
        self.assertEqual(target['retarget_method'].item(), c.method)
        self.assertEqual(len(q), len(source['joint_pos']))
        self.assertEqual(float(target['fps'].item()), float(source['fps'].item()))
        self.assertTrue(np.all(q >= c.soft_lower-1e-6))
        self.assertTrue(np.all(q <= c.soft_upper+1e-6))
        np.testing.assert_allclose(target['joint_vel'], np.gradient(q.astype(float), axis=0)*50, atol=5e-5)
        self.assertTrue(np.all(abs(target['joint_vel']) <= c.velocity_limits*1.001))
        np.testing.assert_allclose(np.linalg.norm(target['body_quat_w'], axis=-1), 1., atol=1e-6)
        for frame in (0, len(q)//2, len(q)-1):
          d.qpos[:] = np.r_[target['body_pos_w'][frame, 0], target['body_quat_w'][frame, 0], q[frame]]
          root_rotation = Rotation.from_quat(target['body_quat_w'][frame, 0, [1, 2, 3, 0]])
          d.qvel[:] = np.r_[target['body_lin_vel_w'][frame, 0],
            root_rotation.inv().apply(target['body_ang_vel_w'][frame, 0]), target['joint_vel'][frame]]
          mujoco.mj_forward(m, d)
          np.testing.assert_allclose(d.xpos[c.body_ids], target['body_pos_w'][frame], atol=1e-6)
          self.assertGreaterEqual(c.target_ground.minimum_z(d), .0019)
          velocity = np.zeros(6)
          for i, body in enumerate(c.body_ids):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            np.testing.assert_allclose(velocity[:3], target['body_ang_vel_w'][frame, i], atol=2e-5)
            np.testing.assert_allclose(velocity[3:], target['body_lin_vel_w'][frame, i], atol=2e-5)
    self.assertEqual(total, 9992)

  def test_all_frame_ground_and_self_collision_quality(self):
    c = self.converter; m = c.target_model; d = c.target_data
    bad = total = 0
    for path in self.paths:
      with np.load(path) as motion:
        for frame in range(len(motion['joint_pos'])):
          d.qpos[:] = np.r_[motion['body_pos_w'][frame, 0], motion['body_quat_w'][frame, 0], motion['joint_pos'][frame]]
          mujoco.mj_forward(m, d)
          self.assertGreaterEqual(c.target_ground.minimum_z(d), .0019, path.name)
          bad += any(contact.dist < -.001 for contact in d.contact)
          total += 1
    # The old joint-mapped dataset has 14.04% self-penetrating frames.
    self.assertGreater(total, 9000)
    self.assertLess(bad/total, .005)

  def test_converter_roundtrip(self):
    c = self.converter
    source = SRC_PATH/'assets/motions/g1/amp/WalkandRun/walk_forward_loop_002__A022.npz'
    with tempfile.TemporaryDirectory() as folder:
      target = Path(folder)/'converted.npz'
      c.convert(source, target, frame_limit=9)
      validate_motion_files([target], c.target.joint_names, c.target.body_names)
      self.assertFalse(target.with_suffix('.tmp').exists())
      with np.load(target, allow_pickle=False) as motion:
        self.assertEqual(motion['joint_pos'].shape, (9, 19))
        self.assertEqual(motion['body_pos_w'].shape, (9, 20, 3))
        self.assertEqual(motion['retarget_method'].item(), c.method)
        self.assertGreaterEqual(motion['ground_clearance'].min(), .0019)


if __name__ == '__main__':
  unittest.main()
