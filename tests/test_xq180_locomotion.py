"""Check the locomotion height term, terrain offset and curriculum behavior."""

from types import SimpleNamespace
import unittest

import torch

import src.tasks  # noqa: F401
from src.tasks.amp_loco.config.xq180.mdp import fallen_height, locomotion_height_reward
from src.tasks.amp_loco.config.xq180.env_cfgs import xq180_amp_flat_env_cfg
from src.tasks.velocity.mdp.curriculums import commands_vel


class Scene(dict):
  pass


class LocomotionBehaviorTest(unittest.TestCase):
  def test_height_reward_and_fall_detection_respect_tile_origin(self):
    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0., 0., 1.05], [0., 0., 3.05], [0., 0., .70]]))))
    scene.env_origins = torch.tensor([[0., 0., 0.], [0., 0., 2.], [0., 0., 0.]])
    env = SimpleNamespace(scene=scene)
    reward = locomotion_height_reward(env, 1.05, .12)
    torch.testing.assert_close(reward[:2], torch.ones(2))
    self.assertLess(reward[2], .001)
    self.assertEqual(fallen_height(env, .75).tolist(), [False, False, True])

  def test_velocity_curriculum_uses_all_three_stages(self):
    cfg = xq180_amp_flat_env_cfg()
    command = cfg.commands["twist"]
    env = SimpleNamespace(command_manager=SimpleNamespace(
      get_term=lambda _: SimpleNamespace(cfg=command)))
    for step, expected in ((1, (-.3, .8)), (48001, (-.5, 1.2)), (144001, (-.8, 1.8))):
      env.common_step_counter = step
      commands_vel(env, torch.tensor([0]), **cfg.curriculum["command_vel"].params)
      self.assertEqual(command.ranges.lin_vel_x, expected)


if __name__ == "__main__":
  unittest.main()
