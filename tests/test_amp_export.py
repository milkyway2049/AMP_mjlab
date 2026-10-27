"""Saving ONNX must not freeze or move the live training normalizer/policy."""

import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import src.tasks  # noqa: F401 - load task packages before the local RSL imports
from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.modules.actor_critic import ActorCritic
from src.tasks.amp_loco.rl.runner import AMPOnPolicyRunner


class ExportStateTest(unittest.TestCase):
  def check_export(self, fails):
    runner = object.__new__(AMPOnPolicyRunner)
    policy = ActorCritic(3, 3, 2, actor_hidden_dims=[4], critic_hidden_dims=[4]).train()
    policy.act(torch.ones(2, 3))  # Cache a distribution containing autograd tensors.
    runner.alg = SimpleNamespace(policy=policy)
    runner.empirical_normalization = True
    runner.obs_normalizer = EmpiricalNormalization([3]).train()
    runner.obs_normalizer(torch.randn(8, 3))
    runner.device = "cpu"
    count = runner.obs_normalizer.count.clone()
    before = {k: v.clone() for k, v in runner.alg.policy.state_dict().items()}

    def serialize(wrapper, sample, *args, **kwargs):
      self.assertFalse(wrapper.training)
      self.assertFalse(wrapper.obs_normalizer.training)
      self.assertIsNot(wrapper.actor, runner.alg.policy.actor)
      self.assertIsNot(wrapper.obs_normalizer, runner.obs_normalizer)
      self.assertEqual(wrapper(sample).shape, (1, 2))
      if fails:
        raise RuntimeError("export failed")

    with tempfile.TemporaryDirectory() as directory, patch("torch.onnx.export", side_effect=serialize):
      if fails:
        with self.assertRaisesRegex(RuntimeError, "export failed"):
          runner._export_policy_to_onnx(directory)
      else:
        runner._export_policy_to_onnx(directory)
    self.assertTrue(runner.alg.policy.training)
    self.assertTrue(runner.obs_normalizer.training)
    self.assertEqual(runner.obs_normalizer.count, count)
    for k, value in runner.alg.policy.state_dict().items():
      torch.testing.assert_close(value, before[k])
    runner.obs_normalizer(torch.ones(4, 3))
    self.assertEqual(runner.obs_normalizer.count, count + 4)

  def test_training_continues_after_export(self):
    self.check_export(False)

  def test_training_state_survives_export_failure(self):
    self.check_export(True)


if __name__ == "__main__":
  unittest.main()
