"""Exercise real PPO/AMP updates, raw statistics, gradients and resume behavior."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import src.tasks  # noqa: F401 - register before local RSL imports
from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner
from rsl_rl.utils import Normalizer


class Expert:
  def feed_forward_generator(self, batches, size):
    for _ in range(batches):
      yield torch.full((size, 2), 120.), torch.full((size, 2), 121.)


class RecordingNormalizer(Normalizer):
  def __init__(self):
    super().__init__(2)
    self.samples = []

  def update(self, samples):
    self.samples.append(samples.copy())
    super().update(samples)


class RecordingDiscriminator(Discriminator):
  def __init__(self):
    super().__init__(4, .1, [8], 'cpu', .75)
    self.inputs = []
    self.penalty_input = None

  def forward(self, x):
    self.inputs.append(x.detach().clone())
    return super().forward(x)

  def compute_grad_pen(self, state, next_state, lambda_=10):
    self.penalty_input = torch.cat([state, next_state], dim=-1).detach().clone()
    return super().compute_grad_pen(state, next_state, lambda_)


def make_algorithm():
  torch.manual_seed(42)
  normalizer = RecordingNormalizer()
  normalizer.update(np.array([[0., 0.], [20., 20.]], dtype=np.float32))
  normalizer.samples.clear()
  policy = ActorCritic(3, 3, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
  alg = AMPPPO(policy, RecordingDiscriminator(), Expert(), normalizer,
    amp_replay_buffer_size=64, num_learning_epochs=1, num_mini_batches=1,
    learning_rate=3e-4, max_grad_norm=.1, desired_kl=.01, action_clip=3.)
  alg.init_storage('rl', 4, 4, [3], [3], [2])
  return alg


def collect(alg):
  with torch.no_grad():
    for _ in range(4):
      obs = torch.randn(4, 3) * .1
      amp = torch.full((4, 2), 100.)
      alg.act(obs, obs, amp)
      alg.process_env_step(torch.ones(4) * .1, torch.zeros(4), {}, amp)
    alg.compute_returns(torch.zeros(4, 3))


def make_runner():
  runner = object.__new__(AmpOnPolicyRunner)
  runner.alg = make_algorithm()
  runner.device = 'cpu'
  runner.cfg = {'empirical_normalization': True,
    'amp_learning_rate_decay_stages': [(6000, 1 / 3), (20000, 1 / 6)]}
  runner.alg_cfg = {'learning_rate': 3e-4}
  runner.current_learning_iteration = 0
  runner.empirical_normalization = True
  runner.obs_normalizer = EmpiricalNormalization([3])
  runner.privileged_obs_normalizer = EmpiricalNormalization([3])
  runner.obs_normalizer(torch.randn(8, 3))
  runner.privileged_obs_normalizer(torch.randn(8, 3))
  runner.logger_type = 'tensorboard'
  return runner


class AmpUpdateTest(unittest.TestCase):
  def test_update_preserves_raw_stats_and_normalizes_penalty_inputs(self):
    alg = make_algorithm()
    collect(alg)
    before = [p.detach().clone() for p in alg.policy.parameters()]
    loss = alg.update()
    self.assertEqual(loss['effective_updates'], 1)
    self.assertEqual(loss['skipped_non_finite_batches'], 0)
    np.testing.assert_array_equal(alg.amp_normalizer.samples[0], np.full((16, 2), 100.))
    np.testing.assert_array_equal(alg.amp_normalizer.samples[1], np.full((16, 2), 120.))
    # The penalty must see precisely the input seen by the expert discriminator.
    torch.testing.assert_close(alg.discriminator.penalty_input, alg.discriminator.inputs[1])
    self.assertLessEqual(alg.discriminator.penalty_input.abs().max(), 10)
    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, alg.policy.parameters())))
    for module in (alg.policy, alg.discriminator):
      norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in module.parameters() if p.grad is not None]))
      self.assertLessEqual(norm, .10001)
      self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))
    self.assertTrue(all(np.isfinite(v) for v in loss.values()))
    self.assertGreaterEqual(loss['kl'], 0)
    for name in ('clip_fraction', 'action_saturation_fraction'):
      self.assertGreaterEqual(loss[name], 0)
      self.assertLessEqual(loss[name], 1)

  def test_nonfinite_amp_loss_does_not_step_or_update_statistics(self):
    alg = make_algorithm()
    collect(alg)
    with patch.object(alg.discriminator, 'compute_grad_pen', return_value=torch.tensor(float('nan'))), \
         patch.object(alg.optimizer, 'step') as step:
      with self.assertRaisesRegex(RuntimeError, 'skipped every minibatch'):
        alg.update()
      step.assert_not_called()
    self.assertEqual(alg.amp_normalizer.samples, [])

  def test_nonfinite_discriminator_gradient_does_not_step(self):
    alg = make_algorithm()
    collect(alg)
    next(alg.discriminator.parameters()).register_hook(lambda g: torch.full_like(g, float('nan')))
    with patch.object(alg.optimizer, 'step') as step:
      with self.assertRaisesRegex(RuntimeError, 'skipped every minibatch'):
        alg.update()
      step.assert_not_called()
    self.assertEqual(alg.amp_normalizer.samples, [])

  def test_one_bad_expert_batch_does_not_poison_valid_updates(self):
    alg = make_algorithm()
    alg.num_mini_batches = 2
    collect(alg)

    def batches(count, size):
      yield torch.full((size, 2), float('nan')), torch.ones(size, 2)
      yield torch.full((size, 2), 120.), torch.full((size, 2), 121.)

    alg.amp_data.feed_forward_generator = batches
    loss = alg.update()
    self.assertEqual(loss['effective_updates'], 1)
    self.assertEqual(loss['skipped_non_finite_batches'], 1)
    self.assertTrue(np.isfinite(alg.amp_normalizer.mean).all())
    self.assertEqual(len(alg.amp_normalizer.samples), 2)

  def test_distributed_gradient_sync_includes_discriminator(self):
    alg = make_algorithm()
    alg.is_multi_gpu = True
    alg.gpu_world_size = 2
    parameters = list(alg.policy.parameters()) + list(alg.discriminator.parameters())
    for p in parameters:
      p.grad = torch.ones_like(p)
    # Model an all-reduce result where the other rank contributes gradient 5.
    with patch('torch.distributed.all_reduce', side_effect=lambda x, **kw: x.fill_(6)):
      alg.reduce_parameters()
    for p in parameters:
      torch.testing.assert_close(p.grad, torch.full_like(p, 3))
    with patch('torch.distributed.all_reduce', side_effect=lambda x, **kw: x.zero_()):
      self.assertFalse(alg._all_finite(torch.ones(1)))

  def test_kl_stop_freezes_policy_but_continues_expert_training(self):
    alg = make_algorithm()
    alg.kl_stop_multiplier = 4.
    alg.num_learning_epochs = 2
    collect(alg)
    # Simulate a policy that has already moved far from the collected rollout.
    with torch.no_grad():
      alg.policy.actor[-1].bias.add_(2.)
    policy_before = deepcopy(alg.policy.state_dict())
    discriminator_before = deepcopy(alg.discriminator.state_dict())
    loss = alg.update()
    self.assertEqual(loss['policy_kl_stopped'], 1)
    self.assertEqual(loss['policy_updates'], 0)
    self.assertEqual(loss['effective_updates'], 2)
    for key, value in alg.policy.state_dict().items():
      torch.testing.assert_close(value, policy_before[key])
    self.assertTrue(any(not torch.equal(v, discriminator_before[k]) for k, v in alg.discriminator.state_dict().items()))
    self.assertEqual(len(alg.amp_normalizer.samples), 4)

  def test_algorithm_checkpoint_version_and_explicit_reset(self):
    alg = make_algorithm()
    collect(alg)
    alg.update()
    state = deepcopy(alg.save())
    self.assertEqual(state['amp_normalizer_version'], 2)
    resumed = make_algorithm()
    resumed.load(state)
    self.assertEqual(resumed.amp_normalizer_version, 2)
    legacy = deepcopy(state)
    legacy.pop('amp_normalizer_version')
    resumed.load(legacy)
    with self.assertRaisesRegex(RuntimeError, 'Legacy AMP normalization'):
      resumed.update()
    policy = deepcopy(resumed.policy.state_dict())
    resumed.load(legacy, {'reset_amp': True})
    self.assertEqual(resumed.amp_normalizer_version, 2)
    self.assertLess(resumed.amp_normalizer.count, 1)
    self.assertEqual(len(resumed.optimizer.state), 0)
    for key, value in resumed.policy.state_dict().items():
      torch.testing.assert_close(value, policy[key])


class AmpResumeTest(unittest.TestCase):
  def test_learning_rate_stages_update_optimizer_groups(self):
    runner = make_runner()
    for iteration, rate in ((0, 3e-4), (5999, 3e-4), (6000, 1e-4), (20000, 5e-5)):
      runner._apply_learning_rate_schedule(iteration)
      self.assertAlmostEqual(runner.alg.learning_rate, rate)
      for group in runner.alg.optimizer.param_groups:
        self.assertAlmostEqual(group['lr'], rate)

  def test_resume_honors_configured_lr_and_legacy_reset_preserves_actor(self):
    saved = make_runner()
    collect(saved.alg)
    saved.alg.update()
    saved.current_learning_iteration = 30000
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'model.pt'
      saved.save(str(path))
      resumed = make_runner()
      resumed.alg_cfg['learning_rate'] = 6e-4  # Different from optimizer in saved checkpoint.
      resumed.load(str(path))
      self.assertEqual(resumed.alg.amp_normalizer_version, 2)
      self.assertAlmostEqual(resumed.alg.learning_rate, 1e-4)
      for group in resumed.alg.optimizer.param_groups:
        self.assertAlmostEqual(group['lr'], 1e-4)
      state = torch.load(path, weights_only=False)
      state.pop('amp_normalizer_version')
      torch.save(state, path)
      resumed.load(str(path))
      # Loading for inference remains supported; training requires explicit reset.
      resumed.get_inference_policy()(torch.ones(1, 3))
      with self.assertRaisesRegex(RuntimeError, 'amp-reset-on-load'):
        resumed.learn(1)
      resumed.load(str(path), reset_amp=True)
      self.assertEqual(resumed.alg.amp_normalizer_version, 2)
      self.assertEqual(resumed.alg.amp_storage.num_samples, 0)
      self.assertEqual(len(resumed.alg.optimizer.state), 0)
      for key, value in resumed.alg.policy.state_dict().items():
        torch.testing.assert_close(value, saved.alg.policy.state_dict()[key])
      torch.testing.assert_close(resumed.obs_normalizer._mean, saved.obs_normalizer._mean)


if __name__ == '__main__':
  unittest.main()
