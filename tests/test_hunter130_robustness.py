"""CPU checks for mat contact sampling and standing/recovery reward behavior."""

from types import SimpleNamespace
import unittest

import torch

from mjlab.actuator import DelayedActuatorCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg

import src.tasks  # noqa: F401 - task registration
from src.tasks.hunter130_robustness import (
  contact_slip,
  randomize_foot_contact,
  standing_motion,
  torso_stability,
)


def _contact_env(num_envs=8):
  # Robot-local geom indices deliberately differ from global model indices.
  defaults = {
    "geom_solref": torch.tensor([[0.02 + i * 0.001, 1.0] for i in range(6)]),
    "geom_solimp": torch.tensor([[0.9, 0.95, 0.001, 0.5, 2.0]] * 6),
  }
  model = SimpleNamespace(**{
    name: value.unsqueeze(0).repeat(num_envs, 1, 1)
    for name, value in defaults.items()
  })
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    physics_dt=0.005,
    cfg=SimpleNamespace(sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.005))),
    scene={"robot": SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.tensor([2, 4])))},
    sim=SimpleNamespace(model=model, get_default_field=lambda name: defaults[name]),
  )
  return env, defaults, SceneEntityCfg("robot", geom_ids=[0, 1])


def _reward_env(commands, recovery=None):
  commands = torch.tensor(commands, dtype=torch.float32)
  num_envs = len(commands)
  data = SimpleNamespace(
    joint_pos=torch.zeros(num_envs, 3),
    default_joint_pos=torch.zeros(num_envs, 3),
    joint_vel=torch.zeros(num_envs, 3),
    site_lin_vel_w=torch.zeros(num_envs, 2, 3),
    projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1),
    gravity_vec_w=torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1),
    body_link_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(num_envs, 1, 1),
    body_link_ang_vel_w=torch.zeros(num_envs, 1, 3),
    root_link_ang_vel_b=torch.zeros(num_envs, 3),
  )
  manager = SimpleNamespace()
  if recovery is not None:
    manager._delay_env_mask = torch.tensor([item[0] for item in recovery])
    manager._delay_counters = torch.tensor([item[1] for item in recovery])
  return SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene={
      "robot": SimpleNamespace(data=data),
      "feet": SimpleNamespace(data=SimpleNamespace(found=torch.ones(num_envs, 2))),
    },
    command_manager=SimpleNamespace(get_command=lambda _: commands),
    termination_manager=manager,
    extras={"log": {}},
  )


class FootContactRandomizationTest(unittest.TestCase):
  def test_soft_surface_shared_by_feet_and_other_geoms_unchanged(self):
    env, defaults, asset_cfg = _contact_env()
    randomize_foot_contact(env, None, asset_cfg=asset_cfg, hard_ground_ratio=0.0)
    refs = env.sim.model.geom_solref
    imps = env.sim.model.geom_solimp
    torch.testing.assert_close(refs[:, 2], refs[:, 4])
    torch.testing.assert_close(imps[:, 2], imps[:, 4])
    self.assertTrue(torch.all((refs[:, 2, 0] >= 0.03) & (refs[:, 2, 0] <= 0.06)))
    self.assertTrue(torch.all((refs[:, 2, 1] >= 0.9) & (refs[:, 2, 1] <= 1.2)))
    torch.testing.assert_close(imps[:, 2], torch.tensor([[0.7, 0.95, 0.01, 0.5, 2.0]]).repeat(env.num_envs, 1))
    for field in defaults:
      torch.testing.assert_close(
        getattr(env.sim.model, field)[:, [0, 1, 3, 5]],
        defaults[field][[0, 1, 3, 5]].unsqueeze(0).repeat(env.num_envs, 1, 1),
      )

  def test_hard_surface_restores_nominal_values_after_soft_sample(self):
    env, defaults, asset_cfg = _contact_env()
    randomize_foot_contact(env, None, asset_cfg=asset_cfg, hard_ground_ratio=0.0)
    randomize_foot_contact(env, None, asset_cfg=asset_cfg, hard_ground_ratio=1.0)
    for field, default in defaults.items():
      torch.testing.assert_close(
        getattr(env.sim.model, field), default.unsqueeze(0).repeat(env.num_envs, 1, 1)
      )

  def test_only_selected_environments_are_changed(self):
    env, defaults, asset_cfg = _contact_env()
    randomize_foot_contact(env, torch.tensor([1, 5]), asset_cfg=asset_cfg, hard_ground_ratio=0.0)
    for field, default in defaults.items():
      values = getattr(env.sim.model, field)
      torch.testing.assert_close(values[[0, 2, 3, 4, 6, 7]], default.unsqueeze(0).repeat(6, 1, 1))
    self.assertTrue(torch.all(env.sim.model.geom_solref[[1, 5], 2, 0] >= 0.03))

  def test_invalid_sampling_ranges_rejected(self):
    invalid = (
      {"hard_ground_ratio": -0.1},
      {"hard_ground_ratio": 1.1},
      {"time_constant_range": (0.009, 0.06)},
      {"time_constant_range": (0.06, 0.03)},
      {"damping_ratio_range": (1.2, 0.9)},
      {"damping_ratio_range": (0.0, 1.0)},
    )
    for kwargs in invalid:
      with self.subTest(kwargs=kwargs):
        env, _, asset_cfg = _contact_env()
        with self.assertRaises(ValueError):
          randomize_foot_contact(env, None, asset_cfg=asset_cfg, **kwargs)

  def test_event_declares_per_environment_contact_fields(self):
    self.assertEqual(set(randomize_foot_contact.model_fields), {"geom_solref", "geom_solimp"})

  def test_empty_selection_is_noop_and_minimum_time_constant_is_allowed(self):
    env, defaults, asset_cfg = _contact_env()
    randomize_foot_contact(env, torch.tensor([], dtype=torch.long), asset_cfg=asset_cfg)
    for field, default in defaults.items():
      torch.testing.assert_close(getattr(env.sim.model, field), default.unsqueeze(0).repeat(env.num_envs, 1, 1))
    randomize_foot_contact(env, None, asset_cfg=asset_cfg,
                          time_constant_range=(0.01, 0.01), hard_ground_ratio=0.0)
    torch.testing.assert_close(env.sim.model.geom_solref[:, 2, 0], torch.full((env.num_envs,), 0.01))


class StandingRewardTest(unittest.TestCase):
  def test_slip_cost_applies_at_zero_command_and_only_during_contact(self):
    env = _reward_env([[0.0, 0.0, 0.0]] * 2)
    env.scene["robot"].data.site_lin_vel_w[:] = torch.tensor(
      [[[3.0, 4.0, 99.0], [0.0, 2.0, 99.0]], [[1.0, 0.0, 99.0], [3.0, 4.0, 99.0]]]
    )
    env.scene["feet"].data.found[:] = torch.tensor([[1.0, 0.0], [1.0, 2.0]])
    cost = contact_slip(env, sensor_name="feet", asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]))
    torch.testing.assert_close(cost, torch.tensor([25.0, 26.0]))
    self.assertTrue(env.extras["log"])

  def test_no_contact_gives_zero_cost_and_finite_log(self):
    env = _reward_env([[0.0, 0.0, 0.0]])
    env.scene["robot"].data.site_lin_vel_w.fill_(10.0)
    env.scene["feet"].data.found.zero_()
    cost = contact_slip(env, sensor_name="feet", asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]))
    torch.testing.assert_close(cost, torch.zeros(1))
    self.assertTrue(all(torch.isfinite(value).all() for value in env.extras["log"].values()))

  def test_standing_cost_ignores_walking_but_penalizes_joint_motion(self):
    env = _reward_env([[0.0, 0.0, 0.0], [0.0, 0.0, 0.05], [0.2, 0.0, 0.0], [0.0, 0.0, 0.2]])
    env.scene["robot"].data.joint_pos[:] = torch.tensor([1.0, 2.0, 99.0])
    env.scene["robot"].data.joint_vel[:] = torch.tensor([3.0, 4.0, 99.0])
    cost = standing_motion(env, command_name="twist", asset_cfg=SceneEntityCfg("robot", joint_ids=[0, 1]))
    torch.testing.assert_close(cost, torch.tensor([5.5, 5.5, 0.0, 0.0]))

  def test_standing_cost_tracks_deviation_from_nominal_pose(self):
    env = _reward_env([[0.0, 0.0, 0.0]])
    env.scene["robot"].data.default_joint_pos[:] = torch.tensor([1.0, 2.0, 3.0])
    env.scene["robot"].data.joint_pos[:] = torch.tensor([1.0, 2.0, 3.0])
    cost = standing_motion(env, command_name="twist", asset_cfg=SceneEntityCfg("robot"))
    torch.testing.assert_close(cost, torch.zeros(1))

  def test_recovery_mask_only_exempts_active_recovery(self):
    env = _reward_env([[0.0, 0.0, 0.0]] * 3, recovery=[(True, 2), (True, 0), (False, 2)])
    data = env.scene["robot"].data
    data.joint_pos.fill_(1.0)
    data.site_lin_vel_w[:, :, 0] = 1.0
    data.body_link_ang_vel_w[:, :, 0] = 1.0
    standing = standing_motion(env, command_name="twist", asset_cfg=SceneEntityCfg("robot"))
    slip = contact_slip(env, sensor_name="feet", asset_cfg=SceneEntityCfg("robot", site_ids=[0, 1]))
    torso = torso_stability(env, asset_cfg=SceneEntityCfg("robot", body_ids=[0]))
    torch.testing.assert_close(standing, torch.tensor([0.0, 3.0, 3.0]))
    torch.testing.assert_close(slip, torch.tensor([0.0, 2.0, 2.0]))
    torch.testing.assert_close(torso, torch.tensor([0.0, 0.1, 0.1]))

  def test_torso_penalty_excludes_turning_around_vertical_axis(self):
    env = _reward_env([[0.0, 0.0, 0.0]] * 2)
    data = env.scene["robot"].data
    data.body_link_ang_vel_w[:] = torch.tensor([[[0.0, 0.0, 5.0]], [[3.0, 4.0, 0.0]]])
    cost = torso_stability(env, asset_cfg=SceneEntityCfg("robot", body_ids=[0]))
    torch.testing.assert_close(cost, torch.tensor([0.0, 2.5]))

  def test_torso_tilt_is_computed_for_selected_body(self):
    env = _reward_env([[0.0, 0.0, 0.0]])
    # Torso tilted 90 degrees around X while pelvis/root gravity remains upright.
    env.scene["robot"].data.body_link_quat_w[:] = torch.tensor([[[2 ** -0.5, 2 ** -0.5, 0.0, 0.0]]])
    cost = torso_stability(env, asset_cfg=SceneEntityCfg("robot", body_ids=[0]))
    torch.testing.assert_close(cost, torch.ones(1))


class Hunter130RobustnessConfigTest(unittest.TestCase):
  def test_training_and_play_keep_contact_coverage(self):
    for task in ("Hunter130-Flat", "Hunter130-Rough", "Hunter130-AMP-Flat", "Hunter130-AMP-Rough"):
      for play in (False, True):
        with self.subTest(task=task, play=play):
          cfg = load_env_cfg(task, play=play)
          contact_events = [event for event in cfg.events.values() if event.func is randomize_foot_contact]
          self.assertEqual(len(contact_events), 1)
          self.assertEqual(contact_events[0].mode, "startup")
          self.assertEqual(cfg.commands["twist"].rel_standing_envs, 0.25)
          self.assertEqual(cfg.rewards["action_rate_l2"].weight, -0.05)
          self.assertIs(cfg.rewards["foot_slip"].func, contact_slip)
          self.assertNotIn("command_name", cfg.rewards["foot_slip"].params)
          self.assertEqual(cfg.observations["actor"].enable_corruption, not play)
          self.assertEqual("pd_gains" in cfg.events, not play)
          if not play:
            self.assertEqual(cfg.events["pd_gains"].mode, "startup")
            params = cfg.events["pd_gains"].params
            self.assertEqual(params["kp_range"], (0.9, 1.1))
            self.assertEqual(params["kd_range"], (0.85, 1.15))

  def test_command_latency_only_applies_to_training_and_is_synchronized(self):
    for play in (False, True):
      cfg = load_env_cfg("Hunter130-AMP-Flat", play=play)
      actuators = cfg.scene.entities["robot"].articulation.actuators
      self.assertTrue(all(isinstance(actuator, DelayedActuatorCfg) == (not play) for actuator in actuators))
      self.assertEqual("actuator_delay" in cfg.events, not play)
      if not play:
        self.assertEqual(cfg.events["actuator_delay"].mode, "reset")
        self.assertEqual(cfg.events["actuator_delay"].params["lag_range"], (0, 2))
        self.assertTrue(all(actuator.delay_hold_prob == 1.0 for actuator in actuators))

  def test_flat_actor_interface_is_preserved(self):
    dims = {"base_ang_vel": 3, "projected_gravity": 3, "command": 3,
            "phase": 2, "joint_pos": 25, "joint_vel": 25, "actions": 25}
    for task, expected in (("Hunter130-Flat", 86), ("Hunter130-AMP-Flat", 336)):
      cfg = load_env_cfg(task)
      actor = cfg.observations["actor"]
      self.assertEqual(sum(dims[name] for name in actor.terms) * actor.history_length, expected)

  def test_other_robots_do_not_receive_hunter130_contact_event(self):
    for task in ("Unitree-G1-Flat", "Unitree-G1-AMP-Flat", "Hunter180-Flat", "Hunter180-AMP-Flat"):
      cfg = load_env_cfg(task)
      self.assertFalse(any(event.func is randomize_foot_contact for event in cfg.events.values()))


if __name__ == "__main__":
  unittest.main()
