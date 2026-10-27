"""Hunter130 training for standing and walking on compliant floor surfaces.

The contact parameters approximate a range of soft supports; they are not a
material model of a measured gym mat. Nominal robot PD and policy IO stay in the
robot configuration. Random PD and command latency apply only during training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.actuator import DelayedActuatorCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


@requires_model_fields("geom_solref", "geom_solimp")
def randomize_foot_contact(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  time_constant_range: tuple[float, float] = (0.03, 0.06),
  damping_ratio_range: tuple[float, float] = (0.9, 1.2),
  impedance: tuple[float, float, float, float, float] = (0.7, 0.95, 0.01, 0.5, 2.0),
  hard_ground_ratio: float = 0.4,
) -> None:
  """Sample one surface per world, shared by both feet; retain some hard floors.

  Foot geoms have priority 1, above terrain priority 0, so their contact solver
  parameters determine foot/ground compliance. Positive solref time constants
  must be at least twice the physics timestep. Each write starts from defaults
  rather than accumulating changes across calls.
  """
  if not 0.0 <= hard_ground_ratio <= 1.0:
    raise ValueError("hard_ground_ratio must be between 0 and 1")
  if not 2 * env.cfg.sim.mujoco.timestep <= time_constant_range[0] <= time_constant_range[1]:
    raise ValueError("Contact time constants must be ordered and >= 2 * physics timestep")
  if not 0 < damping_ratio_range[0] <= damping_ratio_range[1]:
    raise ValueError("Contact damping ratios must be positive and ordered")
  if not (len(impedance) == 5 and 0 < impedance[0] <= impedance[1] < 1
          and impedance[2] > 0 and 0 < impedance[3] < 1 and impedance[4] >= 1):
    raise ValueError("Invalid contact impedance parameters")

  ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
         else env_ids.to(device=env.device, dtype=torch.long))
  if ids.numel() == 0:
    return
  geom_ids = env.scene[asset_cfg.name].indexing.geom_ids[asset_cfg.geom_ids]
  ref = env.sim.model.geom_solref
  imp = env.sim.model.geom_solimp
  tau = torch.empty(len(ids), device=env.device, dtype=ref.dtype).uniform_(*time_constant_range)
  zeta = torch.empty_like(tau).uniform_(*damping_ratio_range)
  soft_ref = torch.stack((tau, zeta), dim=-1)[:, None, :]
  soft_imp = torch.tensor(impedance, device=env.device, dtype=imp.dtype)[None, None, :]
  hard = (torch.rand(len(ids), device=env.device) < hard_ground_ratio)[:, None, None]
  ref[ids[:, None], geom_ids[None, :]] = torch.where(
    hard, env.sim.get_default_field("geom_solref")[geom_ids][None], soft_ref
  )
  imp[ids[:, None], geom_ids[None, :]] = torch.where(
    hard, env.sim.get_default_field("geom_solimp")[geom_ids][None], soft_imp
  )


def _mask_recovery(env: ManagerBasedRlEnv, cost: torch.Tensor) -> torch.Tensor:
  """Do not constrain AMP's deliberately delayed get-up attempts."""
  manager = env.termination_manager
  mask = getattr(manager, "_delay_env_mask", None)
  counters = getattr(manager, "_delay_counters", None)
  if isinstance(mask, torch.Tensor) and isinstance(counters, torch.Tensor):
    cost = torch.where(mask & (counters > 0), torch.zeros_like(cost), cost)
  return cost


def contact_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize support-foot sliding during standing as well as locomotion."""
  robot = env.scene[asset_cfg.name]
  found = env.scene[sensor_name].data.found
  assert found is not None
  contact = (found > 0).to(robot.data.site_lin_vel_w.dtype)
  velocity = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  speed_sq = velocity.square().sum(dim=-1)
  env.extras["log"]["Metrics/slip_velocity_mean"] = (
    (speed_sq.sqrt() * contact).sum() / contact.sum().clamp(min=1)
  )
  return _mask_recovery(env, (speed_sq * contact).sum(dim=-1))


def standing_motion(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  command_threshold: float = 0.1,
  velocity_weight: float = 0.02,
) -> torch.Tensor:
  """Discourage pose drift and joint motion only when commanded to stand."""
  robot = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  standing = command[:, :2].norm(dim=-1) + command[:, 2].abs() <= command_threshold
  error = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  velocity = robot.data.joint_vel[:, asset_cfg.joint_ids]
  cost = error.square().sum(dim=-1) + velocity_weight * velocity.square().sum(dim=-1)
  return _mask_recovery(env, cost * standing)


def torso_stability(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  angular_velocity_weight: float = 0.1,
) -> torch.Tensor:
  """Penalize torso tilt and roll/pitch motion, permitting commanded yaw."""
  robot = env.scene[asset_cfg.name]
  body_id = asset_cfg.body_ids[0]
  orientation = robot.data.body_link_quat_w[:, body_id]
  gravity = quat_apply_inverse(orientation, robot.data.gravity_vec_w)
  omega = quat_apply_inverse(orientation, robot.data.body_link_ang_vel_w[:, body_id])
  cost = gravity[:, :2].square().sum(dim=-1) + angular_velocity_weight * omega[:, :2].square().sum(dim=-1)
  return _mask_recovery(env, cost)


def configure_hunter130_robustness(
  cfg: ManagerBasedRlEnvCfg, *, play: bool = False, amp: bool = False,
) -> None:
  """Apply shared Hunter130 changes without changing observation/action shapes."""
  cfg.commands["twist"].rel_standing_envs = 0.25
  cfg.events["foot_contact"] = EventTermCfg(
    func=randomize_foot_contact,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=("left_foot_collision", "right_foot_collision")),
      "hard_ground_ratio": 0.0 if play else 0.4,
    },
  )
  # Allow a longer settled stance between moderate horizontal perturbations.
  cfg.events["push_robot"].interval_range_s = (4.0, 8.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.3, 0.3), "y": (-0.25, 0.25), "yaw": (-0.35, 0.35),
  }
  cfg.rewards["foot_slip"] = RewardTermCfg(
    func=contact_slip, weight=-0.5,
    params={"sensor_name": "feet_ground_contact",
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
  )
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=standing_motion, weight=-1.0,
    params={"command_name": "twist", "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["action_rate_l2"].weight = -0.05
  if amp:
    cfg.rewards["torso_stability"] = RewardTermCfg(
      func=torso_stability, weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    )
    cfg.rewards["body_ang_vel_xy_l2"].params["std"] = 1.0
  else:
    cfg.rewards["body_ang_vel"].weight = -0.1

  if not play:
    cfg.events["pd_gains"] = EventTermCfg(
      func=dr.pd_gains, mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot"), "kp_range": (0.9, 1.1),
              "kd_range": (0.85, 1.15), "operation": "scale"},
    )
    # One coherent 0/5/10 ms command delay per world, held for an episode.
    articulation = cfg.scene.entities["robot"].articulation
    assert articulation is not None
    articulation.actuators = tuple(
      DelayedActuatorCfg(base_cfg=actuator, delay_min_lag=0, delay_max_lag=2,
                         delay_hold_prob=1.0)
      for actuator in articulation.actuators
    )
    cfg.events["actuator_delay"] = EventTermCfg(
      func=dr.sync_actuator_delays, mode="reset",
      params={"asset_cfg": SceneEntityCfg("robot"), "lag_range": (0, 2)},
    )
