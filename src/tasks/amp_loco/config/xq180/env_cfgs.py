"""XQ180 AMP Locomotion environment configurations."""

from src.assets.robots.xq180.xq180_constants import (
  XQ180_AMP_BODY_NAMES, XQ180_AMP_ANCHOR_NAME,
)
from src.assets.robots import (
  XQ180_ACTION_SCALE,
  get_xq180_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
from .motion import XQ180_LOCOMOTION_DIR, init_xq180_motion_loader
from .mdp import fallen_height, locomotion_height_reward

def xq180_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create XQ180 rough terrain velocity configuration."""
  cfg = make_amp_env_cfg()

  # Keep CCD high enough for stability but avoid Warp OOM from excessive EPA buffers.
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = 48
  cfg.sim.mujoco.iterations = 20

  cfg.scene.entities = {"robot": get_xq180_robot_cfg()}

  # Set raycast sensor frame to XQ180 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  site_names = ("left_foot", "right_foot")
  geom_names = ("left_foot_collision", "right_foot_collision")
  body_names = XQ180_AMP_BODY_NAMES
  anchor_name = XQ180_AMP_ANCHOR_NAME
  root_name = "base_link"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_Link|right_ankle_Link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = XQ180_ACTION_SCALE

  cfg.viewer.body_name = "waist_Link"
  cfg.terminations["bad_base_height"].func = fallen_height
  cfg.terminations["bad_base_height"].params["minimum_height"] = 0.75
  cfg.terminations["bad_orientation"].params["limit_angle"] = 0.8726646259971648  # 50 degrees

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.5
  # Start within a modest walking envelope; grow only after learning balance.
  twist_cmd.ranges.lin_vel_x = (-0.3, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  twist_cmd.resampling_time_range = (4.0, 8.0)
  twist_cmd.rel_standing_envs = 0.1
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-0.3, 0.8), "lin_vel_y": (-0.2, 0.2), "ang_vel_z": (-0.5, 0.5)},
    {"step": 2000 * 24, "lin_vel_x": (-0.5, 1.2), "lin_vel_y": (-0.3, 0.3), "ang_vel_z": (-0.75, 0.75)},
    {"step": 6000 * 24, "lin_vel_x": (-0.8, 1.8), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
  ]

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_Link",)
  cfg.events["foot_friction"].params["ranges"] = (0.5, 1.0)
  cfg.events["base_com"].params["ranges"] = {axis: (-0.01, 0.01) for axis in range(3)}
  cfg.events["push_robot"].interval_range_s = (6.0, 10.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.3, 0.3), "y": (-0.2, 0.2), "z": (0.0, 0.0),
    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
  }
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.max_init_terrain_level = 1

  # Fall -> immediate reset to an upright locomotion reference, never recovery.
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 0

  # Set motion data path for startup loader and reset.
  _motion_dir = str(XQ180_LOCOMOTION_DIR)

  cfg.events["init_motion_loader"].func = init_xq180_motion_loader
  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = None
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)
  # The inherited height reward was masked to recovery-only environments.
  cfg.rewards["track_root_height"] = RewardTermCfg(
    func=locomotion_height_reward, weight=0.5,
    params={"desired_height": 1.05, "std": 0.12},
  )
  for name in ("track_anchor_linear_velocity", "track_anchor_angular_velocity", "body_ang_vel_xy_l2"):
    cfg.rewards[name].params["mask_delay"] = False

  cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

  cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = body_names



  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def xq180_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create XQ180 flat terrain velocity configuration."""
  cfg = xq180_amp_rough_env_cfg(play=play)

  cfg.sim.njmax = 640
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  return cfg
