"""PPO/AMP configuration for XQ180's 19 actuated joints."""

from src.assets.robots.xq180.xq180_constants import (
  XQ180_AMP_ANCHOR_NAME, XQ180_AMP_BODY_NAMES, XQ180_JOINT_NAMES,
)
from src.tasks.amp_loco.config.g1.rl_cfg import RslRlAmpRunnerCfg, g1_amp_ppo_runner_cfg
from .motion import XQ180_LOCOMOTION_DIR


def xq180_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  cfg = g1_amp_ppo_runner_cfg()
  cfg.experiment_name = "xq180_amp_locomotion_no_recovery_userpd_v5_ampfix_v6"
  cfg.amp_motion_files = str(XQ180_LOCOMOTION_DIR)
  # Bound motor targets AND the previous-action observation/reward feedback.
  cfg.clip_actions = 3.0
  cfg.actor.distribution_cfg["init_std"] = 0.5
  # Fixed LR avoids the inherited adaptive scheduler's growth up to 1e-2.
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.schedule = "fixed"
  cfg.amp_learning_rate_decay_stages = [(6000, 1 / 3), (20000, 1 / 6)]
  cfg.amp_kl_stop_multiplier = 4.0
  cfg.max_iterations = 100000
  cfg.min_normalized_std = [0.05] * len(XQ180_JOINT_NAMES)
  cfg.amp_body_names = XQ180_AMP_BODY_NAMES
  cfg.amp_anchor_name = XQ180_AMP_ANCHOR_NAME
  return cfg
