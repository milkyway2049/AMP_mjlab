"""PPO/AMP configuration for Hunter180's 23 actuated joints."""

from src import SRC_PATH
from src.assets.robots.hunter180.hunter180_constants import (
  HUNTER180_AMP_ANCHOR_NAME, HUNTER180_AMP_BODY_NAMES, HUNTER180_JOINT_NAMES,
)
from src.tasks.amp_loco.config.g1.rl_cfg import RslRlAmpRunnerCfg, g1_amp_ppo_runner_cfg


def hunter180_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  cfg = g1_amp_ppo_runner_cfg()
  cfg.experiment_name = "hunter180_amp_locomotion"
  cfg.amp_motion_files = str(SRC_PATH / "assets/motions/hunter180/amp")
  cfg.min_normalized_std = [0.05] * len(HUNTER180_JOINT_NAMES)
  cfg.amp_body_names = HUNTER180_AMP_BODY_NAMES
  cfg.amp_anchor_name = HUNTER180_AMP_ANCHOR_NAME
  return cfg
