"""PPO/AMP configuration for Hunter130's 25 actuated joints."""

from src import SRC_PATH
from src.assets.robots.hunter130.hunter130_constants import (
  HUNTER130_AMP_ANCHOR_NAME, HUNTER130_AMP_BODY_NAMES, HUNTER130_JOINT_NAMES,
)
from src.tasks.amp_loco.config.g1.rl_cfg import RslRlAmpRunnerCfg, g1_amp_ppo_runner_cfg


def hunter130_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  cfg = g1_amp_ppo_runner_cfg()
  cfg.experiment_name = "hunter130_amp_locomotion"
  cfg.amp_motion_files = str(SRC_PATH / "assets/motions/hunter130/amp")
  cfg.min_normalized_std = [0.05] * len(HUNTER130_JOINT_NAMES)
  cfg.amp_body_names = HUNTER130_AMP_BODY_NAMES
  cfg.amp_anchor_name = HUNTER130_AMP_ANCHOR_NAME
  return cfg
