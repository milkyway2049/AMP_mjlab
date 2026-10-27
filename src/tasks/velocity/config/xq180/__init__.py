from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  xq180_flat_env_cfg,
  xq180_rough_env_cfg,
)
from .rl_cfg import xq180_ppo_runner_cfg

register_mjlab_task(
  task_id="XQ180-Rough",
  env_cfg=xq180_rough_env_cfg(),
  play_env_cfg=xq180_rough_env_cfg(play=True),
  rl_cfg=xq180_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="XQ180-Flat",
  env_cfg=xq180_flat_env_cfg(),
  play_env_cfg=xq180_flat_env_cfg(play=True),
  rl_cfg=xq180_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
