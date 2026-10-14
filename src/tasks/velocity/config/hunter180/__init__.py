from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  hunter180_flat_env_cfg,
  hunter180_rough_env_cfg,
)
from .rl_cfg import hunter180_ppo_runner_cfg

register_mjlab_task(
  task_id="Hunter180-Rough",
  env_cfg=hunter180_rough_env_cfg(),
  play_env_cfg=hunter180_rough_env_cfg(play=True),
  rl_cfg=hunter180_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Hunter180-Flat",
  env_cfg=hunter180_flat_env_cfg(),
  play_env_cfg=hunter180_flat_env_cfg(play=True),
  rl_cfg=hunter180_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
