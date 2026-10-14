from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  hunter130_flat_env_cfg,
  hunter130_rough_env_cfg,
)
from .rl_cfg import hunter130_ppo_runner_cfg

register_mjlab_task(
  task_id="Hunter130-Rough",
  env_cfg=hunter130_rough_env_cfg(),
  play_env_cfg=hunter130_rough_env_cfg(play=True),
  rl_cfg=hunter130_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Hunter130-Flat",
  env_cfg=hunter130_flat_env_cfg(),
  play_env_cfg=hunter130_flat_env_cfg(play=True),
  rl_cfg=hunter130_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
