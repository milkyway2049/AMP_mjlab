"""Check Hunter180 motion ordering before the shared AMP loaders use indices."""

from pathlib import Path

import numpy as np

from src.tasks.amp_loco.mdp.events import init_motion_loader


def validate_motion_files(paths, joint_names, body_names) -> None:
  for path in paths:
    with np.load(path, allow_pickle=False) as data:
      for key, expected in (("joint_names", joint_names), ("body_names", body_names)):
        if key not in data or tuple(data[key].tolist()) != tuple(expected):
          raise ValueError(
            f"{path}: {key} must match Hunter180 model order. "
            "Run scripts/retarget_g1_to_hunter180.py or convert Hunter180-retargeted "
            "CSV with scripts/csv_to_npz.py --robot hunter180; "
            "G1 motions cannot be used directly."
          )
      frames = data["joint_pos"].shape[0]
      expected_shapes = {
        "joint_pos": (frames, len(joint_names)), "joint_vel": (frames, len(joint_names)),
        "body_pos_w": (frames, len(body_names), 3), "body_quat_w": (frames, len(body_names), 4),
        "body_lin_vel_w": (frames, len(body_names), 3), "body_ang_vel_w": (frames, len(body_names), 3),
      }
      if frames < 2:
        raise ValueError(f"{path}: at least two motion frames are required")
      for key, shape in expected_shapes.items():
        if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
          raise ValueError(f"{path}: {key} must contain finite values with shape {shape}")
      if "fps" not in data or data["fps"].size != 1 or not np.isfinite(data["fps"]).all() or float(data["fps"].item()) <= 0:
        raise ValueError(f"{path}: fps must be one positive finite number")


def init_hunter180_motion_loader(
  env, env_ids, motion_dir: str, recovery_dir: str | None = None,
  delay_reset_env_ratio: float = 0.0, max_delay_steps: int = 0,
) -> None:
  paths = sorted(Path(motion_dir).glob("*.npz"))
  if not paths:
    raise FileNotFoundError(
      f"No Hunter180 motion clips in {motion_dir}. First retarget motions to its "
      "23 joints with scripts/retarget_g1_to_hunter180.py, or run "
      "scripts/csv_to_npz.py --robot hunter180 with retargeted CSV. "
      "Hunter180-Flat/Rough can train without AMP motion data."
    )
  if recovery_dir is not None:
    paths += sorted(Path(recovery_dir).glob("*.npz"))
  robot = env.scene["robot"]
  validate_motion_files(paths, robot.joint_names, robot.body_names)
  init_motion_loader(
    env, env_ids, motion_dir=motion_dir, recovery_dir=recovery_dir,
    delay_reset_env_ratio=delay_reset_env_ratio, max_delay_steps=max_delay_steps,
  )
