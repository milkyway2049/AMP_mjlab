"""Compare position-only reference servo lag with fixed root and no gravity/contact.

This controlled test isolates the servos. It cannot evaluate floating-base
balance, foot contact forces, learned policy quality, or training convergence.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

if __package__:
  from .compare_xq180_pd import apply_gains, make_model
else:
  from compare_xq180_pd import apply_gains, make_model
from src.assets.robots.xq180.xq180_constants import get_xq180_robot_cfg, XQ180_JOINT_NAMES
from src.tasks.amp_loco.config.xq180.motion import XQ180_LOCOMOTION_DIR


def compare(parameter_files, motion_dir, output):
  model = make_model(get_xq180_robot_cfg(), True)
  profiles = [json.loads(p.read_text()) for p in parameter_files]
  names = [p["pd_profile"] for p in profiles]
  if len(set(names)) != len(names):
    raise ValueError("PD profile names must be unique")
  addresses = [model.joint(n).qposadr[0] for n in XQ180_JOINT_NAMES]
  rows = []
  paths = sorted(motion_dir.rglob("*.npz"))
  if not paths:
    raise FileNotFoundError(f"No motions in {motion_dir}")
  for path in paths:
    with np.load(path, allow_pickle=False) as saved:
      if tuple(saved["joint_names"].tolist()) != XQ180_JOINT_NAMES:
        raise ValueError(f"Joint ordering mismatch: {path}")
      q, dq = saved["joint_pos"], saved["joint_vel"]
      fps = float(saved["fps"].item())
      if not np.isclose(fps, 50):
        raise ValueError("This test uses 50 Hz targets and four 5 ms substeps")
      row = {"clip": str(path.relative_to(motion_dir)), "frames": len(q),
        "root_vertical_accel_p99": float(np.quantile(abs(np.gradient(saved["body_lin_vel_w"][:, 0, 2]) * fps), .99)),
        "profiles": {}}
      for profile in profiles:
        apply_gains(model, profile["joints"])
        data = mujoco.MjData(model)
        data.qpos[addresses] = q[0]
        data.qvel[:] = dq[0]
        error, saturated = [], []
        for frame in range(len(q)):
          data.ctrl[:] = q[frame]
          error.append(data.qpos[addresses] - q[frame])
          for _ in range(4):
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
              raise ValueError(f"Nonfinite servo response: {path}, frame {frame}")
            saturated.append(abs(data.actuator_force) >= .99 * model.actuator_forcerange[:, 1])
        error = np.asarray(error)
        row["profiles"][profile["pd_profile"]] = {
          "rmse_rad": float(np.sqrt(np.mean(error**2))),
          "p95_abs_error_rad": float(np.quantile(abs(error), .95)),
          "torque_saturation_fraction": float(np.mean(saturated)),
          "joint_rmse_rad": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        }
      rows.append(row)
      print(row["clip"], {n: round(r["rmse_rad"], 5) for n, r in row["profiles"].items()}, flush=True)
  summary = {}
  for category in ("WalkandRun", "Recovery"):
    chosen = [r for r in rows if motion_dir.name == category or Path(r["clip"]).parts[0] == category]
    if chosen:
      count = sum(r["frames"] for r in chosen)
      summary[category] = {n: float(np.sqrt(sum(r["frames"] * r["profiles"][n]["rmse_rad"]**2 for r in chosen) / count)) for n in names}
  report = {"note": __doc__, "profiles": names, "parameter_files": list(map(str, parameter_files)),
    "joint_names": XQ180_JOINT_NAMES, "summary_rmse_rad": summary, "motions": rows}
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--parameters", nargs="+", type=Path, default=[
    Path("docs/xq180_policy_io_user_pd_v1.json"),
    Path("docs/xq180_policy_io_g1_inertia_v1.json"),
    Path("docs/xq180_policy_io.json"),
  ])
  parser.add_argument("--motion-dir", type=Path, default=XQ180_LOCOMOTION_DIR)
  parser.add_argument("--output", type=Path, default=Path("docs/xq180_retarget/locomotion_servo_comparison.json"))
  args = parser.parse_args()
  compare(args.parameters, args.motion_dir, args.output)
