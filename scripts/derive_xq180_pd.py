"""Derive integer XQ180 PD using actual kinematic chains and locomotion poses.

Scale Kp and Kd by the same inertia ratio to preserve the local second-order
frequency and damping ratio, then round to nearest integer (minimum one).
Use fixed-root inertia for swing/upper-body joints and own-foot-constrained
inertia for stance-leg joints. Select P90 of paired inertia ratios over evenly
sampled frames in every walking/running clip. This is a model estimate, not
motor identification, contact dynamics, or a demonstrated balance controller.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

if __package__:
  from .analyze_xq180 import inertia_ratios
  from .retarget_g1_to_xq180 import G1ToXQ180
else:
  from analyze_xq180 import inertia_ratios
  from retarget_g1_to_xq180 import G1ToXQ180
from src.assets.robots.xq180.xq180_constants import g1_reference_actuator
from src import SRC_PATH

_GROUPS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle", "waist",
           "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")


def joint_group(name):
  for group in _GROUPS:
    if group in name:
      return group
  raise ValueError(f"Unknown XQ180 joint: {name}")


def configuration_inertias(model, data):
  """M diagonal with base fixed, and T.T M T with each sole fixed.

  Each diagonal holds other actuated joint velocities at zero; full spatial
  inertia/parallel-axis effects follow the model's real parent chain and axes.
  """
  mujoco.mj_forward(model, data)
  mass = np.empty((model.nv, model.nv))
  mujoco.mj_fullM(model, data, mass)
  result = {"swing": np.diag(mass)[6:].copy()}
  for side in ("left", "right"):
    pos, rot = np.empty((3, model.nv)), np.empty((3, model.nv))
    mujoco.mj_jacSite(model, data, pos, rot, model.site(side + "_foot").id)
    jac = np.r_[pos, rot]
    transform = np.r_[-np.linalg.solve(jac[:, :6], jac[:, 6:]), np.eye(model.nv - 6)]
    result[side] = np.diag(transform.T @ mass @ transform)
  return result


def joint_chain(model, data, name):
  joint = model.joint(name)
  body = int(model.jnt_bodyid[joint.id])
  chain = []
  while body:
    start, count = model.body_jntadr[body], model.body_jntnum[body]
    chain[:0] = [model.joint(i).name for i in range(start, start + count)
                 if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    body = int(model.body_parentid[body])
  return {"root_to_joint": chain, "axis_in_joint_frame": joint.axis.tolist(),
    "nominal_axis_world": data.xaxis[joint.id].tolist(),
    "downstream_mass_kg": float(model.body_subtreemass[model.jnt_bodyid[joint.id]])}


def pose_layout(model, saved, entity):
  """Bind each library's named columns to its own model, never G1 indices."""
  joints = tuple(saved["joint_names"].tolist()) if "joint_names" in saved else tuple(entity.joint_names)
  bodies = tuple(saved["body_names"].tolist()) if "body_names" in saved else tuple(entity.body_names)
  if len(set(joints)) != len(joints) or set(joints) != set(entity.joint_names):
    raise ValueError("Motion joints do not match the corresponding model")
  if len(set(bodies)) != len(bodies) or set(bodies) != set(entity.body_names):
    raise ValueError("Motion bodies do not match the corresponding model")
  free = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).item())
  root = bodies.index(model.body(model.jnt_bodyid[free]).name)
  return [model.joint(n).qposadr[0] for n in joints], root


def derive(samples_per_clip=32):
  if samples_per_clip < 2:
    raise ValueError("At least two samples per clip are required")
  converter = G1ToXQ180()
  rows = inertia_ratios(converter)
  for row in rows:
    row["g1_chain"] = joint_chain(converter.source_model, converter.source_data, row["g1_joint"])
    row["xq180_chain"] = joint_chain(converter.target_model, converter.target_data, row["joint"])
  sampled = {group: [] for group in _GROUPS}
  modes = {group: {"stance": 0, "swing_or_upper_body": 0} for group in _GROUPS}
  files = sorted((SRC_PATH / "assets/motions/xq180/amp_ik/WalkandRun").glob("*.npz"))
  sample_count = 0
  for path in files:
    source_path = SRC_PATH / "assets/motions/g1/amp/WalkandRun" / path.name
    with np.load(path, allow_pickle=False) as target, np.load(source_path, allow_pickle=False) as source:
      if len(target["joint_pos"]) != len(source["joint_pos"]):
        raise ValueError(f"Frame mismatch: {path}")
      source_layout = pose_layout(converter.source_model, source, converter.source)
      target_layout = pose_layout(converter.target_model, target, converter.target)
      for frame in np.unique(np.linspace(0, len(target["joint_pos"]) - 1, samples_per_clip).astype(int)):
        for data, saved, (addresses, root) in (
            (converter.source_data, source, source_layout),
            (converter.target_data, target, target_layout)):
          data.qpos[:3] = saved["body_pos_w"][frame, root]
          data.qpos[3:7] = saved["body_quat_w"][frame, root]
          data.qpos[addresses] = saved["joint_pos"][frame]
          data.qvel[:] = 0
        src = configuration_inertias(converter.source_model, converter.source_data)
        tgt = configuration_inertias(converter.target_model, converter.target_data)
        paired = {group: [] for group in _GROUPS}
        for row in rows:
          name, group = row["joint"], joint_group(row["joint"])
          side = "right" if name.startswith("right") else "left"
          leg = group in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
          stance = leg and bool(target["foot_contact"][frame, int(side == "right")])
          mode = side if stance else "swing"
          si = converter.source_model.joint(row["g1_joint"]).dofadr[0] - 6
          ti = converter.target_model.joint(name).dofadr[0] - 6
          paired[group].append(float(tgt[mode][ti] / src[mode][si]))
          modes[group]["stance" if stance else "swing_or_upper_body"] += 1
        for group, values in paired.items():
          sampled[group].append(float(np.mean(values)))
        sample_count += 1
  gains = {}
  for group, values in sampled.items():
    if not values:
      raise ValueError("No locomotion poses available")
    reference = g1_reference_actuator(next(r["joint"] for r in rows if joint_group(r["joint"]) == group))
    ratio = float(np.quantile(values, .9))
    kp, kd = ratio * reference.stiffness, ratio * reference.damping
    gains[group] = {
      "inertia_ratio_p10_p50_p90": np.quantile(values, [.1, .5, .9]).tolist(),
      "selected_inertia_ratio": ratio, "samples_by_mode": modes[group],
      "g1_kp": reference.stiffness, "g1_kd": reference.damping,
      "unrounded_kp": kp, "unrounded_kd": kd,
      "kp": max(1, int(np.floor(kp + .5))),
      "kd": max(1, int(np.floor(kd + .5))),
    }
  return {"pd_profile": "g1_chain_integer_v4", "method": __doc__,
    "symmetry": "Mean left/right ratio at each pose, then P90 across poses, then round; minimum gain 1",
    "clips": len(files), "sampled_poses": sample_count, "samples_per_clip": samples_per_clip,
    "limitations": "Other joint speeds held zero, ideal own-foot constraint, no double-support/contact-force solve. XQ180 armature=0.01 is an assumption. P90 is a conservative tuning choice, not an optimum.",
    "groups": gains, "joint_inertia_ratios": rows}


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=Path("docs/xq180_audit/g1_chain_integer_pd.json"))
  parser.add_argument("--samples-per-clip", type=int, default=32)
  args = parser.parse_args()
  report = derive(args.samples_per_clip)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report["groups"], indent=2))
