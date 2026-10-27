"""Audit G1-to-XQ180 reference quality and derive G1-based inertia gain ratios.

Uses MuJoCo forward kinematics/contact detection, not a policy rollout. Sole-site
speed near the floor is a sliding proxy, not a measured physical contact speed.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

if __package__:
  from .retarget_g1_to_xq180 import G1ToXQ180
else:
  from retarget_g1_to_xq180 import G1ToXQ180
from src import SRC_PATH


def pose_metrics(model, data, root_pos, root_quat, joint_pos, fps):
  feet = []
  contact_frames = 0
  deepest = 0.0
  pairs = Counter()
  for pos, quat, q in zip(root_pos, root_quat, joint_pos):
    data.qpos[:] = np.r_[pos, quat, q]
    mujoco.mj_forward(model, data)
    feet.append(data.site_xpos[[model.site(n).id for n in ('left_foot', 'right_foot')]].copy())
    penetrating = [c for c in data.contact if c.dist < -0.001]
    contact_frames += bool(penetrating)
    for contact in penetrating:
      deepest = min(deepest, float(contact.dist))
      pair = sorted((model.geom(contact.geom1).name, model.geom(contact.geom2).name))
      pairs[' / '.join(pair)] += 1
  feet = np.asarray(feet)
  speed = np.linalg.norm(np.gradient(feet[:, :, :2], axis=0) * fps, axis=-1)
  near_floor = feet[:, :, 2] < 0.035
  return {
    'near_floor_site_speed_mean_m_s': float(speed[near_floor].mean()) if near_floor.any() else None,
    'near_floor_site_samples': int(near_floor.sum()),
    'self_penetration_over_1mm_frame_count': contact_frames,
    'self_penetration_over_1mm_frame_fraction': contact_frames / len(joint_pos),
    'deepest_self_penetration_m': deepest,
    'self_contact_pairs': dict(pairs.most_common(8)),
  }


def inertia_ratios(converter):
  """Joint inertia with own-side sole fixed by eliminating floating-base DOFs.

  Ignores double-support constraints and actuator coupling. Ratios are local
  model-based tuning evidence; they are not identified motor inertias.
  """
  values = []
  for model, data in ((converter.source_model, converter.source_data),
                      (converter.target_model, converter.target_data)):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    stance = []
    for side in ('left', 'right'):
      pos_jac, rot_jac = np.empty((3, model.nv)), np.empty((3, model.nv))
      mujoco.mj_jacSite(model, data, pos_jac, rot_jac, model.site(side + '_foot').id)
      jac = np.r_[pos_jac, rot_jac]
      transform = np.r_[-np.linalg.solve(jac[:, :6], jac[:, 6:]), np.eye(model.nv - 6)]
      stance.append(np.diag(transform.T @ mass @ transform))
    values.append((np.diag(mass)[6:], stance))
  rows = []
  for name in converter.target.joint_names:
    source = converter.joint_mapping[name]
    si = converter.source_model.joint(source).dofadr[0] - 6
    ti = converter.target_model.joint(name).dofadr[0] - 6
    side = int(name.startswith('right'))
    rows.append({
      'joint': name, 'g1_joint': source,
      'fixed_root_inertia_ratio': float(values[1][0][ti] / values[0][0][si]),
      'single_support_inertia_ratio': float(values[1][1][side][ti] / values[0][1][side][si]),
    })
  return rows


def analyze(motion_dir, output_dir):
  converter = G1ToXQ180()
  reports = []
  for path in sorted(motion_dir.rglob('*.npz')):
    relative = path.relative_to(motion_dir)
    source_path = SRC_PATH / 'assets/motions/g1/amp' / relative
    with np.load(path, allow_pickle=False) as target, np.load(source_path, allow_pickle=False) as source:
      q = target['joint_pos']; fps = float(target['fps'].item()); frames = len(q)
      source_names = tuple(source['joint_names'].tolist()) if 'joint_names' in source else converter.source.joint_names
      target_names = tuple(target['joint_names'].tolist())
      assert target_names == tuple(converter.target.joint_names), path
      indices = [source_names.index(converter.joint_mapping[n]) for n in target_names]
      raw = source['joint_pos'][:, indices] + np.array([converter.joint_offsets.get(n, 0) for n in target_names])
      clipped = abs(raw - q) > 1e-6
      lower, upper = converter.limits.T
      middle, half = (lower + upper) / 2, (upper - lower) * 0.45
      soft_changed = (q < middle - half - 1e-7) | (q > middle + half + 1e-7)
      target_metrics = pose_metrics(converter.target_model, converter.target_data,
        target['body_pos_w'][:, 0], target['body_quat_w'][:, 0], q, fps)
      g1_indices = [source_names.index(n) for n in converter.source.joint_names]
      source_metrics = pose_metrics(converter.source_model, converter.source_data,
        source['body_pos_w'][:, 0], source['body_quat_w'][:, 0], source['joint_pos'][:, g1_indices], fps)
      entry = {
        'clip': str(relative), 'frames': frames, 'fps': fps,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'hard_clipped_joint_samples': int(clipped.sum()),
        'hard_clipped_frame_fraction': float(clipped.any(axis=1).mean()),
        'hard_clipped_fraction_by_joint': {n: float(clipped[:, i].mean()) for i, n in enumerate(target_names) if clipped[:, i].any()},
        'reset_soft_limit_changed_frame_fraction': float(soft_changed.any(axis=1).mean()),
        'joint_speed_max_rad_s': float(abs(target['joint_vel']).max()),
        'g1': source_metrics, 'xq180': target_metrics,
      }
      for label, data in (('g1', source), ('xq180', target)):
        entry[label]['root_vertical_accel_p99_m_s2'] = float(np.quantile(abs(np.gradient(data['body_lin_vel_w'][:, 0, 2]) * fps), .99))
      reports.append(entry)
      print(relative, 'clip frames', round(entry['hard_clipped_frame_fraction'], 3),
            'self penetration frames', round(target_metrics['self_penetration_over_1mm_frame_fraction'], 3), flush=True)
  if not reports:
    raise FileNotFoundError(f'No motion files: {motion_dir}')
  total = sum(r['frames'] for r in reports)
  report = {
    'motion_dir': str(motion_dir), 'frames': total, 'clips': len(reports),
    'method': 'All-frame MuJoCo FK/self-contact check; penetration threshold 1mm. Near-floor sole-site horizontal speed uses z<35mm and is only a slip proxy. No policy rollout or physical contact simulation.',
    'summary': {
      'hard_clipped_joint_samples': sum(r['hard_clipped_joint_samples'] for r in reports),
      'hard_clipped_frame_fraction': sum(r['frames'] * r['hard_clipped_frame_fraction'] for r in reports) / total,
      'reset_soft_limit_changed_frame_fraction': sum(r['frames'] * r['reset_soft_limit_changed_frame_fraction'] for r in reports) / total,
      'self_penetration_over_1mm_frame_fraction': sum(r['xq180']['self_penetration_over_1mm_frame_count'] for r in reports) / total,
    },
    'g1_total_mass_kg': float(converter.source_model.body_mass.sum()),
    'xq180_total_mass_kg': float(converter.target_model.body_mass.sum()),
    'nominal_inertia_ratios': inertia_ratios(converter), 'motions': reports,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / 'motion_audit.json').write_text(json.dumps(report, indent=2) + '\n')
  plot_report(reports, output_dir / 'motion_audit.png')
  return report


def plot_report(reports, output):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  x = np.arange(len(reports)); fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
  axes[0].bar(x, [100*r['hard_clipped_frame_fraction'] for r in reports], color='#b45309')
  axes[0].set_ylabel('Hard-clipped frames (%)')
  for ax, key, label in ((axes[1], 'near_floor_site_speed_mean_m_s', 'Sole-site speed (m/s)\nz < 35 mm; proxy'),
                         (axes[2], 'root_vertical_accel_p99_m_s2', 'Root vertical accel\np99 (m/s²)')):
    for shift, robot, color in ((-.18, 'g1', '#3974a7'), (.18, 'xq180', '#d66842')):
      ax.bar(x+shift, [r[robot][key] or 0 for r in reports], .36, label=robot.upper(), color=color)
    ax.set_ylabel(label); ax.legend()
  axes[3].bar(x, [100*r['xq180']['self_penetration_over_1mm_frame_fraction'] for r in reports], color='#b45309')
  axes[3].set_ylabel('XQ self-penetration\nframes >1mm (%)')
  axes[3].set_xticks(x, [Path(r['clip']).stem for r in reports], rotation=65, ha='right', fontsize=8)
  for ax in axes: ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
  fig.suptitle('Original XQ180 reference motion audit: kinematics, not a policy evaluation', fontsize=14)
  fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--motion-dir', type=Path, default=SRC_PATH / 'assets/motions/xq180/amp')
  parser.add_argument('--output-dir', type=Path, default=Path('docs/xq180_audit'))
  args = parser.parse_args()
  analyze(args.motion_dir, args.output_dir)
