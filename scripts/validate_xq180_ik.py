"""Compare the saved contact-IK library with the original XQ180 references."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

if __package__:
  from .analyze_xq180 import pose_metrics
  from .xq180_retarget_ik import G1ToXQ180IK
else:
  from analyze_xq180 import pose_metrics
  from xq180_retarget_ik import G1ToXQ180IK
from src import SRC_PATH
from src.tasks.amp_loco.config.xq180.motion import validate_motion_files


def validate(output_dir, render=False):
  c = G1ToXQ180IK()
  root = SRC_PATH/'assets/motions/xq180/amp_ik'
  legacy_root = SRC_PATH/'assets/motions/xq180/amp'
  baseline = json.loads(Path('docs/xq180_audit/motion_audit.json').read_text())
  old = {r['clip']: r for r in baseline['motions']}
  rows = []
  paths = sorted(root.rglob('*.npz'))
  validate_motion_files(paths, c.target.joint_names, c.target.body_names)
  for path in paths:
    relative = str(path.relative_to(root))
    before = old[relative]
    if hashlib.sha256((legacy_root/relative).read_bytes()).hexdigest() != before['sha256']:
      raise ValueError(f'Baseline changed; re-run analyze_xq180.py: {relative}')
    with np.load(path, allow_pickle=False) as data:
      q = data['joint_pos']; fps = float(data['fps'].item())
      metrics = pose_metrics(c.target_model, c.target_data, data['body_pos_w'][:, 0],
                             data['body_quat_w'][:, 0], q, fps)
      metrics['root_vertical_accel_p99_m_s2'] = float(np.quantile(
        abs(np.gradient(data['body_lin_vel_w'][:, 0, 2])*fps), .99))
      soft_violation = (q < c.soft_lower-1e-6) | (q > c.soft_upper+1e-6)
      metrics['reset_soft_limit_changed_frame_fraction'] = float(soft_violation.any(axis=1).mean())
      metrics['max_joint_speed_rating_ratio'] = float((abs(data['joint_vel'])/c.velocity_limits).max())
      metrics['min_ground_clearance_m'] = float(data['ground_clearance'].min())
      for key in ('body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w', 'joint_pos', 'joint_vel'):
        if not np.isfinite(data[key]).all(): raise ValueError(f'Nonfinite {key}: {path}')
      self_frames = metrics['self_penetration_over_1mm_frame_count']
      if soft_violation.any() or metrics['max_joint_speed_rating_ratio'] > 1.001 or metrics['min_ground_clearance_m'] < .0019:
        raise ValueError(f'Invalid reset/speed/ground geometry: {relative}')
      feet = []
      for frame in range(len(q)):
        c.target_data.qpos[:] = np.r_[data['body_pos_w'][frame, 0], data['body_quat_w'][frame, 0], q[frame]]
        mujoco.mj_kinematics(c.target_model, c.target_data)
        feet.append(c.target_data.site_xpos[[c.target_model.site(n).id for n in ('left_foot', 'right_foot')]].copy())
      error = np.linalg.norm(np.asarray(feet)-data['foot_target_w'], axis=-1)
      contact_error = error[data['foot_contact']]
      metrics['contact_goal_error_mean_m'] = float(contact_error.mean()) if len(contact_error) else None
      metrics['contact_goal_error_p95_m'] = float(np.quantile(contact_error, .95)) if len(contact_error) else None
      metrics['contact_goal_samples'] = len(contact_error)
      rows.append({'clip': relative, 'frames': len(q), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'legacy': {**before['xq180'], 'reset_soft_limit_changed_frame_fraction': before['reset_soft_limit_changed_frame_fraction']},
        'ik': metrics})
      print(relative, 'self frames', self_frames, 'speed ratio', round(metrics['max_joint_speed_rating_ratio'], 3), flush=True)
  frames = sum(r['frames'] for r in rows)
  summary = {}
  for version in ('legacy', 'ik'):
    near_count = sum(r[version]['near_floor_site_samples'] for r in rows)
    summary[version] = {
      'self_penetration_over_1mm_frame_fraction': sum(r[version]['self_penetration_over_1mm_frame_count'] for r in rows)/frames,
      'reset_soft_limit_changed_frame_fraction': sum(r['frames']*r[version]['reset_soft_limit_changed_frame_fraction'] for r in rows)/frames,
      'near_floor_site_speed_mean_m_s': sum((r[version]['near_floor_site_speed_mean_m_s'] or 0)*r[version]['near_floor_site_samples'] for r in rows)/near_count,
    }
  summary['ik']['max_joint_speed_rating_ratio'] = max(r['ik']['max_joint_speed_rating_ratio'] for r in rows)
  summary['ik']['min_ground_clearance_m'] = min(r['ik']['min_ground_clearance_m'] for r in rows)
  contact_count = sum(r['ik']['contact_goal_samples'] for r in rows)
  summary['ik']['contact_goal_error_mean_m'] = sum((r['ik']['contact_goal_error_mean_m'] or 0)*r['ik']['contact_goal_samples'] for r in rows)/contact_count
  report = {'method': c.method, 'clips': len(rows), 'frames': frames,
    'note': 'All-frame saved-data kinematic audit. Near-floor (site z<35mm) speed is a slip proxy, not measured physical contact slip. No dynamics/policy validation.',
    'summary': summary, 'motions': rows}
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
  plot_comparison(rows, output_dir/'comparison.png')
  if render:
    render_preview(c, legacy_root, root, output_dir)
  print(json.dumps(summary, indent=2), flush=True)
  return report


def plot_comparison(rows, output):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
  x = np.arange(len(rows))
  for ax, key, label, scale in (
      (axes[0], 'self_penetration_over_1mm_frame_fraction', 'Self penetration >1 mm (%)', 100),
      (axes[1], 'near_floor_site_speed_mean_m_s', 'Near-floor sole-site speed (m/s)\nproxy only', 1),
      (axes[2], 'root_vertical_accel_p99_m_s2', 'Vertical root acceleration P99\n(m/s²)', 1)):
    for shift, version, color in ((-.18, 'legacy', '#d66842'), (.18, 'ik', '#318264')):
      ax.bar(x+shift, [(r[version][key] or 0)*scale for r in rows], .36, label=version, color=color)
    ax.set_ylabel(label); ax.legend(); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
  axes[-1].set_xticks(x, [Path(r['clip']).stem for r in rows], rotation=60, ha='right', fontsize=8)
  fig.suptitle('XQ180 retargeting: original joint mapping vs contact IK (9992 frames)')
  fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def render_preview(converter, legacy_root, root, output_dir):
  from PIL import Image, ImageDraw
  from scipy.spatial.transform import Rotation
  spec = converter.target.spec.copy()
  spec.worldbody.add_geom(name='preview_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[5, 5, .1], rgba=[.22, .26, .3, 1])
  spec.worldbody.add_light(pos=[1, -2, 3], dir=[-1, 2, -3], diffuse=[.8, .8, .8])
  model = spec.compile(); data = mujoco.MjData(model)
  options = mujoco.MjvOption(); options.geomgroup[3] = 0; options.sitegroup[:] = 0
  camera = mujoco.MjvCamera(); camera.distance = 2.7; camera.azimuth = 130; camera.elevation = -12
  camera.lookat[:] = [0, 0, .85]
  renderer = mujoco.Renderer(model, height=380, width=360)
  try:
    for label, relative in (
        ('walk', 'WalkandRun/walk_forward_loop_002__A022.npz'),
        ('recovery', 'Recovery/fallAndGetUp1_subject1.npz')):
      camera.distance = 3.6 if label == 'recovery' else 2.7
      camera.lookat[:] = [0, 0, .5 if label == 'recovery' else .85]
      with np.load(root/relative) as new:
        if label == 'walk': indices = [35, 55, 75, 95]
        else: indices = [int(np.argmin(new['body_pos_w'][:, 0, 2])), 1100, 1600, 2200]
      canvas = Image.new('RGB', (1440, 820), 'white'); draw = ImageDraw.Draw(canvas)
      for row, (version, directory) in enumerate((('legacy', legacy_root), ('contact IK', root))):
        with np.load(directory/relative) as clip:
          for col, frame in enumerate(indices):
            data.qpos[:] = np.r_[clip['body_pos_w'][frame, 0], clip['body_quat_w'][frame, 0], clip['joint_pos'][frame]]
            data.qpos[:2] = 0
            rotation = Rotation.from_quat(data.qpos[[4, 5, 6, 3]])
            yaw = rotation.as_euler('xyz')[2]
            xyzw = (Rotation.from_euler('z', -yaw)*rotation).as_quat()
            data.qpos[3:7] = xyzw[[3, 0, 1, 2]]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            canvas.paste(Image.fromarray(renderer.render()), (col*360, row*410+25))
            draw.text((col*360+10, row*410+6), f'{version} / frame {frame}', fill='black')
      canvas.save(output_dir/f'{label}_preview.png')
  finally:
    renderer.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, default=Path('docs/xq180_retarget'))
  parser.add_argument('--render', action='store_true')
  args = parser.parse_args()
  validate(args.output_dir, args.render)
