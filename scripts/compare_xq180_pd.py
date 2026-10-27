"""Compare legacy/new XQ180 builtin PD and G1 in controlled CPU experiments.

Fixed-root tests disable gravity/contact and apply a small joint step. They
measure servo dynamics, not learned balance or Unitree torque-speed/friction.
The separate free-root home-target test includes gravity/contact but no policy.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import Entity
from src.assets.robots.xq180.xq180_constants import (
  XQ180_PD_PROFILE, get_xq180_robot_cfg, g1_joint_name,
)
from src.assets.robots.unitree_g1.g1_constants import get_g1_robot_cfg


def make_model(cfg, fixed_root):
  spec = Entity(cfg).spec
  spec.option.timestep = .005
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  if fixed_root:
    for key in spec.keys:
      key.qpos = np.asarray(key.qpos)[7:]
    spec.delete(spec.joints[0])
    spec.option.gravity = [0, 0, 0]
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  else:
    spec.worldbody.add_geom(name='ground', type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[5, 5, .1], friction=[.6, .005, .0001], condim=3)
  return spec.compile()


def apply_gains(model, joints):
  lookup = {j['name']: j for j in joints}
  for actuator_id in range(model.nu):
    name = model.joint(model.actuator_trnid[actuator_id, 0]).name
    model.actuator_gainprm[actuator_id, 0] = lookup[name]['kp']
    model.actuator_biasprm[actuator_id, 1] = -lookup[name]['kp']
    model.actuator_biasprm[actuator_id, 2] = -lookup[name]['kd']


def step_response(model, name):
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  home = data.qpos.copy()
  joint = model.joint(name)
  address = joint.qposadr[0]
  actuator_ids = model.actuator_trnid[:, 0]
  actuator = int(np.flatnonzero(actuator_ids == joint.id)[0])
  controls = home[model.jnt_qposadr[actuator_ids]].copy()
  base = float(home[address])
  signal = []
  force = []
  velocity = []
  for step in range(800):
    data.ctrl[:] = controls
    if step >= 40:
      data.ctrl[actuator] += .08
    mujoco.mj_step(model, data)
    signal.append(float(data.qpos[address] - base))
    force.append(float(abs(data.actuator_force[actuator])))
    velocity.append(float(abs(data.qvel[joint.dofadr[0]])))
    if not np.isfinite(data.qpos).all():
      raise ValueError(f'Nonfinite response: {name}')
  signal = np.asarray(signal)
  reached = np.flatnonzero(signal[40:] >= .072)
  return {
    'rise_to_90pct_s': float((reached[0] + 1) * .005) if len(reached) else None,
    'overshoot_pct': float(max(0, signal.max() / .08 - 1) * 100),
    'final_error_rad': float(abs(signal[-1] - .08)),
    'peak_speed_rad_s': max(velocity), 'peak_torque_nm': max(force),
  }, signal


def home_target_test(model):
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.ctrl[:] = data.qpos[7:]
  first_fall = None
  for _ in range(600):
    mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
      raise ValueError('Nonfinite home-target test')
    if first_fall is None and (data.qpos[2] < .5 or data.xmat[1, 8] < np.cos(1.)):
      first_fall = float(data.time)
  return {'duration_s': 3., 'fall_time_s': first_fall, 'finite': True,
          'note': 'Constant home targets, no balance feedback; not a policy performance test.'}


def compare(legacy_io, output_dir, current_io):
  old = json.loads(legacy_io.read_text())
  current = json.loads(current_io.read_text())
  models = {'g1': make_model(get_g1_robot_cfg(), True),
            'legacy': make_model(get_xq180_robot_cfg(), True),
            'new': make_model(get_xq180_robot_cfg(), True)}
  apply_gains(models['legacy'], old['joints'])
  apply_gains(models['new'], current['joints'])
  groups = ('left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_joint', 'waist_point',
            'left_shoulder_pitch_joint', 'leftelbow_joint')
  report = {'profile': current['pd_profile'], 'parameter_file': str(current_io),
    'fixed_root_test': {'step_rad': .08, 'step_at_s': .2, 'duration_s': 4.,
      'gravity': False, 'contacts': False,
      'note': 'Builtin PD only; G1 custom torque-speed/friction compute is not exercised.'},
    'responses': {}, 'home_target_no_policy': {}}
  traces = {}
  for name in groups:
    report['responses'][name] = {}; traces[name] = {}
    for label, model in models.items():
      model_name = g1_joint_name(name) if label == 'g1' else name
      metrics, trace = step_response(model, model_name)
      report['responses'][name][label] = metrics
      traces[name][label] = trace
  for label in ('legacy', 'new'):
    model = make_model(get_xq180_robot_cfg(), False)
    if label == 'legacy': apply_gains(model, old['joints'])
    else: apply_gains(model, current['joints'])
    report['home_target_no_policy'][label] = home_target_test(model)
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / 'pd_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(4, 2, figsize=(12, 12))
  for ax, name in zip(axes.flat, groups):
    time = np.arange(1, 801) * .005
    ax.plot(time, np.where(time >= .2, .08, 0), 'k:', label='target')
    for label, color in (('g1', '#3974a7'), ('legacy', '#d66842'), ('new', '#318264')):
      ax.plot(time, traces[name][label], color=color, label=label)
    ax.set_title(name, fontsize=10); ax.set_xlabel('Time (s)'); ax.set_ylabel('Joint offset (rad)')
    ax.grid(alpha=.2)
  axes.flat[0].legend()
  fig.suptitle('Small-step servo comparison: fixed base, gravity/contact disabled')
  fig.tight_layout(); fig.savefig(output_dir / 'pd_step_response.png', dpi=150); plt.close(fig)
  print(json.dumps(report['home_target_no_policy'], indent=2))
  return report


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--legacy-io', type=Path, default=Path('docs/xq180_policy_io_legacy.json'))
  parser.add_argument('--current-io', type=Path, default=Path('docs/xq180_policy_io.json'))
  parser.add_argument('--output-dir', type=Path, default=Path('docs/xq180_audit') / XQ180_PD_PROFILE)
  args = parser.parse_args()
  compare(args.legacy_io, args.output_dir, args.current_io)
