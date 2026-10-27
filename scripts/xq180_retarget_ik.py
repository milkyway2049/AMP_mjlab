"""Contact-aware, soft-limit-bounded whole-body G1 -> XQ180 retargeting.

Fits sole trajectories, torso orientation and upper/forearm directions with
MuJoCo Jacobians and bounded least squares. Self/ground contacts are inequality
penalties. This is kinematic optimization, not a dynamics-feasibility solve.
"""

import json

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_closing, binary_opening, maximum_filter1d
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

if __package__:
  from .retarget_g1_to_xq180 import G1ToXQ180
  from .retarget_g1_to_hunter130 import angular_velocity_w
else:
  from retarget_g1_to_xq180 import G1ToXQ180
  from retarget_g1_to_hunter130 import angular_velocity_w
from src.assets.robots.xq180.xq180_constants import XQ180_VELOCITY_LIMITS


def skew(v):
  x, y, z = v
  return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rotation_error(goal, actual):
  return Rotation.from_matrix(goal @ actual.T).as_rotvec()


class G1ToXQ180IK(G1ToXQ180):
  method = 'contact_whole_body_ik_soft_limits_v1'

  def __init__(self, iterations=18):
    if iterations < 1:
      raise ValueError('iterations must be positive')
    super().__init__()
    self.iterations = iterations
    center = self.limits.mean(axis=1)
    half = np.diff(self.limits, axis=1)[:, 0] * .45
    # A small margin keeps float32 serialization inside reset's soft limits.
    self.soft_lower, self.soft_upper = center - half + 1e-5, center + half - 1e-5
    spec = self.target.spec.copy()
    spec.worldbody.add_geom(name='ik_ground', type=mujoco.mjtGeom.mjGEOM_PLANE,
                           size=[10, 10, .1], friction=[.6, .005, .0001])
    self.ik_model = spec.compile()
    self.ik_model.geom_margin[:] = .012
    self.ik_data = mujoco.MjData(self.ik_model)
    self.jac_pos = np.zeros((3, self.ik_model.nv))
    self.jac_rot = np.zeros_like(self.jac_pos)
    self.source_sites = [self.source_model.site(side + '_foot').id for side in ('left', 'right')]
    self.target_sites = [self.ik_model.site(side + '_foot').id for side in ('left', 'right')]
    self.source_torso = self.source_model.body('torso_link').id
    self.target_torso = self.ik_model.body('waist_Link').id
    self.source_arm_ids = [(self.source_model.body(side + '_shoulder_yaw_link').id,
                            self.source_model.body(side + '_elbow_link').id) for side in ('left', 'right')]
    self.target_arm_ids = [(self.ik_model.body(side + '_shoulder_yaw_Link').id,
                            self.ik_model.body(side + 'elbow_Link').id) for side in ('left', 'right')]
    for model, data in ((self.source_model, self.source_data), (self.target_model, self.target_data)):
      mujoco.mj_resetDataKeyframe(model, data, 0)
      mujoco.mj_kinematics(model, data)
    source_height = self.source_data.qpos[2] - self.source_data.site_xpos[self.source_sites, 2].mean()
    target_site_ids = [self.target_model.site(side + '_foot').id for side in ('left', 'right')]
    target_height = self.target_data.qpos[2] - self.target_data.site_xpos[target_site_ids, 2].mean()
    self.height_scale = float(target_height / source_height)
    self.width_offsets = (self.target_data.site_xpos[target_site_ids, :2]
                          - self.source_data.site_xpos[self.source_sites, :2])
    self.width_offsets[:, 0] = 0
    self.velocity_limits = np.array([XQ180_VELOCITY_LIMITS[n] for n in self.target.joint_names])

  def source_targets(self, source_path):
    with np.load(source_path, allow_pickle=False) as source:
      values = {k: source[k].copy() for k in source.files}
    fps = float(values['fps'].item())
    names = tuple(values['joint_names'].tolist()) if 'joint_names' in values else self.source.joint_names
    bodies = tuple(values['body_names'].tolist()) if 'body_names' in values else self.source.body_names
    if (len(names) != len(set(names)) or len(bodies) != len(set(bodies)) or
        set(names) != set(self.source.joint_names) or set(bodies) != set(self.source.body_names)):
      raise ValueError(f'{source_path}: source must be a complete G1 motion')
    q = values['joint_pos'][:, [names.index(n) for n in self.source.joint_names]].astype(float)
    root_id = bodies.index('pelvis')
    root_pos = values['body_pos_w'][:, root_id].astype(float).copy()
    root_quat = values['body_quat_w'][:, root_id].astype(float).copy()
    if len(q) < 2 or not np.isfinite(fps) or fps <= 0 or not np.isfinite(q).all():
      raise ValueError(f'{source_path}: invalid timing or joint positions')
    norms = np.linalg.norm(root_quat, axis=1)
    if np.any(norms < 1e-8) or not np.isfinite(root_pos).all() or not np.isfinite(norms).all():
      raise ValueError(f'{source_path}: invalid root pose')
    root_quat /= norms[:, None]
    mapped = q[:, [self.source.joint_names.index(self.joint_mapping[n]) for n in self.target.joint_names]]
    mapped += np.array([self.joint_offsets.get(n, 0.) for n in self.target.joint_names])
    mapped = gaussian_filter1d(np.clip(mapped, self.soft_lower, self.soft_upper), .8, axis=0, mode='nearest')
    count = len(q)
    feet = np.empty((count, 2, 3)); foot_rot = np.empty((count, 2, 3, 3))
    torso = np.empty((count, 3, 3)); arm = np.empty((count, 2, 2, 3))
    ground_root_z = np.empty(count)
    for f in range(count):
      self.source_data.qpos[:] = np.r_[root_pos[f], root_quat[f], q[f]]
      mujoco.mj_kinematics(self.source_model, self.source_data)
      feet[f] = self.source_data.site_xpos[self.source_sites]
      foot_rot[f] = self.source_data.site_xmat[self.source_sites].reshape(2, 3, 3)
      torso[f] = self.source_data.xmat[self.source_torso].reshape(3, 3)
      for side, (upper, fore) in enumerate(self.source_arm_ids):
        arm[f, side, 0] = -self.source_data.xmat[upper].reshape(3, 3)[:, 2]
        arm[f, side, 1] = self.source_data.xmat[fore].reshape(3, 3)[:, 0]
      self.target_data.qpos[:] = np.r_[root_pos[f], root_quat[f], mapped[f]]
      mujoco.mj_kinematics(self.target_model, self.target_data)
      source_clearance = max(self.source_ground.minimum_z(self.source_data), .004)
      ground_root_z[f] = root_pos[f, 2] + source_clearance - self.target_ground.minimum_z(self.target_data)
    root_rotation = Rotation.from_quat(root_quat[:, [1, 2, 3, 0]])
    yaw = root_rotation.as_euler('xyz')[:, 2]
    upright = root_rotation.as_matrix()[:, 2, 2]
    walking = np.clip((upright - .5) / .35, 0, 1) * np.clip((root_pos[:, 2] - .35) / .2, 0, 1)
    walking = gaussian_filter1d(walking, 2, mode='nearest')
    root_goal = root_pos.copy()
    root_goal[:, 2] = walking * root_pos[:, 2] * self.height_scale + (1-walking) * ground_root_z
    root_goal = gaussian_filter1d(root_goal, 1., axis=0, mode='nearest')
    contacts = np.zeros((count, 2), dtype=bool)
    targets = feet.copy()
    for side in range(2):
      offset = np.zeros((count, 3))
      offset[:, 0] = -np.sin(yaw) * self.width_offsets[side, 1]
      offset[:, 1] = np.cos(yaw) * self.width_offsets[side, 1]
      targets[:, side, :2] += offset[:, :2]
      # XQ180 has wider feet/hips and no ankle roll. Keep walking foot goals
      # on their own side instead of copying G1 crossover steps through a foot.
      relative = targets[:, side, :2] - root_pos[:, :2]
      lateral = -np.sin(yaw)*relative[:, 0] + np.cos(yaw)*relative[:, 1]
      corrected = np.maximum(lateral, .11) if side == 0 else np.minimum(lateral, -.11)
      correction = (corrected-lateral)*walking
      targets[:, side, 0] -= np.sin(yaw)*correction
      targets[:, side, 1] += np.cos(yaw)*correction
      targets[:, side, 2] = np.maximum(feet[:, side, 2] * self.height_scale, .006)
      vertical_speed = np.gradient(feet[:, side, 2]) * fps
      raw_contact = (feet[:, side, 2] < .04) & (abs(vertical_speed) < .5) & (walking > .8)
      padded = np.pad(raw_contact, 3, mode='edge')
      contact = binary_opening(binary_closing(padded, structure=np.ones(3)), structure=np.ones(3))[3:-3]
      contacts[:, side] = contact
      edges = np.diff(np.r_[False, contact, False].astype(int))
      for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        angle = np.unwrap(Rotation.from_matrix(foot_rot[start:end, side]).as_euler('xyz')[:, 2])
        # Split long pivot contacts: welding a foot for a complete 360-degree
        # turn is incompatible with the target hip yaw limits.
        boundaries = [int(start)]
        for frame in range(start+1, end):
          anchor = boundaries[-1]
          moved = np.linalg.norm(targets[frame, side, :2]-targets[anchor, side, :2])
          turned = abs(angle[frame-start]-angle[anchor-start])
          if moved > .07 or turned > .24:
            boundaries.append(frame)
        boundaries.append(int(end))
        for first, last in zip(boundaries[:-1], boundaries[1:]):
          targets[first:last, side, :2] = np.median(targets[first:last, side, :2], axis=0)
          targets[first:last, side, 2] = .006
          heading = np.median(angle[first-start:last-start])
          foot_rot[first:last, side] = Rotation.from_euler('z', heading).as_matrix()
      targets[:, side] = gaussian_filter1d(targets[:, side], .8, axis=0, mode='nearest')
      rotations = Rotation.from_matrix(foot_rot[:, side]).as_quat()
      for frame in range(1, count):
        if np.dot(rotations[frame-1], rotations[frame]) < 0:
          rotations[frame] *= -1
      rotations = gaussian_filter1d(rotations, 1., axis=0, mode='nearest')
      rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
      foot_rot[:, side] = Rotation.from_quat(rotations).as_matrix()
    return {'fps': fps, 'root_pos': root_goal, 'root_quat': root_quat, 'mapped': mapped,
            'feet': targets, 'foot_rot': foot_rot, 'contacts': contacts, 'walking': walking,
            'contact_strength': gaussian_filter1d(contacts.astype(float), 1., axis=0, mode='nearest'),
            'torso': torso, 'arm': arm, 'original_root_pos': root_pos}

  def solve_frame(self, goals, f, previous):
    m, d = self.ik_model, self.ik_data
    q_ref = np.r_[goals['root_pos'][f], goals['root_quat'][f], goals['mapped'][f]]
    if previous is None:
      d.qpos[:] = q_ref
    else:
      d.qpos[:] = previous
      if f:
        d.qpos[:3] += goals['root_pos'][f] - goals['root_pos'][f-1]
    nv = m.nv
    eye = np.eye(nv)
    for iteration in range(self.iterations * (4 if previous is None else 1)):
      mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_collision(m, d)
      rows, errors = [], []
      def task(jac, error, weight):
        rows.append(np.atleast_2d(jac) * weight)
        errors.append(np.atleast_1d(error) * weight)
      tangent = np.empty(nv)
      mujoco.mj_differentiatePos(m, tangent, 1., d.qpos, q_ref)
      task(eye[:2], tangent[:2], 9.)
      task(eye[2:3], tangent[2:3], 40.)
      task(eye[3:6], tangent[3:6], .8)
      task(eye[6:], tangent[6:], .22)
      if previous is not None:
        mujoco.mj_differentiatePos(m, tangent, 1., d.qpos, previous)
        task(eye[6:], tangent[6:], .35)
      for side, site in enumerate(self.target_sites):
        mujoco.mj_jacSite(m, d, self.jac_pos, self.jac_rot, site)
        stance = goals['contact_strength'][f, side]
        task(self.jac_pos.copy(), goals['feet'][f, side] - d.site_xpos[site], 35.+40.*stance)
        actual = d.site_xmat[site].reshape(3, 3)
        task(self.jac_rot.copy(), rotation_error(goals['foot_rot'][f, side], actual), 1.+2.*stance)
      body = self.target_torso
      mujoco.mj_jacBody(m, d, self.jac_pos, self.jac_rot, body)
      task(self.jac_rot.copy(), rotation_error(goals['torso'][f], d.xmat[body].reshape(3, 3)), 3.)
      for side, (upper, fore) in enumerate(self.target_arm_ids):
        for segment, body in enumerate((upper, fore)):
          mujoco.mj_jacBody(m, d, self.jac_pos, self.jac_rot, body)
          actual = (-d.xmat[body].reshape(3, 3)[:, 2] if segment == 0
                    else d.xmat[body].reshape(3, 3)[:, 0])
          jac = -skew(actual) @ self.jac_rot
          task(jac, goals['arm'][f, side, segment] - actual, 2.5)
      # One deepest contact per pair avoids giving mesh manifolds extra weight.
      active = {}
      for contact in d.contact:
        if contact.dist < .006:
          pair = (int(contact.geom1), int(contact.geom2))
          if pair not in active or contact.dist < active[pair].dist:
            active[pair] = contact
      for (g1, g2), contact in active.items():
        j1 = np.zeros((3, nv)); j2 = np.zeros((3, nv))
        mujoco.mj_jac(m, d, j1, None, contact.pos, m.geom_bodyid[g1])
        mujoco.mj_jac(m, d, j2, None, contact.pos, m.geom_bodyid[g2])
        normal = contact.frame[:3]
        task(normal @ (j2-j1), [.006-contact.dist], 350.)
      jac = np.concatenate(rows); error = np.concatenate(errors)
      # Trust region and exact soft position bounds, plus a frame-to-frame
      # velocity box. Box constraints are solved, not clipped after an IK solve.
      lower = np.r_[np.full(3, -.035), np.full(3, -.13), np.full(19, -.2)]
      upper = -lower
      lower[6:] = np.maximum(lower[6:], self.soft_lower-d.qpos[7:])
      upper[6:] = np.minimum(upper[6:], self.soft_upper-d.qpos[7:])
      if previous is not None:
        step_limit = self.velocity_limits / goals['fps'] * .9
        lower[6:] = np.maximum(lower[6:], previous[7:]-step_limit-d.qpos[7:])
        upper[6:] = np.minimum(upper[6:], previous[7:]+step_limit-d.qpos[7:])
      # Keep strict feasibility in the bounded solver at roundoff boundaries.
      upper = np.maximum(upper, lower+1e-10)
      solution = lsq_linear(np.r_[jac, eye*.08], np.r_[error, np.zeros(nv)],
        bounds=(lower, upper), method='bvls', tol=1e-6, max_iter=40).x
      mujoco.mj_integratePos(m, d.qpos, solution, .6)
      if np.linalg.norm(solution) < 2e-4:
        break
    return d.qpos.copy()

  def convert(self, source_path, destination, frame_limit=None):
    if source_path.resolve() == destination.resolve():
      raise ValueError('Source and destination must differ')
    goals = self.source_targets(source_path)
    frames = len(goals['mapped']) if frame_limit is None else min(frame_limit, len(goals['mapped']))
    poses = np.empty((frames, self.target_model.nq)); previous = None
    for f in range(frames):
      poses[f] = self.solve_frame(goals, f, previous)
      previous = poses[f]
    poses = self.finish_trajectory(poses, goals['fps'])
    metrics = self.write_motion(source_path, destination, poses, goals)
    print(source_path.name, json.dumps(metrics), flush=True)
    return metrics

  def finish_trajectory(self, poses, fps):
    """Smooth solve noise, then lift a smooth envelope above exact CAD bounds.

    The envelope never lowers a frame back through the floor. It may slightly
    raise an unreachable support goal; the saved report measures that error.
    """
    poses[:, :3] = gaussian_filter1d(poses[:, :3], .8, axis=0, mode='nearest')
    poses[:, 7:] = gaussian_filter1d(poses[:, 7:], .8, axis=0, mode='nearest')
    quaternion = poses[:, 3:7].copy()
    for f in range(1, len(poses)):
      if np.dot(quaternion[f-1], quaternion[f]) < 0:
        quaternion[f] *= -1
    quaternion = gaussian_filter1d(quaternion, .8, axis=0, mode='nearest')
    poses[:, 3:7] = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    # Repair occasional collisions introduced by averaging nonlinear geometry.
    # The projection holds the nearby solved pose, rather than reintroducing
    # an unreachable source arm/foot target.
    for frame in range(len(poses)):
      poses[frame] = self.project_collisions(poses[frame])
    needed = np.zeros(len(poses))
    for f, pose in enumerate(poses):
      self.target_data.qpos[:] = pose
      mujoco.mj_kinematics(self.target_model, self.target_data)
      needed[f] = max(.002-self.target_ground.minimum_z(self.target_data), 0.)
    envelope = gaussian_filter1d(maximum_filter1d(needed, size=7, mode='nearest'), .8, mode='nearest')
    poses[:, 2] += np.maximum(envelope, needed)
    rate = np.abs(np.gradient(poses[:, 7:], axis=0)*fps) / self.velocity_limits
    if not np.isfinite(poses).all() or rate.max() > 1.001:
      raise ValueError('Retargeted trajectory is nonfinite or exceeds joint velocity ratings')
    return poses

  def project_collisions(self, pose):
    m, d = self.ik_model, self.ik_data
    reference = pose.copy(); d.qpos[:] = pose
    mujoco.mj_kinematics(m, d)
    feet = d.site_xpos[self.target_sites].copy()
    eye = np.eye(m.nv)
    for iteration in range(35):
      mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_collision(m, d)
      if not any(c.dist < -.0002 for c in d.contact):
        break
      error = np.empty(m.nv)
      mujoco.mj_differentiatePos(m, error, 1., d.qpos, reference)
      weights = np.r_[np.full(3, 35.), np.full(3, 2.), np.full(19, 2.)]
      rows = [eye*weights[:, None]]; errors = [error*weights]
      for side, site in enumerate(self.target_sites):
        mujoco.mj_jacSite(m, d, self.jac_pos, self.jac_rot, site)
        rows.append(self.jac_pos.copy()*20.)
        errors.append((feet[side]-d.site_xpos[site])*20.)
      active = {}
      for c in d.contact:
        pair = (int(c.geom1), int(c.geom2))
        if c.dist < .004 and (pair not in active or c.dist < active[pair].dist):
          active[pair] = c
      for (g1, g2), c in active.items():
        j1 = np.zeros((3, m.nv)); j2 = np.zeros_like(j1)
        mujoco.mj_jac(m, d, j1, None, c.pos, m.geom_bodyid[g1])
        mujoco.mj_jac(m, d, j2, None, c.pos, m.geom_bodyid[g2])
        rows.append((c.frame[:3]@(j2-j1))[None]*350.)
        errors.append(np.array([.004-c.dist])*350.)
      lower = np.r_[np.full(3, -.015), np.full(3, -.08), np.full(19, -.13)]
      upper = -lower
      lower[6:] = np.maximum(lower[6:], self.soft_lower-d.qpos[7:])
      upper[6:] = np.minimum(upper[6:], self.soft_upper-d.qpos[7:])
      step = lsq_linear(np.concatenate(rows), np.concatenate(errors),
        bounds=(lower, np.maximum(upper, lower+1e-10)), method='bvls', tol=1e-6).x
      mujoco.mj_integratePos(m, d.qpos, step, .65)
    return d.qpos.copy()

  def write_motion(self, source_path, destination, poses, goals):
    m, d = self.target_model, self.target_data
    fps = goals['fps']; count = len(poses)
    joint_vel = np.gradient(poses[:, 7:], axis=0)*fps
    lin_vel = np.gradient(poses[:, :3], axis=0)*fps
    ang_vel = angular_velocity_w(poses[:, 3:7], fps)
    local_ang_vel = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).inv().apply(ang_vel)
    shape = (count, len(self.target.body_names))
    positions = np.empty((*shape, 3)); quaternions = np.empty((*shape, 4))
    linear = np.empty((*shape, 3)); angular = np.empty((*shape, 3))
    feet = np.empty((count, 2, 3)); clearances = np.empty(count)
    contacts = np.zeros(count, dtype=bool); depth = np.zeros(count)
    site_ids = [m.site(side+'_foot').id for side in ('left', 'right')]
    velocity = np.empty(6)
    for f, qpos in enumerate(poses):
      d.qpos[:] = qpos; d.qvel[:] = np.r_[lin_vel[f], local_ang_vel[f], joint_vel[f]]
      mujoco.mj_forward(m, d)
      positions[f] = d.xpos[self.body_ids]; quaternions[f] = d.xquat[self.body_ids]
      feet[f] = d.site_xpos[site_ids]
      clearances[f] = self.target_ground.minimum_z(d)
      depth[f] = min((float(c.dist) for c in d.contact), default=0.)
      contacts[f] = depth[f] < -.001
      for i, body in enumerate(self.body_ids):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
        angular[f, i] = velocity[:3]; linear[f, i] = velocity[3:]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    with temporary.open('wb') as stream:
      np.savez_compressed(stream, fps=np.array([fps], np.float32),
        joint_names=np.array(self.target.joint_names), body_names=np.array(self.target.body_names),
        joint_pos=poses[:, 7:].astype(np.float32), joint_vel=joint_vel.astype(np.float32),
        body_pos_w=positions.astype(np.float32), body_quat_w=quaternions.astype(np.float32),
        body_lin_vel_w=linear.astype(np.float32), body_ang_vel_w=angular.astype(np.float32),
        source_robot='g1', target_robot='xq180', source_file=source_path.name,
        retarget_method=self.method, soft_joint_pos_limit_factor=np.float32(.9),
        height_scale=np.float32(self.height_scale),
        source_joint_names=np.array([self.joint_mapping[n] for n in self.target.joint_names]),
        joint_position_offsets=np.array([self.joint_offsets.get(n, 0.) for n in self.target.joint_names]),
        ground_clearance=clearances.astype(np.float32),
        foot_contact=goals['contacts'][:count], foot_target_w=goals['feet'][:count].astype(np.float32))
    temporary.replace(destination)
    speed = np.linalg.norm(np.gradient(feet[:, :, :2], axis=0)*fps, axis=-1)
    near = feet[:, :, 2] < .035
    contact = goals['contacts'][:count]
    error = np.linalg.norm(feet-goals['feet'][:count], axis=-1)
    return {'frames': count, 'fps': fps, 'self_penetration_over_1mm_frame_fraction': float(contacts.mean()),
      'deepest_self_penetration_m': float(depth.min()), 'ground_min_m': float(clearances.min()),
      'near_floor_site_speed_mean_m_s': float(speed[near].mean()) if near.any() else None,
      'contact_foot_error_mean_m': float(error[contact].mean()) if contact.any() else None,
      'root_vertical_accel_p99_m_s2': float(np.quantile(abs(np.gradient(lin_vel[:, 2])*fps), .99))}


def convert_ik_directory(source_dir, output_dir, iterations=18):
  files = sorted(source_dir.rglob('*.npz'))
  if not files:
    raise FileNotFoundError(f'No G1 motion clips in {source_dir}')
  if source_dir.resolve() == output_dir.resolve():
    raise ValueError('Source and output directories must differ')
  converter = G1ToXQ180IK(iterations)
  rows = []
  for index, path in enumerate(files):
    print(f'[{index+1}/{len(files)}] {path.relative_to(source_dir)}', flush=True)
    destination = output_dir / path.relative_to(source_dir)
    rows.append({'path': str(path.relative_to(source_dir)), **converter.convert(path, destination)})
  report = {'source_dir': str(source_dir), 'output_dir': str(output_dir),
    'source_robot': 'g1', 'target_robot': 'xq180', 'method': converter.method,
    'clips': len(rows), 'frames': sum(r['frames'] for r in rows),
    'iterations': iterations, 'height_scale': converter.height_scale,
    'soft_joint_pos_limit_factor': .9,
    'limitations': 'Kinematic IK with collision penalties; not an inverse dynamics, stability or trained-policy validation.',
    'motions': rows}
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / 'retarget_report.json').write_text(json.dumps(report, indent=2)+'\n')
  return report
