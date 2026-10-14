"""Map G1 NPZ motions to Hunter130 and recompute kinematics on the target robot.

Preserves timing, root XY/orientation and shared joint angles within target
limits. Root Z preserves the source robot's lowest collision-shape clearance,
clamped to zero to remove ground penetration. This is joint-space retargeting,
not an IK solve for foot/hand trajectories or dynamic feasibility.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from mjlab.entity import Entity
from src import SRC_PATH
from src.assets.robots import get_g1_robot_cfg, get_hunter130_robot_cfg


def angular_velocity_w(quaternions: np.ndarray, fps: float) -> np.ndarray:
  """Finite differences of rotations, insensitive to quaternion sign flips."""
  rotation = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
  velocity = np.empty((len(quaternions), 3))
  velocity[0] = (rotation[1] * rotation[0].inv()).as_rotvec() * fps
  velocity[-1] = (rotation[-1] * rotation[-2].inv()).as_rotvec() * fps
  if len(quaternions) > 2:
    velocity[1:-1] = (rotation[2:] * rotation[:-2].inv()).as_rotvec() * (fps / 2)
  return velocity


class GroundClearance:
  """Exact vertical support bounds of the model's collision geometry.

  Mesh hull vertices give identical extrema to the original mesh in every
  direction; the collision or visual mesh assets themselves are not changed.
  """

  def __init__(self, model: mujoco.MjModel):
    self.model = model
    self.geom_ids = np.flatnonzero((model.geom_contype | model.geom_conaffinity) != 0)
    self.mesh_vertices = {}
    for geom_id in self.geom_ids:
      if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id not in self.mesh_vertices:
          start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
          vertices = model.mesh_vert[start:start + count].astype(np.float64)
          self.mesh_vertices[mesh_id] = vertices[ConvexHull(vertices).vertices]

  def minimum_z(self, data: mujoco.MjData) -> float:
    model = self.model
    lowest = np.inf
    for geom_id in self.geom_ids:
      kind = model.geom_type[geom_id]
      size = model.geom_size[geom_id]
      direction = data.geom_xmat[geom_id].reshape(3, 3)[2]
      if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = size[0]
      elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + size[1] * abs(direction[2])
      elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = size[0] * np.linalg.norm(direction[:2]) + size[1] * abs(direction[2])
      elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(direction) @ size
      elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = np.linalg.norm(direction * size)
      elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        extent = -np.min(self.mesh_vertices[int(model.geom_dataid[geom_id])] @ direction)
      else:
        raise ValueError(f"Unsupported collision geometry: {model.geom(geom_id).name}")
      lowest = min(lowest, data.geom_xpos[geom_id, 2] - extent)
    return float(lowest)


class G1ToHunter130:
  def __init__(self, target_cfg=None, target_robot: str = "hunter130", joint_offsets: dict[str, float] | None = None):
    self.source = Entity(get_g1_robot_cfg())
    self.target = Entity(get_hunter130_robot_cfg() if target_cfg is None else target_cfg)
    self.target_robot = target_robot
    self.joint_offsets = joint_offsets or {}
    if set(self.joint_offsets) - set(self.target.joint_names):
      raise ValueError("Joint offset names must belong to the target robot")
    self.source_model = self.source.spec.compile()
    self.target_model = self.target.spec.compile()
    self.source_data = mujoco.MjData(self.source_model)
    self.target_data = mujoco.MjData(self.target_model)
    self.source_ground = GroundClearance(self.source_model)
    self.target_ground = GroundClearance(self.target_model)
    self.body_ids = np.array([self.target_model.body(n).id for n in self.target.body_names])
    self.limits = np.array([self.target_model.joint(n).range for n in self.target.joint_names])

  def convert(self, source_path: Path, destination: Path) -> dict:
    if source_path.resolve() == destination.resolve():
      raise ValueError("Source and destination must differ")
    with np.load(source_path, allow_pickle=False) as source:
      fps = float(source["fps"].item())
      joint_names = tuple(source["joint_names"].tolist()) if "joint_names" in source else self.source.joint_names
      body_names = tuple(source["body_names"].tolist()) if "body_names" in source else self.source.body_names
      frames = len(source["joint_pos"])
      if frames < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid length/fps: {source_path}")
      if len(set(joint_names)) != len(joint_names) or len(set(body_names)) != len(body_names):
        raise ValueError(f"Duplicate joint/body names: {source_path}")
      if set(joint_names) != set(self.source.joint_names) or set(body_names) != set(self.source.body_names):
        raise ValueError(f"Source joint/body names do not match G1: {source_path}")
      expected = {
        "joint_pos": (frames, len(joint_names)), "body_pos_w": (frames, len(body_names), 3),
        "body_quat_w": (frames, len(body_names), 4),
      }
      for key, shape in expected.items():
        if source[key].shape != shape or not np.isfinite(source[key]).all():
          raise ValueError(f"Invalid {key} in {source_path}; expected {shape}")
      g1_indices = [joint_names.index(n) for n in self.source.joint_names]
      target_indices = [joint_names.index(n) for n in self.target.joint_names]
      g1_joint_pos = source["joint_pos"][:, g1_indices].astype(np.float64)
      raw_joint_pos = source["joint_pos"][:, target_indices].astype(np.float64)
      raw_joint_pos += np.array([self.joint_offsets.get(n, 0.0) for n in self.target.joint_names])
      joint_pos = np.clip(raw_joint_pos, self.limits[:, 0], self.limits[:, 1])
      root_id = body_names.index("pelvis")
      root_pos = source["body_pos_w"][:, root_id].astype(np.float64).copy()
      root_quat = source["body_quat_w"][:, root_id].astype(np.float64).copy()
      norms = np.linalg.norm(root_quat, axis=1)
      if np.any(norms < 1e-8):
        raise ValueError(f"Zero root quaternion: {source_path}")
      root_quat /= norms[:, None]

      # Legacy NPZs have no names. Verify their presumed order against the model.
      source_body_ids = [self.source_model.body(n).id for n in body_names]
      for frame in np.unique(np.linspace(0, frames - 1, 3, dtype=int)):
        self.source_data.qpos[:] = np.r_[root_pos[frame], root_quat[frame], g1_joint_pos[frame]]
        mujoco.mj_kinematics(self.source_model, self.source_data)
        if not np.allclose(self.source_data.xpos[source_body_ids], source["body_pos_w"][frame], atol=1e-5, rtol=0):
          raise ValueError(f"G1 forward kinematics/order mismatch at frame {frame}: {source_path}")

    ground_clearance = np.empty(frames)
    original_clearance = np.empty(frames)
    root_z_offset = np.empty(frames)
    for frame in range(frames):
      self.source_data.qpos[:] = np.r_[root_pos[frame], root_quat[frame], g1_joint_pos[frame]]
      self.target_data.qpos[:] = np.r_[root_pos[frame], root_quat[frame], joint_pos[frame]]
      mujoco.mj_kinematics(self.source_model, self.source_data)
      mujoco.mj_kinematics(self.target_model, self.target_data)
      original_clearance[frame] = self.source_ground.minimum_z(self.source_data)
      ground_clearance[frame] = max(original_clearance[frame], 0.0)
      root_z_offset[frame] = ground_clearance[frame] - self.target_ground.minimum_z(self.target_data)
    root_pos[:, 2] += root_z_offset

    joint_vel = np.gradient(joint_pos, axis=0) * fps
    root_lin_vel = np.gradient(root_pos, axis=0) * fps
    root_ang_vel = angular_velocity_w(root_quat, fps)
    root_ang_vel_local = Rotation.from_quat(root_quat[:, [1, 2, 3, 0]]).inv().apply(root_ang_vel)
    shape = (frames, len(self.target.body_names))
    body_pos = np.empty((*shape, 3))
    body_quat = np.empty((*shape, 4))
    body_lin_vel = np.empty((*shape, 3))
    body_ang_vel = np.empty((*shape, 3))
    velocity = np.empty(6)
    for frame in range(frames):
      self.target_data.qpos[:] = np.r_[root_pos[frame], root_quat[frame], joint_pos[frame]]
      self.target_data.qvel[:] = np.r_[root_lin_vel[frame], root_ang_vel_local[frame], joint_vel[frame]]
      mujoco.mj_kinematics(self.target_model, self.target_data)
      mujoco.mj_comPos(self.target_model, self.target_data)
      mujoco.mj_comVel(self.target_model, self.target_data)
      body_pos[frame] = self.target_data.xpos[self.body_ids]
      body_quat[frame] = self.target_data.xquat[self.body_ids]
      for i, body_id in enumerate(self.body_ids):
        # XBODY selects the link origin, while BODY selects its COM.
        mujoco.mj_objectVelocity(self.target_model, self.target_data, mujoco.mjtObj.mjOBJ_XBODY, body_id, velocity, 0)
        body_ang_vel[frame, i] = velocity[:3]
        body_lin_vel[frame, i] = velocity[3:]
    np.testing.assert_allclose(body_lin_vel[:, 0], root_lin_vel, atol=1e-8)
    np.testing.assert_allclose(body_ang_vel[:, 0], root_ang_vel, atol=1e-8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      destination, fps=np.array([fps], dtype=np.float32),
      joint_names=np.asarray(self.target.joint_names), body_names=np.asarray(self.target.body_names),
      joint_pos=joint_pos.astype(np.float32), joint_vel=joint_vel.astype(np.float32),
      body_pos_w=body_pos.astype(np.float32), body_quat_w=body_quat.astype(np.float32),
      body_lin_vel_w=body_lin_vel.astype(np.float32), body_ang_vel_w=body_ang_vel.astype(np.float32),
      source_robot="g1", target_robot=self.target_robot, source_file=source_path.name,
      retarget_method="joint_mapping_limit_clip_ground_clearance_fk",
      root_z_offset=root_z_offset.astype(np.float32), ground_clearance=ground_clearance.astype(np.float32),
      joint_position_offsets=np.array([self.joint_offsets.get(n, 0.0) for n in self.target.joint_names], dtype=np.float32),
    )
    changed = np.abs(joint_pos - raw_joint_pos) > 1e-7
    return {
      "file": source_path.name, "frames": frames, "fps": fps,
      "clipped_joint_samples": int(changed.sum()),
      "clipped_joints": {n: int(changed[:, i].sum()) for i, n in enumerate(self.target.joint_names) if changed[:, i].any()},
      "root_z_offset_range": [float(root_z_offset.min()), float(root_z_offset.max())],
      "source_ground_clearance_range": [float(original_clearance.min()), float(original_clearance.max())],
      "target_ground_clearance_range": [float(ground_clearance.min()), float(ground_clearance.max())],
    }


def convert_directory(source_dir: Path, output_dir: Path, retargeter: G1ToHunter130) -> dict:
  files = sorted(source_dir.rglob("*.npz"))
  if not files:
    raise FileNotFoundError(f"No NPZ clips in {source_dir}")
  if source_dir.resolve() == output_dir.resolve():
    raise ValueError("Source/output directories must differ")
  report = []
  for i, source in enumerate(files):
    destination = output_dir / source.relative_to(source_dir)
    stats = retargeter.convert(source, destination)
    report.append({"path": str(source.relative_to(source_dir)), **stats})
    print(f"[{i + 1}/{len(files)}] {source.relative_to(source_dir)}: {stats['frames']} frames, {stats['clipped_joint_samples']} clipped samples", flush=True)
  summary = {
    "source_dir": str(source_dir), "output_dir": str(output_dir), "target_robot": retargeter.target_robot,
    "joint_position_offsets": retargeter.joint_offsets,
    "clips": len(report), "frames": sum(r["frames"] for r in report),
    "dropped_joints": [n for n in retargeter.source.joint_names if n not in retargeter.target.joint_names],
    "method": "Shared joint mapping, target limit clipping, per-frame ground clearance matching, target FK and link-origin velocities; no IK/dynamics solve.",
    "motions": report,
  }
  (output_dir / "retarget_report.json").write_text(json.dumps(summary, indent=2) + "\n")
  print(f"Converted {summary['clips']} clips / {summary['frames']} frames to {output_dir}")
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-dir", type=Path, default=SRC_PATH / "assets/motions/g1/amp")
  parser.add_argument("--output-dir", type=Path, default=SRC_PATH / "assets/motions/hunter130/amp")
  args = parser.parse_args()
  convert_directory(args.source_dir, args.output_dir, G1ToHunter130())


if __name__ == "__main__":
  main()
