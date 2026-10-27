"""Play every training reference NPZ in MuJoCo, without a policy or GPU.

This is kinematic replay: root and joints are set from each saved frame.
It shows reference motion quality, not PD tracking or learned policy quality.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import mujoco
import numpy as np

from mjlab.entity import Entity
from src import SRC_PATH


@dataclass
class Clip:
  path: Path
  fps: float
  qpos: np.ndarray

  @property
  def duration(self):
    return len(self.qpos) / self.fps


def make_model(robot):
  if robot == "xq180":
    from src.assets.robots.xq180.xq180_constants import get_xq180_robot_cfg
    cfg = get_xq180_robot_cfg()
  else:
    from src.assets.robots.unitree_g1.g1_constants import get_g1_robot_cfg
    cfg = get_g1_robot_cfg()
  entity = Entity(cfg)
  spec = entity.spec.copy()
  spec.add_texture(name="replay_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=256, height=256,
    rgb1=[.18, .22, .26], rgb2=[.32, .36, .40])
  floor_material = spec.add_material(name="replay_floor_material",
    texuniform=True, texrepeat=[2, 2])
  floor_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "replay_grid"
  spec.worldbody.add_geom(name="replay_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0, 0, .1], material="replay_floor_material")
  spec.worldbody.add_light(pos=[0, -3, 5], dir=[0, 1, -2])
  return spec.compile(), tuple(entity.joint_names), tuple(entity.body_names)


def load_clip(path, model, joint_names, body_names):
  with np.load(path, allow_pickle=False) as saved:
    names = tuple(saved["joint_names"].tolist()) if "joint_names" in saved else joint_names
    bodies = tuple(saved["body_names"].tolist()) if "body_names" in saved else body_names
    if len(set(names)) != len(names) or set(names) != set(joint_names):
      raise ValueError(f"{path}: joint names do not match the selected robot")
    if len(set(bodies)) != len(bodies) or set(bodies) != set(body_names):
      raise ValueError(f"{path}: body names do not match the selected robot")
    fps = float(saved["fps"].item())
    joints = saved["joint_pos"]
    frames = len(joints)
    if not np.isfinite(fps) or fps <= 0 or frames < 2:
      raise ValueError(f"{path}: invalid fps or frame count")
    for key, shape in (("joint_pos", (frames, len(names))),
                       ("body_pos_w", (frames, len(bodies), 3)),
                       ("body_quat_w", (frames, len(bodies), 4))):
      if saved[key].shape != shape or not np.isfinite(saved[key]).all():
        raise ValueError(f"{path}: invalid {key}, expected finite {shape}")
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or model.jnt_qposadr[free[0]] != 0:
      raise ValueError("Replay requires a single free root at qpos[0:7]")
    root = bodies.index(model.body(model.jnt_bodyid[free[0]]).name)
    quats = saved["body_quat_w"][:, root]
    if not np.allclose(np.linalg.norm(quats, axis=1), 1, atol=1e-3):
      raise ValueError(f"{path}: root quaternions must be unit wxyz quaternions")
    qpos = np.zeros((frames, model.nq))
    qpos[:, :3] = saved["body_pos_w"][:, root]
    qpos[:, 3:7] = quats
    for index, name in enumerate(names):
      qpos[:, model.joint(name).qposadr[0]] = joints[:, index]
  return Clip(path, fps, qpos)


def set_frame(model, data, clip, frame):
  data.qpos[:] = clip.qpos[frame]
  data.qvel[:] = 0
  data.time = frame / clip.fps
  mujoco.mj_forward(model, data)


def check_playlist(model, clips):
  """Exercise exactly the same frame application used by the GUI, headlessly."""
  data = mujoco.MjData(model)
  max_fk_error = 0.
  for clip in clips:
    with np.load(clip.path, allow_pickle=False) as saved:
      names = tuple(saved["body_names"].tolist()) if "body_names" in saved else None
      ids = [model.body(n).id for n in names] if names else None
      for frame in range(len(clip.qpos)):
        set_frame(model, data, clip, frame)
        if not np.isfinite(data.xpos).all():
          raise ValueError(f"{clip.path}: nonfinite FK at frame {frame}")
        if ids is not None:
          error = float(np.max(np.linalg.norm(data.xpos[ids] - saved["body_pos_w"][frame], axis=-1)))
          max_fk_error = max(max_fk_error, error)
          if error > 1e-4:
            raise ValueError(f"{clip.path}: saved body positions disagree with FK ({error} m)")
    print(f"OK {clip.path.name}: {len(clip.qpos)} frames, {clip.fps:g} Hz, {clip.duration:.2f} s", flush=True)
  print(f"Checked {len(clips)} clips / {sum(len(c.qpos) for c in clips)} frames; "
        f"maximum saved-body FK error {max_fk_error:.3g} m. Kinematic check only.")


def play(model, clips, speed, once):
  import mujoco.viewer

  keys = []
  key_lock = threading.Lock()

  def on_key(key):
    with key_lock:
      keys.append(key)

  index, frame = 0, 0
  paused, follow = False, True
  data = mujoco.MjData(model)
  set_frame(model, data, clips[0], 0)
  print("Kinematic reference replay (no PD / no policy).\n"
        "Space: pause | Right/Left or ]/[: next/previous clip | R: restart clip\n"
        "Comma/Period: step one frame while paused | F: follow root | Esc: exit", flush=True)

  def announce():
    c = clips[index]
    print(f"[{index + 1}/{len(clips)}] {c.path.parent.name}/{c.path.name} "
          f"({len(c.qpos)} frames, {c.duration:.2f} s, speed {speed:g}x)", flush=True)

  announce()
  with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    viewer.cam.distance = 3.5
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -15
    viewer.cam.lookat[:] = [*data.qpos[:2], .8]
    viewer.opt.geomgroup[3] = 0
    viewer.opt.sitegroup[:] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    cursor = 0.
    previous_time = time.monotonic()
    while viewer.is_running():
      now = time.monotonic()
      elapsed = now - previous_time
      previous_time = now
      with key_lock:
        pending = keys[:]
        keys.clear()
      reset_clock = False
      for key in pending:
        if key == 32:
          paused = not paused
          reset_clock = True
        elif key in (262, 93, 263, 91):  # GLFW arrows, ] and [.
          index = (index + (1 if key in (262, 93) else -1)) % len(clips)
          cursor = 0.
          reset_clock = True
          announce()
        elif key in (82, 114):
          cursor = 0.
          reset_clock = True
        elif key in (70, 102):
          follow = not follow
        elif key in (44, 46):
          paused = True
          cursor = float(np.clip(int(cursor) + (1 if key == 46 else -1), 0, len(clips[index].qpos) - 1))
          reset_clock = True
      if not paused and not reset_clock:
        cursor += elapsed * clips[index].fps * speed
      if cursor >= len(clips[index].qpos):
        if once and index == len(clips) - 1:
          return
        index = (index + 1) % len(clips)
        cursor = 0.
        announce()
      frame = min(int(cursor), len(clips[index].qpos) - 1)
      with viewer.lock():
        set_frame(model, data, clips[index], frame)
        if follow:
          viewer.cam.lookat[:2] = data.qpos[:2]
      viewer.set_texts(texts=[(
        None, None, f"[{index + 1}/{len(clips)}] {clips[index].path.name}",
        f"{frame + 1}/{len(clips[index].qpos)}  {frame / clips[index].fps:.2f}s  "
        f"{speed:g}x {'PAUSED' if paused else ''}",
      )])
      viewer.sync()
      time.sleep(.008)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--robot", choices=("xq180", "g1"), default="xq180")
  parser.add_argument("--motion-dir", type=Path, help="Recursively load NPZs from this directory")
  parser.add_argument("--speed", type=float, default=1.)
  parser.add_argument("--start", help="Start from this unique filename or stem")
  parser.add_argument("--once", action="store_true", help="Exit after one playlist pass")
  parser.add_argument("--check", action="store_true", help="Validate all replay frames without opening a window")
  args = parser.parse_args()
  if not np.isfinite(args.speed) or args.speed <= 0:
    parser.error("--speed must be positive and finite")
  if args.motion_dir is None:
    if args.robot == "xq180":
      from src.tasks.amp_loco.config.xq180.motion import XQ180_LOCOMOTION_DIR
      args.motion_dir = XQ180_LOCOMOTION_DIR
    else:
      args.motion_dir = SRC_PATH / "assets/motions/g1/amp"
  paths = sorted(args.motion_dir.rglob("*.npz"), key=lambda p: ("Recovery" in p.parts, str(p)))
  if not paths:
    parser.error(f"No NPZ motions in {args.motion_dir}")
  if args.start:
    matches = [i for i, p in enumerate(paths) if args.start in (p.name, p.stem)]
    if len(matches) != 1:
      parser.error("--start must match exactly one filename or stem")
    start = matches[0]
    paths = paths[start:] + paths[:start]
  model, joints, bodies = make_model(args.robot)
  clips = [load_clip(p, model, joints, bodies) for p in paths]
  print(f"{args.robot}: {len(clips)} clips / {sum(len(c.qpos) for c in clips)} frames / "
        f"{sum(c.duration for c in clips):.2f} s\nDirectory: {args.motion_dir}", flush=True)
  if args.check:
    check_playlist(model, clips)
  else:
    play(model, clips, args.speed, args.once)


if __name__ == "__main__":
  main()
