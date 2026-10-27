"""Retarget G1 AMP motions to XQ180 with contact IK (or legacy joint mapping)."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

if __package__:
  from .retarget_g1_to_hunter130 import G1ToHunter130, convert_directory
else:
  from retarget_g1_to_hunter130 import G1ToHunter130, convert_directory

from src import SRC_PATH
from src.assets.robots.xq180.xq180_constants import XQ180_URDF, get_xq180_robot_cfg


class G1ToXQ180(G1ToHunter130):
  def __init__(self):
    # G1 elbow rest frames have zero rotation; XQ180 has R_y(pi/2).
    # Both hinges use +Y, so R_y(rest + q_target) = R_y(q_source).
    urdf = ET.parse(XQ180_URDF).getroot()
    offsets = {}
    for side in ("left", "right"):
      name = f"{side}elbow_joint"
      joint = next(j for j in urdf.findall("joint") if j.get("name") == name)
      rpy = np.fromstring(joint.find("origin").get("rpy", "0 0 0"), sep=" ")
      axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
      if not np.allclose(rpy[[0, 2]], 0) or not np.allclose(axis, [0, 1, 0]):
        raise ValueError(f"Cannot compensate non-Y elbow rest frame: {name}")
      offsets[name] = -float(rpy[1])
    mapping = {"waist_point": "waist_yaw_joint"}
    for side in ("left", "right"):
      mapping[f"{side}_ankle_joint"] = f"{side}_ankle_pitch_joint"
      mapping[f"{side}elbow_joint"] = f"{side}_elbow_joint"
    super().__init__(get_xq180_robot_cfg(), "xq180", offsets, mapping)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-dir", type=Path, default=SRC_PATH / "assets/motions/g1/amp")
  parser.add_argument("--output-dir", type=Path, default=None)
  parser.add_argument("--method", choices=("ik", "joint-map"), default="ik")
  parser.add_argument("--iterations", type=int, default=18)
  args = parser.parse_args()
  output_dir = args.output_dir or SRC_PATH / "assets/motions/xq180" / ("amp_ik" if args.method == "ik" else "amp")
  if args.method == "ik":
    if __package__:
      from .xq180_retarget_ik import convert_ik_directory
    else:
      from xq180_retarget_ik import convert_ik_directory
    convert_ik_directory(args.source_dir, output_dir, args.iterations)
  else:
    convert_directory(args.source_dir, output_dir, G1ToXQ180())
