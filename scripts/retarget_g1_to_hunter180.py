"""Convert G1 AMP motions to Hunter180, including its elbow rest-frame offset."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

if __package__:
  from .retarget_g1_to_hunter130 import G1ToHunter130, convert_directory
else:
  from retarget_g1_to_hunter130 import G1ToHunter130, convert_directory

from src import SRC_PATH
from src.assets.robots.hunter180.hunter180_constants import HUNTER180_URDF, get_hunter180_robot_cfg


class G1ToHunter180(G1ToHunter130):
  def __init__(self):
    # G1 elbow rest frames have zero rotation; Hunter180 has R_y(1.57).
    # Both hinges use +Y, so R_y(rest + q_target) = R_y(q_source).
    urdf = ET.parse(HUNTER180_URDF).getroot()
    offsets = {}
    for side in ("left", "right"):
      name = f"{side}_elbow_joint"
      joint = next(j for j in urdf.findall("joint") if j.get("name") == name)
      rpy = np.fromstring(joint.find("origin").get("rpy", "0 0 0"), sep=" ")
      axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
      if not np.allclose(rpy[[0, 2]], 0) or not np.allclose(axis, [0, 1, 0]):
        raise ValueError(f"Cannot compensate non-Y elbow rest frame: {name}")
      offsets[name] = -float(rpy[1])
    super().__init__(get_hunter180_robot_cfg(), "hunter180", offsets)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-dir", type=Path, default=SRC_PATH / "assets/motions/g1/amp")
  parser.add_argument("--output-dir", type=Path, default=SRC_PATH / "assets/motions/hunter180/amp")
  args = parser.parse_args()
  convert_directory(args.source_dir, args.output_dir, G1ToHunter180())
