"""Vendor XQ180 URDF/meshes and generate its 19-joint training MJCF."""

import argparse
from pathlib import Path

if __package__:
  from .convert_hunter130 import convert
else:
  from convert_hunter130 import convert

SOURCE_DIR = Path('/home/milkyway/opensource/hunter130_train/legged_lab/assets/xq180')
ASSET_DIR = Path(__file__).resolve().parents[1] / 'src/assets/robots/xq180'


def convert_xq180(source_dir: Path = SOURCE_DIR, output_dir: Path = ASSET_DIR) -> Path:
  return convert(
    source_dir, output_dir, robot_name='xq180', urdf_relative_path='urdf/xq180.urdf',
    root_height=1.08, expected_joint_count=19, root_name='base_link',
    foot_link_suffix='ankle_Link', torso_name='waist_Link',
    repair_inertia=True,
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR)
  parser.add_argument('--output-dir', type=Path, default=ASSET_DIR)
  args = parser.parse_args()
  print(convert_xq180(args.source_dir, args.output_dir))
