"""XQ180 articulation, nominal pose and collision configuration.

Effort and velocity ratings come from the vendored URDF. PD gains use the
user's v5 table; G1 remains the angular action-scale reference.
"""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import re

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from src.assets.robots.unitree_g1.g1_constants import G1_ARTICULATION


ASSET_DIR = Path(__file__).resolve().parent
XQ180_XML = ASSET_DIR / "xmls/xq180.xml"
XQ180_URDF = ASSET_DIR / "xq180.urdf"
_JOINTS = ET.parse(XQ180_URDF).getroot().findall("joint")
XQ180_TOTAL_MASS = sum(float(m.get("value")) for m in ET.parse(XQ180_URDF).getroot().findall("link/inertial/mass"))
XQ180_JOINT_NAMES = tuple(j.attrib["name"] for j in _JOINTS)
XQ180_EFFORT_LIMITS = {j.attrib["name"]: float(j.find("limit").attrib["effort"]) for j in _JOINTS}
# URDF velocity ratings are metadata, not a hard joint velocity constraint.
XQ180_VELOCITY_LIMITS = {j.attrib["name"]: float(j.find("limit").attrib["velocity"]) for j in _JOINTS}
XQ180_FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
XQ180_FOOT_SITE_NAMES = ("left_foot", "right_foot")
XQ180_AMP_BODY_NAMES = (
  "base_link", "left_hip_roll_Link", "left_knee_Link", "left_ankle_Link",
  "right_hip_roll_Link", "right_knee_Link", "right_ankle_Link",
  "left_shoulder_roll_Link", "leftelbow_Link",
  "right_shoulder_roll_Link", "rightelbow_Link",
)
XQ180_AMP_ANCHOR_NAME = "waist_Link"
XQ180_PD_PROFILE = "user_pd_v5"
# User-specified integer (Kp, Kd), identical on left and right.
XQ180_PD_GROUP_GAINS = {
  "hip_yaw": (200, 10), "hip_roll": (400, 20), "hip_pitch": (300, 15),
  "knee": (300, 15), "ankle": (100, 5), "waist": (100, 5),
  "shoulder_pitch": (50, 2), "shoulder_roll": (50, 2),
  "shoulder_yaw": (25, 1), "elbow": (25, 1),
}


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(XQ180_XML))
  assets: dict[str, bytes] = {}
  for mesh in spec.meshes:
    assets[f"{spec.meshdir}/{mesh.file}"] = (ASSET_DIR / "meshes" / mesh.file).read_bytes()
  spec.assets = assets
  return spec


def g1_joint_name(name: str) -> str:
  if name == "waist_point":
    return "waist_yaw_joint"
  if name.endswith("_ankle_joint"):
    return name.replace("_ankle_joint", "_ankle_pitch_joint")
  if name in ("leftelbow_joint", "rightelbow_joint"):
    return name.replace("elbow_joint", "_elbow_joint")
  return name


def g1_reference_actuator(name: str):
  reference = g1_joint_name(name)
  matches = [a for a in G1_ARTICULATION.actuators
             if any(re.fullmatch(pattern, reference) for pattern in a.target_names_expr)]
  if len(matches) != 1:
    raise ValueError(f"Expected one G1 reference actuator for {name}; found {len(matches)}")
  return matches[0]


def _gains(name: str) -> tuple[float, float, float]:
  group = next(key for key in XQ180_PD_GROUP_GAINS if key in name)
  kp, kd = XQ180_PD_GROUP_GAINS[group]
  return kp, kd, 0.01


XQ180_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(name,), stiffness=_gains(name)[0], damping=_gains(name)[1],
      armature=_gains(name)[2], effort_limit=XQ180_EFFORT_LIMITS[name],
    )
    for name in XQ180_JOINT_NAMES
  ),
  soft_joint_pos_limit_factor=0.9,
)
XQ180_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.08),
  joint_pos={
    ".*_hip_pitch_joint": -0.25, ".*_knee_joint": 0.5, ".*_ankle_joint": -0.25,
    ".*_shoulder_pitch_joint": -0.1, ".*elbow_joint": -0.5,
    "left_shoulder_roll_joint": 0.15, "right_shoulder_roll_joint": -0.15,
  },
  joint_vel={".*": 0.0},
)
XQ180_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={".*_foot_collision": 3, ".*_collision": 1},
  priority={".*_foot_collision": 1},
  friction={".*_foot_collision": (0.6,)},
)
XQ180_ACTION_SCALE = {
  # Cap exploration at the corresponding G1 angular action scale.
  name: min(
    0.25 * XQ180_EFFORT_LIMITS[name] / _gains(name)[0],
    0.25 * g1_reference_actuator(name).effort_limit / g1_reference_actuator(name).stiffness,
  )
  for name in XQ180_JOINT_NAMES
}


def get_xq180_robot_cfg() -> EntityCfg:
  """Return an independent configuration; actuators follow URDF joint order."""
  return EntityCfg(
    spec_fn=get_spec, init_state=deepcopy(XQ180_HOME),
    articulation=deepcopy(XQ180_ARTICULATION), collisions=(deepcopy(XQ180_COLLISION),),
    sort_actuators=True,
  )


if __name__ == "__main__":
  import mujoco.viewer
  from mjlab.entity import Entity

  mujoco.viewer.launch(Entity(get_xq180_robot_cfg()).spec.compile())
