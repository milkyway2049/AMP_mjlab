"""Hunter180 articulation, nominal pose and collision configuration.

Effort and velocity ratings come from the vendored URDF. For the heavier robot,
PD gains use the requested 3x Hunter130 baseline; armatures remain estimates.
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
from src.assets.robots.hunter130.hunter130_constants import HUNTER130_URDF


ASSET_DIR = Path(__file__).resolve().parent
HUNTER180_XML = ASSET_DIR / "xmls/hunter180.xml"
HUNTER180_URDF = ASSET_DIR / "hunter180.urdf"
_JOINTS = ET.parse(HUNTER180_URDF).getroot().findall("joint")
HUNTER180_TOTAL_MASS = sum(float(m.get("value")) for m in ET.parse(HUNTER180_URDF).getroot().findall("link/inertial/mass"))
HUNTER130_REFERENCE_MASS = sum(float(m.get("value")) for m in ET.parse(HUNTER130_URDF).getroot().findall("link/inertial/mass"))
HUNTER180_MASS_RATIO = HUNTER180_TOTAL_MASS / HUNTER130_REFERENCE_MASS
# User-selected multiplier, above the 2.084 total-mass ratio.
HUNTER180_PD_SCALE = 3.0
HUNTER180_JOINT_NAMES = tuple(j.attrib["name"] for j in _JOINTS)
HUNTER180_EFFORT_LIMITS = {j.attrib["name"]: float(j.find("limit").attrib["effort"]) for j in _JOINTS}
# URDF velocity ratings are metadata, not a hard joint velocity constraint.
HUNTER180_VELOCITY_LIMITS = {j.attrib["name"]: float(j.find("limit").attrib["velocity"]) for j in _JOINTS}
HUNTER180_FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
HUNTER180_FOOT_SITE_NAMES = ("left_foot", "right_foot")
HUNTER180_AMP_BODY_NAMES = (
  "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
  "left_shoulder_roll_link", "left_elbow_link",
  "right_shoulder_roll_link", "right_elbow_link",
)
HUNTER180_AMP_ANCHOR_NAME = "torso_link"


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(HUNTER180_XML))
  assets: dict[str, bytes] = {}
  for mesh in spec.meshes:
    assets[f"{spec.meshdir}/{mesh.file}"] = (ASSET_DIR / "meshes" / mesh.file).read_bytes()
  spec.assets = assets
  return spec


def _gains(name: str) -> tuple[float, float, float]:
  matches = [
    actuator for actuator in G1_ARTICULATION.actuators
    if any(re.fullmatch(pattern, name) for pattern in actuator.target_names_expr)
  ]
  if len(matches) != 1:
    raise ValueError(f"Expected one G1 PD configuration for {name}; found {len(matches)}")
  actuator = matches[0]
  return (
    actuator.stiffness * HUNTER180_PD_SCALE,
    actuator.damping * HUNTER180_PD_SCALE,
    actuator.armature,
  )


HUNTER180_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(name,), stiffness=_gains(name)[0], damping=_gains(name)[1],
      armature=_gains(name)[2], effort_limit=HUNTER180_EFFORT_LIMITS[name],
    )
    for name in HUNTER180_JOINT_NAMES
  ),
  soft_joint_pos_limit_factor=0.9,
)
HUNTER180_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.925),
  joint_pos={
    ".*_hip_pitch_joint": -0.25, ".*_knee_joint": 0.5, ".*_ankle_pitch_joint": -0.25,
    ".*_shoulder_pitch_joint": -0.1, ".*_elbow_joint": -0.5,
    "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)
HUNTER180_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={".*_foot_collision": 3, ".*_collision": 1},
  priority={".*_foot_collision": 1},
  friction={".*_foot_collision": (0.6,)},
)
HUNTER180_ACTION_SCALE = {
  name: 0.25 * HUNTER180_EFFORT_LIMITS[name] / _gains(name)[0]
  for name in HUNTER180_JOINT_NAMES
}


def get_hunter180_robot_cfg() -> EntityCfg:
  """Return an independent configuration; actuators follow URDF joint order."""
  return EntityCfg(
    spec_fn=get_spec, init_state=deepcopy(HUNTER180_HOME),
    articulation=deepcopy(HUNTER180_ARTICULATION), collisions=(deepcopy(HUNTER180_COLLISION),),
    sort_actuators=True,
  )


if __name__ == "__main__":
  import mujoco.viewer
  from mjlab.entity import Entity

  mujoco.viewer.launch(Entity(get_hunter180_robot_cfg()).spec.compile())
