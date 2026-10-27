"""Hunter130 articulation, nominal pose and collision configuration.

Effort and velocity ratings come from the vendored URDF. PD gains use the
requested Hunter130 values; estimated armatures retain rounded G1 baselines.
"""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import re

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


ASSET_DIR = Path(__file__).resolve().parent
HUNTER130_XML = ASSET_DIR / "xmls/hunter130.xml"
HUNTER130_URDF = ASSET_DIR / "hunter130.urdf"
_JOINTS = ET.parse(HUNTER130_URDF).getroot().findall("joint")
HUNTER130_JOINT_NAMES = tuple(j.attrib["name"] for j in _JOINTS)
HUNTER130_EFFORT_LIMITS = {j.attrib["name"]: float(j.find("limit").attrib["effort"]) for j in _JOINTS}
# URDF velocity ratings are metadata, not a hard joint velocity constraint.
HUNTER130_VELOCITY_LIMITS = {j.attrib["name"]: float(j.find("limit").attrib["velocity"]) for j in _JOINTS}
HUNTER130_FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
HUNTER130_FOOT_SITE_NAMES = ("left_foot", "right_foot")
HUNTER130_AMP_BODY_NAMES = (
  "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_roll_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_roll_link",
)
HUNTER130_AMP_ANCHOR_NAME = "torso_link"

# (Kp [Nm/rad], Kd [Nm*s/rad], armature [kg*m^2]). Keep the requested
# values explicit so training, exported metadata and deployment agree.
_MOTOR_PARAMS = {
  r".*_(hip_pitch|hip_roll|knee)_joint": (100.0, 8.0, 0.0251),
  r".*_hip_yaw_joint": (100.0, 8.0, 0.0102),
  r".*_ankle_(pitch|roll)_joint": (30.0, 5.0, 0.0072),
  r"waist_yaw_joint": (60.0, 5.0, 0.0102),
  r"waist_(pitch|roll)_joint": (60.0, 5.0, 0.0072),
  r".*_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint": (25.0, 1.0, 0.0036),
}


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(HUNTER130_XML))
  assets: dict[str, bytes] = {}
  for mesh in spec.meshes:
    assets[f"{spec.meshdir}/{mesh.file}"] = (ASSET_DIR / "meshes" / mesh.file).read_bytes()
  spec.assets = assets
  return spec


def _gains(name: str) -> tuple[float, float, float]:
  matches = [
    params for pattern, params in _MOTOR_PARAMS.items() if re.fullmatch(pattern, name)
  ]
  if len(matches) != 1:
    raise ValueError(f"Expected one Hunter130 motor configuration for {name}; found {len(matches)}")
  return matches[0]


HUNTER130_ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(name,), stiffness=_gains(name)[0], damping=_gains(name)[1],
      armature=_gains(name)[2], effort_limit=HUNTER130_EFFORT_LIMITS[name],
    )
    for name in HUNTER130_JOINT_NAMES
  ),
  soft_joint_pos_limit_factor=0.9,
)
HUNTER130_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.68),
  joint_pos={
    ".*_hip_pitch_joint": -0.1, ".*_knee_joint": 0.3, ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2, ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.18, "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)
HUNTER130_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={".*_foot_collision": 3, ".*_collision": 1},
  priority={".*_foot_collision": 1},
  friction={".*_foot_collision": (0.6,)},
)
HUNTER130_ACTION_SCALE = {
  name: round(0.25 * HUNTER130_EFFORT_LIMITS[name] / _gains(name)[0], 3)
  for name in HUNTER130_JOINT_NAMES
}


def get_hunter130_robot_cfg() -> EntityCfg:
  """Return an independent configuration; actuators follow URDF joint order."""
  return EntityCfg(
    spec_fn=get_spec, init_state=deepcopy(HUNTER130_HOME),
    articulation=deepcopy(HUNTER130_ARTICULATION), collisions=(deepcopy(HUNTER130_COLLISION),),
    sort_actuators=True,
  )


if __name__ == "__main__":
  import mujoco.viewer
  from mjlab.entity import Entity

  mujoco.viewer.launch(Entity(get_hunter130_robot_cfg()).spec.compile())
