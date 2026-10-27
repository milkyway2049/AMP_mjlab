"""XQ180 locomotion rewards and fall detection without ground recovery."""

import torch


def root_height_above_origin(env):
  """Root height relative to the terrain tile origin (not absolute world Z)."""
  return env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def locomotion_height_reward(env, desired_height: float, std: float):
  return torch.exp(-torch.square((root_height_above_origin(env) - desired_height) / std))


def fallen_height(env, minimum_height: float):
  return root_height_above_origin(env) < minimum_height
