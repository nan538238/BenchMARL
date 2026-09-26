"""Role-matched movement shaping for a fixed-spread learning diagnostic."""

import torch
from torch import Tensor


def assigned_defender_progress(
    red_pos: Tensor,
    old_blue_pos: Tensor,
    new_blue_pos: Tensor,
    alive: Tensor,
    available: Tensor,
) -> Tensor:
    """Credit each eligible blue agent for approaching its indexed red target."""
    old_distance = torch.linalg.vector_norm(red_pos - old_blue_pos, dim=-1)
    new_distance = torch.linalg.vector_norm(red_pos - new_blue_pos, dim=-1)
    return ((old_distance - new_distance) * alive * available).sum(dim=-1)
