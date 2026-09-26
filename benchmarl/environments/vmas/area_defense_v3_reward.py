"""Movement-only reward shaping for area defense v3."""

import torch
from torch import Tensor


def defender_progress(
    red_pos: Tensor,
    old_blue_pos: Tensor,
    new_blue_pos: Tensor,
    alive: Tensor,
    available: Tensor,
) -> Tensor:
    """Change in red-to-nearest-eligible-blue distance at fixed red positions."""
    old_dist = torch.cdist(red_pos, old_blue_pos).masked_fill(
        ~available[:, None, :], float("inf")
    )
    new_dist = torch.cdist(red_pos, new_blue_pos).masked_fill(
        ~available[:, None, :], float("inf")
    )
    has_defender = available.any(dim=-1, keepdim=True)
    old_nearest = torch.where(has_defender, old_dist.amin(dim=-1), 0.0)
    new_nearest = torch.where(has_defender, new_dist.amin(dim=-1), 0.0)
    return ((old_nearest - new_nearest) * alive).sum(dim=-1)
