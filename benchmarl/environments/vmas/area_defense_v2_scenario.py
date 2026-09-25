"""Area-defense v2: each clustered defender can intercept once per episode."""

from typing import Optional

import torch
from torch import Tensor
from vmas.simulator.core import World

from benchmarl.environments.vmas.area_defense_scenario import AreaDefenseScenario


class AreaDefenseV2Scenario(AreaDefenseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        world = super().make_world(batch_dim, device, **kwargs)
        self.blue_available = torch.ones(batch_dim, 3, dtype=torch.bool, device=device)
        return world

    def reset_world_at(self, env_index: Optional[int] = None) -> None:
        super().reset_world_at(env_index)
        index = slice(None) if env_index is None else env_index
        self.blue_available[index] = True
        batch = self.world.batch_dim if env_index is None else 1
        for i, agent in enumerate(self.blue):
            pos = torch.tensor(
                [-0.62, 0.12 * (i - 1)], device=self.world.device
            ).expand(batch, 2)
            agent.set_pos(pos, batch_index=env_index)
        self._last_dist[index] = self._blue_red_distance()[index]

    def _capture(self, distances: Tensor, active: Tensor) -> Tensor:
        # One-to-one matching: each blue defender can capture only one red agent.
        # Red-index order breaks ties deterministically and is fixed across modes.
        positions = torch.stack([agent.state.pos for agent in self.blue], dim=1)
        distances_to_blue = torch.cdist(self.red_pos, positions)
        available = self.blue_available.clone()
        captured = torch.zeros_like(self.alive)
        batch_indices = torch.arange(self.world.batch_dim, device=self.world.device)
        for red_index in range(3):
            eligible = (
                available
                & active[:, None]
                & self.alive[:, red_index, None]
                & (distances_to_blue[:, red_index] <= self.capture_radius)
            )
            winner = (
                distances_to_blue[:, red_index]
                .masked_fill(~eligible, float("inf"))
                .argmin(dim=-1)
            )
            caught = eligible.any(dim=-1)
            captured[:, red_index] = caught
            available[batch_indices[caught], winner[caught]] = False
        self.blue_available = available
        return captured

    def observation(self, agent) -> Tensor:
        return torch.cat(
            (super().observation(agent), self.blue_available.float()), dim=-1
        )
