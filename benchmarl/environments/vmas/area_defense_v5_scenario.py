"""Area-defense v5: individual pursuit credit plus shared terminal rewards."""

import torch
from torch import Tensor
from vmas.simulator.core import Agent, World

from benchmarl.environments.vmas.area_defense_v4_reward import (
    assigned_defender_progress_per_agent,
)
from benchmarl.environments.vmas.area_defense_v4_scenario import AreaDefenseV4Scenario


class AreaDefenseV5Scenario(AreaDefenseV4Scenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        world = super().make_world(batch_dim, device, **kwargs)
        self._agent_rewards = torch.zeros(batch_dim, 3, device=device)
        return world

    def reset_world_at(self, env_index=None) -> None:
        super().reset_world_at(env_index)
        index = slice(None) if env_index is None else env_index
        self._agent_rewards[index] = 0

    def _advance_and_score(self) -> None:
        active = ~self._done
        old_blue_pos = self._last_blue_pos.clone()
        old_alive = self.alive.clone()
        old_available = self.blue_available.clone()
        super()._advance_and_score()
        individual = assigned_defender_progress_per_agent(
            self.red_pos,
            old_blue_pos,
            self._last_blue_pos,
            old_alive,
            old_available,
        ) * active[:, None]
        # Preserve the v4 mean team reward while assigning movement credit
        # only to the defender responsible for that red target.
        team_shaping = 0.4 * individual.sum(dim=-1, keepdim=True)
        self._agent_rewards = self._reward[:, None] - team_shaping + 1.2 * individual

    def reward(self, agent: Agent) -> Tensor:
        if agent is self.blue[0]:
            self._advance_and_score()
        for index, blue in enumerate(self.blue):
            if agent is blue:
                return self._agent_rewards[:, index]
        raise ValueError("Unknown area-defense agent")
