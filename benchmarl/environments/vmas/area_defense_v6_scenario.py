"""Area-defense v6: expose a stable role ID to the shared blue policy."""

import torch
from torch import Tensor
from vmas.simulator.core import Agent

from benchmarl.environments.vmas.area_defense_v5_scenario import AreaDefenseV5Scenario


class AreaDefenseV6Scenario(AreaDefenseV5Scenario):
    def observation(self, agent: Agent) -> Tensor:
        observation = super().observation(agent)
        for index, blue in enumerate(self.blue):
            if agent is blue:
                role = torch.eye(3, device=self.world.device)[index]
                return torch.cat(
                    (observation, role.expand(self.world.batch_dim, -1)), dim=-1
                )
        raise ValueError("Unknown area-defense agent")
