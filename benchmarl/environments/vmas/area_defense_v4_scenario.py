"""Area-defense v4 diagnostic with indexed blue-to-red pursuit shaping."""

import torch

from benchmarl.environments.vmas.area_defense_v3_scenario import AreaDefenseV3Scenario
from benchmarl.environments.vmas.area_defense_v4_reward import (
    assigned_defender_progress,
)


class AreaDefenseV4Scenario(AreaDefenseV3Scenario):
    def _movement_progress(self, blue_pos: torch.Tensor) -> torch.Tensor:
        return assigned_defender_progress(
            self.red_pos, self._last_blue_pos, blue_pos, self.alive, self.blue_available
        )
