"""Area-defense v3: reward pursuit caused by defender motion only."""

from typing import Optional

import torch
from vmas.simulator.core import World

from benchmarl.environments.vmas.area_defense_v2_scenario import AreaDefenseV2Scenario
from benchmarl.environments.vmas.area_defense_v3_reward import defender_progress


class AreaDefenseV3Scenario(AreaDefenseV2Scenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        world = super().make_world(batch_dim, device, **kwargs)
        self._last_blue_pos = torch.zeros(batch_dim, 3, 2, device=device)
        return world

    def reset_world_at(self, env_index: Optional[int] = None) -> None:
        super().reset_world_at(env_index)
        index = slice(None) if env_index is None else env_index
        positions = torch.stack([agent.state.pos for agent in self.blue], dim=1)
        self._last_blue_pos[index] = positions[index]

    def _advance_and_score(self) -> None:
        active = ~self._done
        self.steps[active] += 1
        for i, red in enumerate(self.red):
            moving = active & self.alive[:, i]
            target_lane = self.true_lanes[:, i]
            switch_step = torch.where(self.styles == 3, 45, 25)
            target_lane = torch.where(
                self.steps < switch_step, self.initial_lanes[:, i], target_lane
            )
            target_y = self.lanes[target_lane]
            self.red_pos[moving, i, 0] -= self.red_speed
            delta_y = (target_y - self.red_pos[:, i, 1]).clamp(-0.035, 0.035)
            self.red_pos[moving, i, 1] += delta_y[moving]
            red.set_pos(self.red_pos[:, i], batch_index=None)

        blue_pos = torch.stack([agent.state.pos for agent in self.blue], dim=1)
        shaping = defender_progress(
            self.red_pos, self._last_blue_pos, blue_pos, self.alive, self.blue_available
        )
        distances = self._blue_red_distance()
        captured = self._capture(distances, active)
        self.alive &= ~captured
        capture_count = captured.sum(dim=-1).float()
        self._captures = capture_count
        breach = ((self.red_pos[:, :, 0] <= -0.92) & self.alive).any(-1) & active
        success = (
            ((~self.alive).all(-1) | (self.steps >= self.episode_horizon))
            & ~breach
            & active
        )
        self._breach = breach
        self._success = success
        self._done |= breach | success

        self._reward = active.float() * (0.4 * shaping + 1.5 * capture_count - 0.005)
        self._reward += 5.0 * success.float() - 5.0 * breach.float()
        self._last_dist = distances
        self._last_blue_pos = blue_pos.clone()
