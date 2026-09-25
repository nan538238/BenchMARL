"""Trainable VMAS 3v3 area defense with an optional privileged tactic signal.

Blue agents are learned VMAS agents. Three red landmarks follow scripted lanes.
The oracle supplies only a four-way team intent signal in observations; it
does not directly control blue actions. This is an information upper-bound
ablation, not a human intervention implementation.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class AreaDefenseScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.guidance_mode = kwargs.pop("guidance_mode")
        self.opponent_style = kwargs.pop("opponent_style")
        self.sensor_range = float(kwargs.pop("sensor_range"))
        self.red_speed = float(kwargs.pop("red_speed"))
        self.capture_radius = float(kwargs.pop("capture_radius"))
        self.episode_horizon = int(kwargs.pop("episode_horizon"))
        if kwargs:
            raise ValueError(f"Unexpected area-defense settings: {sorted(kwargs)}")
        if self.guidance_mode not in ("none", "oracle"):
            raise ValueError("guidance_mode must be none or oracle")
        if self.opponent_style not in (
            "mixed",
            "concentrated",
            "spread",
            "feint",
            "late_switch",
        ):
            raise ValueError("Unknown opponent_style")
        if min(self.sensor_range, self.red_speed, self.capture_radius) <= 0:
            raise ValueError(
                "Sensor range, red speed and capture radius must be positive"
            )

        world = World(batch_dim, device, x_semidim=1.2, y_semidim=1.0)
        self.blue = []
        self.red = []
        for i in range(3):
            agent = Agent(
                name=f"blue_{i}",
                collide=False,
                shape=Sphere(radius=0.05),
                u_multiplier=0.25,
                max_speed=0.3,
            )
            world.add_agent(agent)
            self.blue.append(agent)
        for i in range(3):
            landmark = Landmark(
                name=f"red_{i}",
                collide=False,
                shape=Sphere(radius=0.04),
                color=Color.RED,
            )
            world.add_landmark(landmark)
            self.red.append(landmark)

        self.lanes = torch.tensor([-0.6, 0.0, 0.6], device=device)
        self.true_lanes = torch.zeros(batch_dim, 3, dtype=torch.long, device=device)
        self.initial_lanes = self.true_lanes.clone()
        self.styles = torch.zeros(batch_dim, dtype=torch.long, device=device)
        self.red_pos = torch.zeros(batch_dim, 3, 2, device=device)
        self.alive = torch.ones(batch_dim, 3, dtype=torch.bool, device=device)
        self.steps = torch.zeros(batch_dim, dtype=torch.long, device=device)
        self._done = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self._success = self._done.clone()
        self._breach = self._done.clone()
        self._captures = torch.zeros(batch_dim, device=device)
        self._reward = torch.zeros(batch_dim, device=device)
        self._last_dist = torch.zeros(batch_dim, 3, device=device)
        return world

    def _blue_red_distance(self) -> Tensor:
        blue_pos = torch.stack([agent.state.pos for agent in self.blue], dim=1)
        return torch.cdist(self.red_pos, blue_pos).amin(dim=-1)

    def reset_world_at(self, env_index: Optional[int] = None) -> None:
        batch = self.world.batch_dim if env_index is None else 1
        device = self.world.device
        if self.opponent_style == "mixed":
            styles = torch.randint(0, 3, (batch,), device=device)
        else:
            style_id = {"concentrated": 0, "spread": 1, "feint": 2, "late_switch": 3}[
                self.opponent_style
            ]
            styles = torch.full((batch,), style_id, dtype=torch.long, device=device)
        focus = torch.randint(0, 3, (batch,), device=device)
        if env_index is None:
            self.styles[:] = styles
        else:
            self.styles[env_index] = styles[0]

        true_lanes = focus[:, None].expand(-1, 3).clone()
        spread = styles == 1
        true_lanes[spread] = torch.arange(3, device=device)
        initial_lanes = true_lanes.clone()
        switched = styles >= 2
        initial_lanes[switched] = ((focus[switched] + 1) % 3)[:, None]

        index = slice(None) if env_index is None else env_index
        self.true_lanes[index] = true_lanes if env_index is None else true_lanes[0]
        self.initial_lanes[index] = (
            initial_lanes if env_index is None else initial_lanes[0]
        )
        self.alive[index] = True
        self.steps[index] = 0
        self._done[index] = False
        self._success[index] = False
        self._breach[index] = False
        self._captures[index] = 0
        self._reward[index] = 0

        for i, agent in enumerate(self.blue):
            pos = torch.tensor([-0.62, float(self.lanes[i])], device=device).expand(
                batch, 2
            )
            agent.set_pos(pos, batch_index=env_index)
        for i, red in enumerate(self.red):
            pos = torch.stack(
                (
                    torch.full((batch,), 0.95 + 0.04 * i, device=device),
                    self.lanes[initial_lanes[:, i]],
                ),
                dim=-1,
            )
            red.set_pos(pos, batch_index=env_index)
            self.red_pos[index, i] = pos if env_index is None else pos[0]
        self._last_dist[index] = self._blue_red_distance()[index]

    def _advance_and_score(self) -> None:
        active = ~self._done
        self.steps[active] += 1
        for i, red in enumerate(self.red):
            moving = active & self.alive[:, i]
            target_lane = self.true_lanes[:, i]
            switch_step = torch.where(self.styles == 3, 45, 25)
            before_switch = self.steps < switch_step
            target_lane = torch.where(
                before_switch, self.initial_lanes[:, i], target_lane
            )
            target_y = self.lanes[target_lane]
            self.red_pos[moving, i, 0] -= self.red_speed
            delta_y = (target_y - self.red_pos[:, i, 1]).clamp(-0.035, 0.035)
            self.red_pos[moving, i, 1] += delta_y[moving]
            red.set_pos(self.red_pos[:, i])

        distances = self._blue_red_distance()
        captured = (distances <= self.capture_radius) & self.alive & active[:, None]
        self.alive &= ~captured
        capture_count = captured.sum(dim=-1).to(torch.float)
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

        shaping = ((self._last_dist - distances) * self.alive).sum(-1)
        self._reward = active.float() * (0.4 * shaping + 1.5 * capture_count - 0.005)
        self._reward += 5.0 * success.float() - 5.0 * breach.float()
        self._last_dist = distances

    def reward(self, agent: Agent) -> Tensor:
        if agent is self.blue[0]:
            self._advance_and_score()
        return self._reward

    def observation(self, agent: Agent) -> Tensor:
        own = torch.cat((agent.state.pos, agent.state.vel), dim=-1)
        team = torch.cat(
            [
                other.state.pos - agent.state.pos
                for other in self.blue
                if other is not agent
            ],
            dim=-1,
        )
        relative_red = self.red_pos - agent.state.pos[:, None, :]
        visible = (relative_red.norm(dim=-1) <= self.sensor_range) & self.alive
        red_obs = torch.cat(
            (relative_red * visible[:, :, None], visible[:, :, None].float()), dim=-1
        ).flatten(1)
        if self.guidance_mode == "oracle":
            # Index 3 means red is split over all three lanes.
            hint_id = torch.where(self.styles == 1, 3, self.true_lanes[:, 0])
            hint = F.one_hot(hint_id, num_classes=4).float()
        else:
            hint = torch.zeros(self.world.batch_dim, 4, device=self.world.device)
        return torch.cat((own, team, red_obs, hint), dim=-1)

    def done(self) -> Tensor:
        return self._done

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        signature = (
            self.styles * 27
            + self.true_lanes[:, 0] * 9
            + self.true_lanes[:, 1] * 3
            + self.true_lanes[:, 2]
        )
        return {
            "success": self._success.float(),
            "breach": self._breach.float(),
            "captures": self._captures,
            "scenario_id": signature.float(),
        }
