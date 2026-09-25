"""Dependency-light checks for the oracle lane-assignment diagnostic."""

import unittest

import torch

from examples.check_area_defense_v2_rule import oracle_lane_rule


class FakeTensorDict:
    def __init__(self, observation):
        self.values = {("blue", "observation"): observation}

    def get(self, key):
        return self.values[key]

    def set(self, key, value):
        self.values[key] = value
        return self


class AreaDefenseRuleTest(unittest.TestCase):
    def test_focus_hint_moves_all_defenders_toward_focus(self):
        observation = torch.zeros(1, 3, 24)
        observation[0, :, 0] = -0.62
        observation[0, :, 1] = torch.tensor([-0.12, 0.0, 0.12])
        observation[0, :, 17] = 1.0
        action = oracle_lane_rule(FakeTensorDict(observation)).get(("blue", "action"))
        self.assertEqual(tuple(action.shape), (1, 3, 2))
        self.assertTrue((action[0, :, 1] < 0).all().item())
        self.assertLessEqual(action.abs().max().item(), 1.0)

    def test_spread_hint_assigns_three_lanes(self):
        observation = torch.zeros(1, 3, 24)
        observation[0, :, 0] = -0.62
        observation[0, :, 20] = 1.0
        action = oracle_lane_rule(FakeTensorDict(observation)).get(("blue", "action"))
        self.assertLess(action[0, 0, 1].item(), 0)
        self.assertEqual(action[0, 1, 1].item(), 0)
        self.assertGreater(action[0, 2, 1].item(), 0)


if __name__ == "__main__":
    unittest.main()
