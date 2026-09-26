"""Check individual movement credit and the v5 VMAS interface."""

import importlib.util
import unittest
from pathlib import Path

import torch


class AreaDefenseV5Test(unittest.TestCase):
    def test_individual_progress_preserves_team_total(self):
        reward_file = (
            Path(__file__).resolve().parents[1]
            / "benchmarl/environments/vmas/area_defense_v4_reward.py"
        )
        spec = importlib.util.spec_from_file_location("area_defense_v4_reward", reward_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        old = torch.tensor([[[-0.6, -0.1], [-0.6, 0.0], [-0.6, 0.1]]])
        new = old.clone()
        new[0, 0, 1] -= 0.1
        new[0, 2, 1] += 0.1
        red = torch.tensor([[[0.0, -0.6], [0.0, 0.0], [0.0, 0.6]]])
        alive = torch.ones(1, 3, dtype=torch.bool)
        available = alive.clone()
        individual = module.assigned_defender_progress_per_agent(
            red, old, new, alive, available
        )
        team = module.assigned_defender_progress(red, old, new, alive, available)
        torch.testing.assert_close(individual.sum(dim=-1), team)
        self.assertGreater(individual[0, 0].item(), 0)
        self.assertEqual(individual[0, 1].item(), 0)
        self.assertGreater(individual[0, 2].item(), 0)

    def test_v5_environment_steps(self):
        if (
            importlib.util.find_spec("torchrl") is None
            or importlib.util.find_spec("vmas") is None
        ):
            self.skipTest("TorchRL and VMAS are required for the runtime smoke test")

        from benchmarl.environments.vmas.common import VmasTask

        task = VmasTask.AREA_DEFENSE_V5.get_task()
        task.config["opponent_style"] = "spread"
        env = task.get_env_fun(
            num_envs=2, continuous_actions=True, seed=7, device="cpu"
        )()
        try:
            reset = env.reset()
            self.assertEqual(tuple(reset.get(("blue", "observation")).shape), (2, 3, 24))
            rollout = env.rollout(3)
            self.assertTrue(
                torch.isfinite(rollout.get(("next", "blue", "reward"))).all()
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
