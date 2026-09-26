"""Check indexed pursuit shaping and the v4 VMAS task interface."""

import importlib.util
import unittest
from pathlib import Path

import torch


class AreaDefenseV4Test(unittest.TestCase):
    def test_assigned_progress_rewards_lane_coverage(self):
        reward_file = (
            Path(__file__).resolve().parents[1]
            / "benchmarl/environments/vmas/area_defense_v4_reward.py"
        )
        spec = importlib.util.spec_from_file_location("area_defense_v4_reward", reward_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        progress = module.assigned_defender_progress

        red = torch.tensor([[[0.0, -0.6], [0.0, 0.0], [0.0, 0.6]]])
        old = torch.tensor([[[-0.6, -0.12], [-0.6, 0.0], [-0.6, 0.12]]])
        alive = torch.ones(1, 3, dtype=torch.bool)
        available = alive.clone()
        distributed = old.clone()
        distributed[0, 0, 1] -= 0.1
        distributed[0, 2, 1] += 0.1
        collapsed = old.clone()
        collapsed[:, :, 1] += 0.1

        self.assertGreater(progress(red, old, distributed, alive, available).item(), 0)
        self.assertLess(
            progress(red, old, collapsed, alive, available).item(),
            progress(red, old, distributed, alive, available).item(),
        )
        self.assertEqual(
            progress(red, old, distributed, alive, torch.zeros_like(available)).item(),
            0,
        )

    def test_v4_environment_steps(self):
        if (
            importlib.util.find_spec("torchrl") is None
            or importlib.util.find_spec("vmas") is None
        ):
            self.skipTest("TorchRL and VMAS are required for the runtime smoke test")

        from benchmarl.environments.vmas.common import VmasTask

        task = VmasTask.AREA_DEFENSE_V4.get_task()
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
