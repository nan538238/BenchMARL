"""Check v3 movement-based shaping and its VMAS interface."""

import importlib.util
import unittest
from pathlib import Path

import torch


class AreaDefenseV3Test(unittest.TestCase):
    def test_progress_excludes_red_motion_and_spent_defenders(self):
        reward_file = (
            Path(__file__).resolve().parents[1]
            / "benchmarl/environments/vmas/area_defense_v3_reward.py"
        )
        spec = importlib.util.spec_from_file_location("area_defense_v3_reward", reward_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        defender_progress = module.defender_progress

        red_pos = torch.tensor([[[0.5, 0.0]]])
        old_blue = torch.tensor([[[-0.5, 0.0], [-0.5, 0.0]]])
        alive = torch.tensor([[True]])
        available = torch.tensor([[True, False]])

        no_blue_motion = defender_progress(
            red_pos, old_blue, old_blue, alive, available
        )
        self.assertEqual(no_blue_motion.item(), 0.0)

        spent_only_moves = old_blue.clone()
        spent_only_moves[0, 1, 0] = 0.4
        self.assertEqual(
            defender_progress(
                red_pos, old_blue, spent_only_moves, alive, available
            ).item(),
            0.0,
        )

        active_moves = old_blue.clone()
        active_moves[0, 0, 0] = -0.4
        self.assertAlmostEqual(
            defender_progress(red_pos, old_blue, active_moves, alive, available)
            .item(),
            0.1,
            places=5,
        )
        self.assertEqual(
            defender_progress(
                red_pos, old_blue, active_moves, alive, torch.zeros_like(available)
            ).item(),
            0.0,
        )

    def test_v3_environment_steps(self):
        if (
            importlib.util.find_spec("torchrl") is None
            or importlib.util.find_spec("vmas") is None
        ):
            self.skipTest("TorchRL and VMAS are required for the runtime smoke test")

        from benchmarl.environments.vmas.common import VmasTask

        for mode in ("none", "oracle"):
            with self.subTest(mode=mode):
                task = VmasTask.AREA_DEFENSE_V3.get_task()
                task.config["guidance_mode"] = mode
                env = task.get_env_fun(
                    num_envs=2, continuous_actions=True, seed=7, device="cpu"
                )()
                try:
                    reset = env.reset()
                    self.assertEqual(
                        tuple(reset.get(("blue", "observation")).shape),
                        (2, 3, 24),
                    )
                    rollout = env.rollout(
                        1,
                        policy=lambda td: td.update(env.full_action_spec.zero()),
                    )
                    self.assertTrue(
                        torch.isfinite(rollout.get(("next", "blue", "reward"))).all()
                    )
                    torch.testing.assert_close(
                        rollout.get(("next", "blue", "reward")),
                        torch.full((2, 1, 3, 1), -0.005),
                        atol=1e-5,
                        rtol=0,
                    )
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
