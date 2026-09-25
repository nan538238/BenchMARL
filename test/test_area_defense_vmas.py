"""Runtime smoke checks for the trainable VMAS area-defense task."""

import importlib.util
import unittest


class AreaDefenseVmasTest(unittest.TestCase):
    def test_both_guidance_modes_step_with_fixed_observation_shape(self):
        if (
            importlib.util.find_spec("torchrl") is None
            or importlib.util.find_spec("vmas") is None
        ):
            self.skipTest("TorchRL and VMAS are required for the runtime smoke test")

        from torchrl.envs.libs.vmas import VmasEnv

        from benchmarl.environments.vmas.area_defense_scenario import (
            AreaDefenseScenario,
        )

        for mode in ("none", "oracle"):
            with self.subTest(mode=mode):
                env = VmasEnv(
                    scenario=AreaDefenseScenario(),
                    num_envs=2,
                    max_steps=120,
                    seed=7,
                    device="cpu",
                    continuous_actions=True,
                    categorical_actions=True,
                    clamp_actions=True,
                    guidance_mode=mode,
                    opponent_style="mixed",
                    sensor_range=0.65,
                    red_speed=0.022,
                    capture_radius=0.11,
                    episode_horizon=120,
                )
                try:
                    reset = env.reset()
                    self.assertEqual(
                        tuple(reset.get(("blue", "observation")).shape), (2, 3, 21)
                    )
                    rollout = env.rollout(3)
                    self.assertIn(("next", "blue", "reward"), rollout.keys(True))
                    self.assertIn(
                        ("next", "blue", "info", "success"), rollout.keys(True)
                    )
                    self.assertIn(
                        ("next", "blue", "info", "scenario_id"), rollout.keys(True)
                    )
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
