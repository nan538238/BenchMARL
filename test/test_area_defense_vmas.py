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

    def test_v2_zero_action_cannot_win_train_styles(self):
        if (
            importlib.util.find_spec("torchrl") is None
            or importlib.util.find_spec("vmas") is None
        ):
            self.skipTest("TorchRL and VMAS are required for the runtime smoke test")

        from benchmarl.environments.vmas.common import VmasTask

        for style in ("concentrated", "spread", "feint"):
            with self.subTest(style=style):
                task = VmasTask.AREA_DEFENSE_V2.get_task()
                task.config["opponent_style"] = style
                env = task.get_env_fun(
                    num_envs=1, continuous_actions=True, seed=7, device="cpu"
                )()

                def stay_still(td):
                    return td.update(env.full_action_spec.zero())

                try:
                    reset = env.reset()
                    self.assertEqual(
                        tuple(reset.get(("blue", "observation")).shape), (1, 3, 24)
                    )
                    rollout = env.rollout(
                        120, policy=stay_still, break_when_any_done=True
                    )
                    success = rollout.get(("next", "blue", "info", "success"))
                    breach = rollout.get(("next", "blue", "info", "breach"))
                    self.assertFalse(success.any().item())
                    self.assertTrue(breach.any().item())
                finally:
                    env.close()

    def test_v2_oracle_rule_steps(self):
        if (
            importlib.util.find_spec("torchrl") is None
            or importlib.util.find_spec("vmas") is None
        ):
            self.skipTest("TorchRL and VMAS are required for the runtime smoke test")

        from benchmarl.environments.vmas.common import VmasTask
        from examples.check_area_defense_v2_rule import oracle_lane_rule

        task = VmasTask.AREA_DEFENSE_V2.get_task()
        task.config["guidance_mode"] = "oracle"
        env = task.get_env_fun(
            num_envs=1, continuous_actions=True, seed=7, device="cpu"
        )()
        try:
            rollout = env.rollout(3, policy=oracle_lane_rule)
            action = rollout.get(("blue", "action"))
            self.assertEqual(tuple(action.shape[-2:]), (3, 2))
            self.assertLessEqual(action.abs().max().item(), 1.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
