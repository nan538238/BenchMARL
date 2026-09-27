"""Check stable role IDs and the v6 VMAS task interface."""

import importlib.util
import unittest

import torch


class AreaDefenseV6Test(unittest.TestCase):
    @staticmethod
    def _requirements_available():
        return (
            importlib.util.find_spec("torchrl") is not None
            and importlib.util.find_spec("vmas") is not None
        )

    def test_v6_observation_contains_distinct_role_ids(self):
        if not self._requirements_available():
            self.skipTest("TorchRL and VMAS are required for the runtime smoke test")

        from benchmarl.environments.vmas.common import VmasTask

        task = VmasTask.AREA_DEFENSE_V6.get_task()
        task.config["opponent_style"] = "spread"
        env = task.get_env_fun(
            num_envs=2, continuous_actions=True, seed=7, device="cpu"
        )()
        try:
            reset = env.reset()
            observation = reset.get(("blue", "observation"))
            self.assertEqual(tuple(observation.shape), (2, 3, 27))
            expected_roles = torch.eye(3).expand(2, -1, -1)
            torch.testing.assert_close(observation[..., 24:27], expected_roles)
            rollout = env.rollout(3)
            self.assertTrue(
                torch.isfinite(rollout.get(("next", "blue", "reward"))).all()
            )
        finally:
            env.close()

    def test_lower_role_gets_more_reward_for_moving_down(self):
        if not self._requirements_available():
            self.skipTest("TorchRL and VMAS are required for the runtime smoke test")

        from benchmarl.environments.vmas.common import VmasTask

        rewards = {}
        for label, y_action in (("down", -1.0), ("still", 0.0), ("up", 1.0)):
            task = VmasTask.AREA_DEFENSE_V6.get_task()
            task.config["opponent_style"] = "spread"
            env = task.get_env_fun(
                num_envs=1, continuous_actions=True, seed=19, device="cpu"
            )()
            try:
                td = env.reset()
                action = env.full_action_spec.zero()
                action.get(("blue", "action"))[0, 0, 1] = y_action
                stepped = env.step(td.update(action))
                rewards[label] = stepped.get(("next", "blue", "reward"))[
                    0, 0, 0
                ].item()
            finally:
                env.close()

        self.assertGreater(rewards["down"], rewards["still"])
        self.assertGreater(rewards["still"], rewards["up"])


if __name__ == "__main__":
    unittest.main()
