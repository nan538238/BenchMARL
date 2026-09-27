"""Check stable role IDs and the v6 VMAS task interface."""

import importlib.util
import unittest

import torch


class AreaDefenseV6Test(unittest.TestCase):
    def test_v6_observation_contains_distinct_role_ids(self):
        if (
            importlib.util.find_spec("torchrl") is None
            or importlib.util.find_spec("vmas") is None
        ):
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


if __name__ == "__main__":
    unittest.main()
