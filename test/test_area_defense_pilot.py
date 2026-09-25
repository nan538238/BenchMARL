import unittest

from examples.area_defense_pilot import (
    Config,
    MODES,
    evaluate,
    make_scenario,
    run_episode,
)


class AreaDefensePilotTest(unittest.TestCase):
    def test_paired_scenarios_and_repeatability(self):
        config = Config(episodes=12, seed=7, bootstrap_samples=100)
        first, summary = evaluate(config)
        second, same_summary = evaluate(config)
        self.assertEqual(first, second)
        self.assertEqual(summary, same_summary)
        self.assertEqual(len(first), 12 * len(MODES))
        for episode in range(12):
            group = first[episode * len(MODES) : (episode + 1) * len(MODES)]
            self.assertEqual({row.scenario_seed for row in group}, {7 + episode * 1009})
            self.assertEqual({row.family for row in group}, {group[0].family})

    def test_budget_and_baseline(self):
        config = Config(episodes=9, seed=5, budget=1, bootstrap_samples=50)
        rows, summary = evaluate(config)
        self.assertTrue(all(row.calls == 0 for row in rows if row.mode == "auto"))
        self.assertTrue(
            all(row.calls <= 1 for row in rows if row.mode == "oracle_budget")
        )
        self.assertEqual(summary["modes"]["auto"]["paired_gain_vs_auto"], 0)

    def test_no_information_leak_from_run_order(self):
        config = Config(episodes=1, seed=1)
        scenario = make_scenario(config, 0)
        forward = {mode: run_episode(config, scenario, mode) for mode in MODES}
        backward = {
            mode: run_episode(config, scenario, mode) for mode in reversed(MODES)
        }
        self.assertEqual(forward, backward)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            evaluate(Config(episodes=0))


if __name__ == "__main__":
    unittest.main()
