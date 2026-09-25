"""Measure whether a zero-action policy can win the revised VMAS task."""

import argparse

from benchmarl.environments.vmas.common import VmasTask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    for style in ("concentrated", "spread", "feint", "late_switch"):
        task = VmasTask.AREA_DEFENSE_V2.get_task()
        task.config["guidance_mode"] = "none"
        task.config["opponent_style"] = style
        env = task.get_env_fun(
            num_envs=1, continuous_actions=True, seed=args.seed, device="cpu"
        )()

        def stay_still(td):
            return td.update(env.full_action_spec.zero())

        wins = 0
        returns = []
        try:
            for episode in range(args.episodes):
                env.set_seed(args.seed + episode * 1009)
                rollout = env.rollout(
                    max_steps=task.config["max_steps"],
                    policy=stay_still,
                    break_when_any_done=True,
                )
                wins += int(
                    rollout.get(("next", "blue", "info", "success")).any().item()
                )
                returns.append(
                    rollout.get(("next", "blue", "reward")).mean(dim=-2).sum().item()
                )
            print(
                f"{style}: success={wins}/{args.episodes}, "
                f"mean_return={sum(returns) / args.episodes:.3f}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
