"""Test a hand-written oracle lane-assignment controller on area-defense v2.

This controller reads the same four-way oracle hint as the oracle policy. It
steers all defenders to the indicated final lane, or assigns one per lane when
red attackers are spread. It is a feasibility diagnostic, not a learned agent.
"""

import argparse

import torch


def oracle_lane_rule(td):
    observation = td.get(("blue", "observation"))
    if observation.shape[-2:] != (3, 24):
        raise ValueError(
            f"Expected three blue agents with 24 features: {observation.shape}"
        )
    lanes = observation.new_tensor([-0.6, 0.0, 0.6])
    intent = observation[..., 17:21].argmax(dim=-1)
    spread_targets = lanes.expand_as(intent)
    focused_targets = lanes[intent.clamp(max=2)]
    target_y = torch.where(intent == 3, spread_targets, focused_targets)
    target_x = torch.full_like(target_y, -0.62)
    target = torch.stack((target_x, target_y), dim=-1)
    position = observation[..., :2]
    velocity = observation[..., 2:4]
    action = (5.0 * (target - position) - 2.0 * velocity).clamp(-1.0, 1.0)
    return td.set(("blue", "action"), action)


def main() -> None:
    from benchmarl.environments.vmas.common import VmasTask

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    for style in ("concentrated", "spread", "feint", "late_switch"):
        task = VmasTask.AREA_DEFENSE_V2.get_task()
        task.config["guidance_mode"] = "oracle"
        task.config["opponent_style"] = style
        env = task.get_env_fun(
            num_envs=1, continuous_actions=True, seed=args.seed, device="cpu"
        )()
        wins = 0
        captures_total = 0
        try:
            for episode in range(args.episodes):
                env.set_seed(args.seed + episode * 1009)
                rollout = env.rollout(
                    max_steps=task.config["max_steps"],
                    policy=oracle_lane_rule,
                    break_when_any_done=True,
                )
                wins += int(
                    rollout.get(("next", "blue", "info", "success")).any().item()
                )
                capture_events = rollout.get(("next", "blue", "info", "captures"))
                captures_total += int(capture_events.reshape(-1, 3)[:, 0].sum().item())
            print(
                f"{style}: success={wins}/{args.episodes}, "
                f"mean_captures={captures_total / args.episodes:.2f}/3"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
