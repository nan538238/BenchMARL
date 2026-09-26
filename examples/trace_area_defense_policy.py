"""Print one deterministic area-defense rollout without changing training state."""

import argparse
import json
from pathlib import Path

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from evaluate_area_defense import _load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--interval", type=int, default=20)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    experiment = _load(args.checkpoint)
    try:
        if experiment.task.name.lower() not in (
            "area_defense_v2", "area_defense_v3", "area_defense_v4", "area_defense_v5"
        ):
            parser.error("this trace expects a v2-v5 checkpoint")
        experiment.test_env.set_seed(args.seed)
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            rollout = experiment.test_env.rollout(
                max_steps=experiment.max_steps,
                policy=experiment.policy,
                auto_cast_to_device=True,
                break_when_any_done=True,
            )

        group = next(iter(experiment.group_map))
        steps = rollout.batch_size[-1]
        observation = rollout.get((group, "observation")).reshape(steps, 3, 24)
        action = rollout.get((group, "action")).reshape(steps, 3, -1)
        capture_events = rollout.get(("next", group, "info", "captures"))
        capture_events = capture_events.reshape(steps, 3, -1)[:, 0, 0]
        success = rollout.get(("next", group, "info", "success")).max().item()
        breach = rollout.get(("next", group, "info", "breach")).max().item()
        scenario_id = rollout.get(("next", group, "info", "scenario_id"))

        samples = []
        for step in sorted(set(range(0, steps, args.interval)) | {steps - 1}):
            blue = []
            for agent in range(3):
                obs = observation[step, agent]
                blue.append(
                    {
                        "agent": agent,
                        "pos": [round(value, 3) for value in obs[:2].tolist()],
                        "action": [
                            round(value, 3) for value in action[step, agent].tolist()
                        ],
                        "visible_red": [int(obs[index].item()) for index in (10, 13, 16)],
                        "available": int(obs[21 + agent].item()),
                    }
                )
            samples.append({"step": step, "blue": blue})

        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "steps": steps,
                    "scenario_id": int(scenario_id.flatten()[0].item()),
                    "success": bool(success),
                    "breach": bool(breach),
                    "captures": int(capture_events.sum().item()),
                    "capture_steps": [
                        step + 1
                        for step in range(steps)
                        if capture_events[step].item() > 0
                    ],
                    "samples": samples,
                },
                indent=2,
            )
        )
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
