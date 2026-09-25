"""Paired evaluation of two trained area-defense MAPPO checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.experiment import Experiment


def _load(checkpoint: Path) -> Experiment:
    experiment = Experiment.reload_from_file(
        str(checkpoint),
        experiment_patch={
            "evaluation_episodes": 1,
            "evaluation": False,
            "render": False,
            "loggers": [],
            "create_json": False,
        },
    )
    if experiment.task.name.lower() != "area_defense":
        raise ValueError(f"Not an area-defense checkpoint: {checkpoint}")
    return experiment


@torch.no_grad()
def _one(experiment: Experiment, scenario_seed: int) -> Dict[str, float]:
    experiment.test_env.set_seed(scenario_seed)
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = experiment.test_env.rollout(
            max_steps=experiment.max_steps,
            policy=experiment.policy,
            auto_cast_to_device=True,
            break_when_any_done=True,
        )
    group = next(iter(experiment.group_map))
    success = rollout.get(("next", group, "info", "success"))
    breach = rollout.get(("next", group, "info", "breach"))
    reward = rollout.get(("next", group, "reward"))
    scenario_id = rollout.get(("next", group, "info", "scenario_id"))
    return {
        "success": int(success.max().item() > 0),
        "breach": int(breach.max().item() > 0),
        "return": float(reward.mean(dim=-2).sum().item()),
        "steps": int(rollout.batch_size[-1]),
        "scenario_id": int(scenario_id.flatten()[0].item()),
    }


def _paired_ci(deltas: List[int], seed: int, samples: int = 2000) -> List[float]:
    rng = random.Random(seed)
    n = len(deltas)
    values = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)
    )
    return [values[int(0.025 * (samples - 1))], values[int(0.975 * (samples - 1))]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    try:
        output.relative_to(repo_root)
    except ValueError:
        parser.error("--output must be inside this BenchMARL project")
    if output.exists():
        parser.error("--output already exists; choose a new directory")
    for checkpoint in (args.baseline, args.oracle):
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found: {checkpoint}")

    baseline = _load(args.baseline)
    oracle = _load(args.oracle)
    try:
        if baseline.task.config["guidance_mode"] != "none":
            raise ValueError("baseline checkpoint must have guidance_mode=none")
        if oracle.task.config["guidance_mode"] != "oracle":
            raise ValueError("oracle checkpoint must have guidance_mode=oracle")
        if (
            baseline.task.config["opponent_style"]
            != oracle.task.config["opponent_style"]
        ):
            raise ValueError("Both checkpoints must use the same opponent_style")
        rows = []
        for episode in range(args.episodes):
            scenario_seed = args.seed + episode * 1009
            base = _one(baseline, scenario_seed)
            guide = _one(oracle, scenario_seed)
            if base["scenario_id"] != guide["scenario_id"]:
                raise RuntimeError(
                    f"Paired scenario mismatch at episode {episode}: "
                    f"{base['scenario_id']} != {guide['scenario_id']}"
                )
            rows.append(
                {
                    "episode": episode,
                    "scenario_seed": scenario_seed,
                    "scenario_id": base["scenario_id"],
                    "baseline_success": base["success"],
                    "oracle_success": guide["success"],
                    "baseline_breach": base["breach"],
                    "oracle_breach": guide["breach"],
                    "baseline_return": base["return"],
                    "oracle_return": guide["return"],
                    "baseline_steps": base["steps"],
                    "oracle_steps": guide["steps"],
                }
            )
    finally:
        baseline.close()
        oracle.close()

    deltas = [row["oracle_success"] - row["baseline_success"] for row in rows]
    summary = {
        "episodes": args.episodes,
        "seed": args.seed,
        "baseline_checkpoint": str(args.baseline.resolve()),
        "oracle_checkpoint": str(args.oracle.resolve()),
        "baseline_success_rate": sum(row["baseline_success"] for row in rows)
        / args.episodes,
        "oracle_success_rate": sum(row["oracle_success"] for row in rows)
        / args.episodes,
        "paired_success_gain": sum(deltas) / args.episodes,
        "paired_gain_ci95": _paired_ci(deltas, args.seed + 701),
        "baseline_mean_return": sum(row["baseline_return"] for row in rows)
        / args.episodes,
        "oracle_mean_return": sum(row["oracle_return"] for row in rows) / args.episodes,
        "warning": "Different trained policies; oracle has privileged information. Not a human-intervention effect.",
    }
    output.mkdir(parents=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
