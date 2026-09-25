"""Paired, scripted 3v3 area-defense test of privileged tactical advice.

This is a diagnostic simulator, not a VMAS task or a trained MAPPO policy.
All modes use the same low-level defender controller and pre-sampled attacker
trajectories. The only intervention is a high-level lane-defense instruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


LANES = (-0.6, 0.0, 0.6)
MODES = ("auto", "oracle_all", "oracle_budget")
FAMILIES = {
    "train": ("concentrated", "spread", "feint"),
    "ood": ("late_switch", "staggered_concentrated", "mixed"),
}


@dataclass(frozen=True)
class Config:
    episodes: int = 200
    seed: int = 100
    split: str = "ood"
    budget: int = 2
    horizon: int = 110
    decision_interval: int = 10
    sensor_range: float = 0.72
    defender_speed: float = 0.030
    capture_radius: float = 0.075
    capture_cooldown: int = 15
    bootstrap_samples: int = 2000

    def validate(self) -> None:
        if self.episodes < 1 or self.horizon < 1 or self.decision_interval < 1:
            raise ValueError("episodes, horizon and decision_interval must be positive")
        if self.budget < 0 or self.bootstrap_samples < 1:
            raise ValueError(
                "budget must be nonnegative and bootstrap_samples positive"
            )
        if self.split not in FAMILIES:
            raise ValueError("split must be 'train' or 'ood'")
        if self.sensor_range <= 0 or self.defender_speed <= 0:
            raise ValueError("sensor_range and defender_speed must be positive")
        if self.capture_radius <= 0 or self.capture_cooldown < 0:
            raise ValueError("capture_radius must be positive and cooldown nonnegative")


@dataclass(frozen=True)
class Scenario:
    episode: int
    scenario_seed: int
    family: str
    # Each trajectory contains an (x, y) point for every simulation step.
    trajectories: Tuple[Tuple[Tuple[float, float], ...], ...]


@dataclass
class Defender:
    x: float
    y: float
    cooldown: int = 0


@dataclass(frozen=True)
class Result:
    episode: int
    scenario_seed: int
    family: str
    mode: str
    success: int
    breached: int
    captures: int
    steps: int
    calls: int
    episode_return: int


def _nearest_lane(y: float) -> int:
    return min(range(3), key=lambda i: abs(LANES[i] - y))


def _move_towards(
    x: float, y: float, tx: float, ty: float, speed: float
) -> Tuple[float, float]:
    dx, dy = tx - x, ty - y
    distance = math.hypot(dx, dy)
    if distance <= speed or distance == 0:
        return tx, ty
    return x + speed * dx / distance, y + speed * dy / distance


def make_scenario(config: Config, episode: int) -> Scenario:
    scenario_seed = config.seed + episode * 1009
    rng = random.Random(scenario_seed)
    family = FAMILIES[config.split][episode % len(FAMILIES[config.split])]
    focus = rng.randrange(3)
    decoy = (focus + rng.choice((1, 2))) % 3
    if family in ("concentrated", "staggered_concentrated"):
        initial = final = (focus, focus, focus)
    elif family == "spread":
        initial = final = (0, 1, 2)
    elif family in ("feint", "late_switch"):
        initial = (decoy, decoy, decoy)
        final = (focus, focus, focus)
    else:  # mixed: one decoy and two attackers switch to a shared target
        initial = (decoy, decoy, focus)
        final = (focus, focus, decoy)

    switch_step = 28 if family == "feint" else 39
    all_paths: List[Tuple[Tuple[float, float], ...]] = []
    for attacker in range(3):
        delay = attacker * 8 if family == "staggered_concentrated" else attacker * 2
        speed = rng.uniform(0.034, 0.040)
        start_x = 1.05 + rng.uniform(-0.06, 0.06)
        initial_y = LANES[initial[attacker]] + rng.uniform(-0.035, 0.035)
        final_y = LANES[final[attacker]] + rng.uniform(-0.035, 0.035)
        path: List[Tuple[float, float]] = []
        for step in range(config.horizon):
            x = start_x - speed * max(0, step - delay)
            if initial[attacker] == final[attacker]:
                y = initial_y
            else:
                fraction = min(1.0, max(0.0, (step - switch_step) / 14.0))
                y = initial_y + fraction * (final_y - initial_y)
            path.append((x, y))
        all_paths.append(tuple(path))
    return Scenario(episode, scenario_seed, family, tuple(all_paths))


def _visible_attackers(
    defenders: Sequence[Defender],
    scenario: Scenario,
    alive: Sequence[bool],
    step: int,
    sensor_range: float,
) -> List[Tuple[float, float]]:
    visible = []
    for index, path in enumerate(scenario.trajectories):
        if not alive[index]:
            continue
        x, y = path[step]
        if any(
            math.hypot(x - blue.x, y - blue.y) <= sensor_range for blue in defenders
        ):
            visible.append((x, y))
    return visible


def _auto_intent(visible: Sequence[Tuple[float, float]]) -> Optional[int]:
    counts = [0, 0, 0]
    for _, y in visible:
        counts[_nearest_lane(y)] += 1
    peak = max(counts)
    return counts.index(peak) if peak >= 2 else None


def _oracle_intent(
    scenario: Scenario, alive: Sequence[bool], step: int
) -> Optional[int]:
    """Privileged commander sees future attacker lanes, not future outcomes."""
    counts = [0, 0, 0]
    lookahead = min(len(scenario.trajectories[0]) - 1, step + 25)
    for index, path in enumerate(scenario.trajectories):
        if alive[index]:
            counts[_nearest_lane(path[lookahead][1])] += 1
    peak = max(counts)
    return counts.index(peak) if peak >= 2 else None


def run_episode(config: Config, scenario: Scenario, mode: str) -> Result:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    defenders = [Defender(-0.58, lane) for lane in LANES]
    alive = [True, True, True]
    calls = 0
    captures = 0
    current_intent: Optional[int] = None
    breached = False
    steps = config.horizon

    for step in range(config.horizon):
        if step % config.decision_interval == 0:
            visible = _visible_attackers(
                defenders, scenario, alive, step, config.sensor_range
            )
            auto = _auto_intent(visible)
            oracle = _oracle_intent(scenario, alive, step)
            if mode == "oracle_all":
                current_intent = oracle
                calls += 1
            elif mode == "oracle_budget" and oracle != auto and calls < config.budget:
                current_intent = oracle
                calls += 1
            else:
                current_intent = auto

        for index, blue in enumerate(defenders):
            if blue.cooldown > 0:
                blue.cooldown -= 1
            if current_intent is None:
                target_y = LANES[index]
            else:
                target_y = max(
                    -0.75, min(0.75, LANES[current_intent] + (index - 1) * 0.11)
                )
            target_x = -0.48
            # Both conditions use the same local pursuit reflex.
            nearby = [
                scenario.trajectories[j][step]
                for j in range(3)
                if alive[j]
                and math.hypot(
                    scenario.trajectories[j][step][0] - blue.x,
                    scenario.trajectories[j][step][1] - blue.y,
                )
                < 0.42
            ]
            if nearby:
                target_x, target_y = min(
                    nearby,
                    key=lambda point: math.hypot(point[0] - blue.x, point[1] - blue.y),
                )
            blue.x, blue.y = _move_towards(
                blue.x, blue.y, target_x, target_y, config.defender_speed
            )

        for index, path in enumerate(scenario.trajectories):
            if not alive[index]:
                continue
            red_x, red_y = path[step]
            for blue in defenders:
                if (
                    blue.cooldown == 0
                    and math.hypot(red_x - blue.x, red_y - blue.y)
                    <= config.capture_radius
                ):
                    alive[index] = False
                    blue.cooldown = config.capture_cooldown
                    captures += 1
                    break
            if alive[index] and red_x <= -0.95:
                breached = True
                steps = step + 1
                break
        if breached or not any(alive):
            steps = step + 1
            break

    success = int(not breached and not any(alive))
    return Result(
        scenario.episode,
        scenario.scenario_seed,
        scenario.family,
        mode,
        success,
        int(breached),
        captures,
        steps,
        calls,
        1 if success else -1,
    )


def paired_ci(deltas: Sequence[int], seed: int, samples: int) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    boot = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)
    )
    return boot[int(0.025 * (samples - 1))], boot[int(0.975 * (samples - 1))]


def evaluate(config: Config) -> Tuple[List[Result], Dict[str, object]]:
    config.validate()
    results: List[Result] = []
    for episode in range(config.episodes):
        scenario = make_scenario(config, episode)
        results.extend(run_episode(config, scenario, mode) for mode in MODES)

    by_mode = {mode: [row for row in results if row.mode == mode] for mode in MODES}
    baseline = by_mode["auto"]
    summaries: Dict[str, object] = {}
    for mode in MODES:
        rows = by_mode[mode]
        deltas = [row.success - ref.success for row, ref in zip(rows, baseline)]
        lo, hi = paired_ci(deltas, config.seed + 97, config.bootstrap_samples)
        by_family = {}
        for family in FAMILIES[config.split]:
            family_pairs = [
                (row, ref) for row, ref in zip(rows, baseline) if row.family == family
            ]
            if family_pairs:
                by_family[family] = {
                    "episodes": len(family_pairs),
                    "success_rate": sum(row.success for row, _ in family_pairs)
                    / len(family_pairs),
                    "paired_gain_vs_auto": sum(
                        row.success - ref.success for row, ref in family_pairs
                    )
                    / len(family_pairs),
                }
        summaries[mode] = {
            "success_rate": sum(row.success for row in rows) / config.episodes,
            "mean_calls": sum(row.calls for row in rows) / config.episodes,
            "paired_gain_vs_auto": sum(deltas) / config.episodes,
            "paired_gain_ci95": [lo, hi],
            "wins_from_intervention": sum(delta == 1 for delta in deltas),
            "losses_from_intervention": sum(delta == -1 for delta in deltas),
            "by_family": by_family,
        }
    return results, {
        "config": asdict(config),
        "modes": summaries,
        "interpretation": "scripted diagnostic only; not VMAS, MAPPO, or a human study",
    }


def save_results(
    output: Path, results: Sequence[Result], summary: Dict[str, object]
) -> None:
    for name in ("episodes.csv", "summary.json"):
        if (output / name).exists():
            raise FileExistsError(
                f"{output / name} already exists; choose a new --output directory"
            )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Result.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--split", choices=tuple(FAMILIES), default="ood")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    try:
        output.relative_to(repo_root)
    except ValueError:
        parser.error("--output must be inside this BenchMARL project")
    config = Config(
        episodes=args.episodes, seed=args.seed, split=args.split, budget=args.budget
    )
    results, summary = evaluate(config)
    try:
        save_results(output, results, summary)
    except FileExistsError as error:
        parser.error(str(error))
    for mode, metrics in summary["modes"].items():
        print(
            f"{mode}: success={metrics['success_rate']:.3f}, "
            f"calls={metrics['mean_calls']:.2f}, paired_gain={metrics['paired_gain_vs_auto']:+.3f}, "
            f"CI95={metrics['paired_gain_ci95']}"
        )
    print(f"Wrote {output / 'episodes.csv'} and {output / 'summary.json'}")


if __name__ == "__main__":
    main()
