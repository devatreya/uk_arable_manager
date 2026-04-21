"""
Generate train/validation/test task JSON files for uk_arable_manager.

Usage:
  python scripts/build_tasks.py                    # dry_run scale (24/8/8)
  python scripts/build_tasks.py --scale medium     # 64/16/16
  python scripts/build_tasks.py --scale full       # 128/32/32

Each task is one full 40-quarter episode specification.
Scenario mix: 50% standard, 25% drought_stressed, 15% input_cost_shock, 10% recovery.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CROPS,
    DATA_PROCESSED,
    DEFAULT_SCALE,
    SCENARIO_MIX,
    SPLIT_SCALES,
    STARTING_CASH_DEFAULT,
    WEATHER_REGIMES,
)


def _scenario_params(scenario: str, seed_rng: random.Random) -> Dict[str, Any]:
    """Return scenario-specific task parameters."""
    if scenario == "standard":
        return {
            "starting_cash": seed_rng.uniform(120_000, 200_000),
            "initial_weather_regime": seed_rng.choice(["normal", "normal", "wet"]),
            "dry_bias": 0.0,
            "price_volatility": seed_rng.uniform(0.06, 0.10),
            "fertiliser_cost_multiplier": seed_rng.uniform(0.95, 1.10),
            "irrigation_cost_multiplier": 1.0,
        }
    elif scenario == "drought_stressed":
        return {
            "starting_cash": seed_rng.uniform(130_000, 190_000),
            "initial_weather_regime": "dry",
            "dry_bias": seed_rng.uniform(0.3, 0.7),
            "price_volatility": seed_rng.uniform(0.07, 0.12),
            "fertiliser_cost_multiplier": seed_rng.uniform(1.00, 1.15),
            "irrigation_cost_multiplier": seed_rng.uniform(0.85, 1.05),
        }
    elif scenario == "input_cost_shock":
        return {
            "starting_cash": seed_rng.uniform(120_000, 180_000),
            "initial_weather_regime": seed_rng.choice(["normal", "normal", "dry"]),
            "dry_bias": seed_rng.uniform(0.0, 0.2),
            "price_volatility": seed_rng.uniform(0.10, 0.18),
            "fertiliser_cost_multiplier": seed_rng.uniform(1.30, 1.80),
            "irrigation_cost_multiplier": seed_rng.uniform(1.10, 1.40),
        }
    elif scenario == "recovery":
        # Farm starts in poor shape: low soil, low cash
        return {
            "starting_cash": seed_rng.uniform(80_000, 130_000),
            "initial_weather_regime": seed_rng.choice(["normal", "dry"]),
            "dry_bias": seed_rng.uniform(0.0, 0.3),
            "price_volatility": seed_rng.uniform(0.06, 0.10),
            "fertiliser_cost_multiplier": seed_rng.uniform(0.90, 1.05),
            "irrigation_cost_multiplier": 1.0,
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def _initial_soil(scenario: str, seed_rng: random.Random) -> List[float]:
    if scenario == "recovery":
        return [seed_rng.uniform(0.25, 0.38) for _ in range(4)]
    elif scenario == "drought_stressed":
        return [seed_rng.uniform(0.40, 0.55) for _ in range(4)]
    else:
        return [seed_rng.uniform(0.48, 0.68) for _ in range(4)]


def _initial_crops(seed_rng: random.Random) -> List[str]:
    varied = ["wheat", "barley", "oilseed_rape", "field_beans", "fallow"]
    return [seed_rng.choice(varied) for _ in range(4)]


def build_tasks_for_split(
    split: str,
    n: int,
    base_seed: int,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    global_rng = random.Random(base_seed)

    # Compute scenario distribution
    scenario_pool: List[str] = []
    for scenario, frac in SCENARIO_MIX.items():
        count = max(1, round(n * frac))
        scenario_pool.extend([scenario] * count)
    # Trim / pad to exactly n
    while len(scenario_pool) < n:
        scenario_pool.append("standard")
    scenario_pool = scenario_pool[:n]
    global_rng.shuffle(scenario_pool)

    for i, scenario in enumerate(scenario_pool):
        task_seed = base_seed * 10_000 + i
        seed_rng = random.Random(task_seed)
        params = _scenario_params(scenario, seed_rng)

        task: Dict[str, Any] = {
            "task_id": f"{split}_{i:04d}",
            "seed": task_seed,
            "split": split,
            "scenario_type": scenario,
            "starting_cash": round(params["starting_cash"], 2),
            "initial_weather_regime": params["initial_weather_regime"],
            "dry_bias": round(params["dry_bias"], 4),
            "price_volatility": round(params["price_volatility"], 4),
            "fertiliser_cost_multiplier": round(params["fertiliser_cost_multiplier"], 4),
            "irrigation_cost_multiplier": round(params["irrigation_cost_multiplier"], 4),
            "initial_soil_by_plot": [round(v, 4) for v in _initial_soil(scenario, seed_rng)],
            "initial_crop_by_plot": _initial_crops(seed_rng),
        }
        tasks.append(task)

    return tasks


def main(scale: str = DEFAULT_SCALE) -> None:
    scales = SPLIT_SCALES.get(scale)
    if scales is None:
        raise ValueError(f"Unknown scale '{scale}'. Options: {list(SPLIT_SCALES)}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Use fixed base seeds per split for reproducibility
    seeds = {"train": 1001, "validation": 2001, "test": 3001}

    for split, n in scales.items():
        tasks = build_tasks_for_split(split, n, base_seed=seeds[split])
        out_path = DATA_PROCESSED / f"scenario_tasks_{split}.json"
        with open(out_path, "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"  Wrote {len(tasks)} {split} tasks → {out_path}")

    # Print summary
    print(f"\nTask generation complete (scale={scale})")
    for split, n in scales.items():
        print(f"  {split}: {n} tasks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        default=DEFAULT_SCALE,
        choices=list(SPLIT_SCALES.keys()),
        help="Task distribution scale",
    )
    args = parser.parse_args()
    main(scale=args.scale)
