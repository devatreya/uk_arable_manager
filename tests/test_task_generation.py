"""Tests for task generation and distribution."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_build_tasks_generates_correct_counts():
    """build_tasks should produce exactly the right number of tasks per split."""
    from scripts.build_tasks import build_tasks_for_split
    from config import SPLIT_SCALES

    scales = SPLIT_SCALES["dry_run"]
    for split, n in scales.items():
        tasks = build_tasks_for_split(split, n, base_seed=9999)
        assert len(tasks) == n, f"{split}: expected {n} tasks, got {len(tasks)}"


def test_tasks_have_required_fields():
    """Every task must have the required keys."""
    from scripts.build_tasks import build_tasks_for_split

    required = {
        "task_id", "seed", "split", "scenario_type",
        "starting_cash", "initial_weather_regime",
        "dry_bias", "price_volatility",
        "fertiliser_cost_multiplier", "irrigation_cost_multiplier",
        "initial_soil_by_plot", "initial_crop_by_plot",
    }
    tasks = build_tasks_for_split("train", 10, base_seed=42)
    for task in tasks:
        missing = required - set(task.keys())
        assert not missing, f"Task missing keys: {missing}"


def test_tasks_are_deterministic():
    """Same base_seed must produce identical tasks."""
    from scripts.build_tasks import build_tasks_for_split
    t1 = build_tasks_for_split("train", 8, base_seed=123)
    t2 = build_tasks_for_split("train", 8, base_seed=123)
    assert t1 == t2


def test_scenario_mix_coverage():
    """All 4 scenario types should appear in a large enough batch."""
    from scripts.build_tasks import build_tasks_for_split
    tasks = build_tasks_for_split("train", 64, base_seed=1)
    scenario_types = {t["scenario_type"] for t in tasks}
    expected = {"standard", "drought_stressed", "input_cost_shock", "recovery"}
    assert expected == scenario_types, f"Missing scenarios: {expected - scenario_types}"


def test_tasks_have_valid_initial_crops():
    """initial_crop_by_plot must only contain valid crops."""
    from scripts.build_tasks import build_tasks_for_split
    from config import CROPS
    tasks = build_tasks_for_split("validation", 16, base_seed=42)
    for task in tasks:
        for crop in task["initial_crop_by_plot"]:
            assert crop in CROPS, f"Invalid crop: {crop}"


def test_soil_values_in_range():
    """Initial soil values must be within [0.20, 1.30]."""
    from scripts.build_tasks import build_tasks_for_split
    tasks = build_tasks_for_split("test", 16, base_seed=42)
    for task in tasks:
        for val in task["initial_soil_by_plot"]:
            assert 0.20 <= val <= 1.30, f"Soil value out of range: {val}"


def test_task_runs_full_episode():
    """A generated task must survive a 40-quarter episode with the simulator."""
    from scripts.build_tasks import build_tasks_for_split
    from sim import FarmSimulator, FarmAction, PlotAction
    from config import NUM_PLOTS, TOTAL_QUARTERS

    tasks = build_tasks_for_split("train", 3, base_seed=42)
    for task in tasks:
        sim = FarmSimulator(task)
        for _ in range(TOTAL_QUARTERS):
            action = FarmAction("none", [PlotAction(i, "wheat", "medium", "ipm") for i in range(NUM_PLOTS)])
            result = sim.step(action)
            if result.finished:
                break
        assert sim.state.quarter == TOTAL_QUARTERS or result.finished
