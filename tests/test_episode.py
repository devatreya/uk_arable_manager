"""Tests for full episode mechanics and ORS environment compliance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from sim import FarmSimulator, FarmAction, PlotAction
from config import NUM_PLOTS, TOTAL_QUARTERS


def _full_action():
    return FarmAction("none", [PlotAction(i, "wheat", "medium", "ipm") for i in range(NUM_PLOTS)])


def test_episode_runs_exactly_40_quarters():
    """A non-bankrupt episode should run for exactly 40 quarters."""
    sim = FarmSimulator({"seed": 42, "starting_cash": 150_000.0})
    steps = 0
    for _ in range(TOTAL_QUARTERS + 5):
        result = sim.step(_full_action())
        steps += 1
        if result.finished:
            break
    assert steps == TOTAL_QUARTERS
    assert sim.state.quarter == TOTAL_QUARTERS


def test_cannot_step_after_finished():
    """Stepping after episode finishes should raise ValueError."""
    sim = FarmSimulator({"seed": 42, "starting_cash": 150_000.0})
    for _ in range(TOTAL_QUARTERS):
        result = sim.step(_full_action())
    assert result.finished
    with pytest.raises(ValueError):
        sim.step(_full_action())


def test_determinism_same_seed():
    """Two sims with the same seed must produce identical trajectories."""
    spec = {"seed": 777, "starting_cash": 150_000.0}
    sim1 = FarmSimulator(spec)
    sim2 = FarmSimulator(spec)

    for _ in range(10):
        a = _full_action()
        r1 = sim1.step(a)
        r2 = sim2.step(a)
        assert r1.pnl == r2.pnl
        assert r1.weather.regime == r2.weather.regime


def test_different_seeds_differ():
    """Different seeds must produce different outcomes."""
    sim1 = FarmSimulator({"seed": 1, "starting_cash": 150_000.0})
    sim2 = FarmSimulator({"seed": 2, "starting_cash": 150_000.0})

    pnls1, pnls2 = [], []
    for _ in range(5):
        pnls1.append(sim1.step(_full_action()).pnl)
        pnls2.append(sim2.step(_full_action()).pnl)

    assert pnls1 != pnls2


def test_state_serialisation_roundtrip():
    """FarmState.to_dict / from_dict round-trips correctly."""
    from sim import FarmState
    sim = FarmSimulator({"seed": 5, "starting_cash": 150_000.0})
    for _ in range(5):
        sim.step(_full_action())

    d = sim.state.to_dict()
    restored = FarmState.from_dict(d)

    assert restored.quarter == sim.state.quarter
    assert abs(restored.cash - sim.state.cash) < 0.01
    assert restored.irrigation_owned == sim.state.irrigation_owned
    for i in range(NUM_PLOTS):
        assert abs(restored.plots[i].soil_health - sim.state.plots[i].soil_health) < 1e-3


def test_ors_environment_tools_return_tool_output():
    """ORS environment tools must return ToolOutput instances."""
    from ors import ToolOutput
    from env import UKArableManager, ReadSoilInput, ReadWeatherInput

    task = {
        "task_id": "test_001",
        "seed": 42,
        "starting_cash": 150_000.0,
        "initial_weather_regime": "normal",
        "dry_bias": 0.0,
        "price_volatility": 0.08,
        "fertiliser_cost_multiplier": 1.0,
        "irrigation_cost_multiplier": 1.0,
        "initial_soil_by_plot": [0.55, 0.55, 0.55, 0.55],
        "initial_crop_by_plot": ["fallow", "fallow", "fallow", "fallow"],
    }

    env = UKArableManager(task_spec=task)
    env.setup()

    result = env.read_farm_state()
    assert isinstance(result, ToolOutput)
    assert result.finished is False
    assert result.reward is None
    assert len(result.blocks) > 0

    result2 = env.read_soil_report(ReadSoilInput(plots=[0, 1, 2, 3]))
    assert isinstance(result2, ToolOutput)

    result3 = env.read_weather_history(ReadWeatherInput(lookback_quarters=4))
    assert isinstance(result3, ToolOutput)

    result4 = env.read_price_board()
    assert isinstance(result4, ToolOutput)


def test_ors_commit_plan_advances_quarter():
    """commit_plan should advance the quarter counter."""
    from env import UKArableManager, CommitPlanInput, PlotPlanInput

    task = {"seed": 42, "starting_cash": 150_000.0}
    env = UKArableManager(task_spec=task)
    env.setup()
    assert env.sim.state.quarter == 0

    plan = CommitPlanInput(
        capital_action="none",
        plot_1=PlotPlanInput(crop="wheat", fertiliser="medium", pest_control="ipm"),
        plot_2=PlotPlanInput(crop="barley", fertiliser="medium", pest_control="ipm"),
        plot_3=PlotPlanInput(crop="field_beans", fertiliser="low", pest_control="none"),
        plot_4=PlotPlanInput(crop="cover_crop", fertiliser="low", pest_control="none"),
    )
    result = env.commit_plan(plan)
    assert env.sim.state.quarter == 1
    assert result.reward is not None


def test_ors_list_splits():
    """list_splits must return train/validation/test."""
    from env import UKArableManager
    splits = UKArableManager.list_splits()
    names = {s.name if hasattr(s, "name") else s for s in splits}
    assert "train" in names
    assert "validation" in names
    assert "test" in names
