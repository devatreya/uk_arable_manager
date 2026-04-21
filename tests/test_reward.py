"""Tests for reward and P&L mechanics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from sim import FarmSimulator, FarmAction, PlotAction
from config import NUM_PLOTS, REWARD_SCALE, ACRES_PER_PLOT


def _sim(seed=42):
    return FarmSimulator({"seed": seed, "starting_cash": 150_000.0})


def test_cash_crop_generates_positive_pnl():
    """A wheat quarter with medium inputs should produce positive P&L."""
    sim = _sim()
    action = FarmAction("none", [PlotAction(i, "wheat", "medium", "ipm") for i in range(NUM_PLOTS)])
    result = sim.step(action)
    assert result.pnl > 0, f"Expected positive P&L, got {result.pnl}"


def test_fallow_generates_negative_pnl():
    """Fallow on all plots should produce negative P&L (cost only)."""
    sim = _sim()
    action = FarmAction("none", [PlotAction(i, "fallow", "low", "none") for i in range(NUM_PLOTS)])
    result = sim.step(action)
    assert result.pnl < 0, f"Fallow should be net negative, got {result.pnl}"


def test_reward_scale():
    """Reward should be P&L × REWARD_SCALE (approximately, before terminal bonus)."""
    sim = _sim()
    action = FarmAction("none", [PlotAction(i, "barley", "medium", "ipm") for i in range(NUM_PLOTS)])
    result = sim.step(action)
    expected_approx = result.pnl * REWARD_SCALE
    # Terminal bonus only added on finished=True steps
    if not result.finished:
        assert abs(result.reward - expected_approx) < 0.01, \
            f"Reward {result.reward} doesn't match scaled P&L {expected_approx}"


def test_irrigation_purchase_reduces_cash():
    """Buying irrigation should deduct its cost from cash."""
    sim_no_irr = _sim(seed=42)
    sim_with_irr = _sim(seed=42)

    action_no = FarmAction("none",            [PlotAction(i, "wheat", "medium", "ipm") for i in range(NUM_PLOTS)])
    action_irr = FarmAction("buy_irrigation", [PlotAction(i, "wheat", "medium", "ipm") for i in range(NUM_PLOTS)])

    r_no  = sim_no_irr.step(action_no)
    r_irr = sim_with_irr.step(action_irr)

    assert sim_with_irr.state.irrigation_owned is True
    # The cash difference between the two sims should be ~£35,000 (irrigation cost)
    cash_diff = sim_no_irr.state.cash - sim_with_irr.state.cash
    assert 30_000 < cash_diff < 40_000, f"Expected ~35k diff, got {cash_diff}"


def test_no_double_irrigation():
    """Buying irrigation twice should not deduct cost twice."""
    sim = _sim()
    action_buy = FarmAction("buy_irrigation", [PlotAction(i, "wheat", "medium", "ipm") for i in range(NUM_PLOTS)])
    sim.step(action_buy)
    cash_after_first = sim.state.cash

    action_buy_again = FarmAction("buy_irrigation", [PlotAction(i, "wheat", "medium", "ipm") for i in range(NUM_PLOTS)])
    sim.step(action_buy_again)
    # Cash difference between the two steps should not include another irrigation deduction
    pnl_second = sim.state.cash - cash_after_first
    # The second buy_irrigation should be ignored (already owned)
    assert pnl_second > -40_000, "Should not deduct irrigation cost twice"


def test_terminal_score_in_final_step():
    """Final step (quarter 40) should have terminal_score in result."""
    sim = _sim()
    for i in range(40):
        action = FarmAction("none", [PlotAction(j, "wheat", "medium", "ipm") for j in range(NUM_PLOTS)])
        result = sim.step(action)
        if result.finished:
            assert result.terminal_score is not None
            assert 0.0 <= result.terminal_score
            break

    assert result.finished, "Episode should finish at or before quarter 40"


def test_bankruptcy_flag():
    """Spending cash below zero sets ever_bankrupt flag."""
    # Start with very low cash
    sim = FarmSimulator({"seed=": 1, "starting_cash": 1.0, "seed": 99})
    action = FarmAction("none", [PlotAction(i, "cover_crop", "low", "none") for i in range(NUM_PLOTS)])
    for _ in range(5):
        sim.step(action)
    assert sim.state.ever_bankrupt is True


def test_plot_pnl_sum_equals_total():
    """Sum of per-plot P&L should equal total P&L."""
    sim = _sim()
    action = FarmAction("none", [
        PlotAction(0, "wheat",       "medium", "ipm"),
        PlotAction(1, "barley",      "medium", "ipm"),
        PlotAction(2, "oilseed_rape","medium", "ipm"),
        PlotAction(3, "field_beans", "medium", "ipm"),
    ])
    result = sim.step(action)
    assert abs(sum(result.plot_pnl) - result.pnl) < 0.01
