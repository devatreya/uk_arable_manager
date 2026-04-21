"""Tests for all three grader variants."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from grader import (
    ScalarFinalScoreGrader,
    BankruptcyAwareGrader,
    StewardshipWeightedGrader,
    grade,
    DEFAULT_GRADER,
)


def _trajectory(
    ending_cash: float = 200_000.0,
    starting_cash: float = 150_000.0,
    ever_bankrupt: bool = False,
    quarters: int = 40,
    mean_soil: float = 0.65,
):
    """Build a minimal trajectory dict for grader testing."""
    plots = [
        {"_organic_matter": mean_soil, "_structure": mean_soil, "_ph": mean_soil, "_nutrient_balance": mean_soil}
        for _ in range(4)
    ]
    return {
        "task_spec": {"starting_cash": starting_cash},
        "final_state": {
            "cash": ending_cash,
            "starting_cash": starting_cash,
            "ever_bankrupt": ever_bankrupt,
            "quarter": quarters,
            "plots": plots,
        },
        "steps": [],
        "terminal_score": None,
    }


class TestScalarGrader:
    def test_perfect_case(self):
        t = _trajectory(ending_cash=300_000.0, ever_bankrupt=False, mean_soil=1.0)
        g = ScalarFinalScoreGrader().grade(t)
        assert g.score > 0.0           # score can exceed 1.0 when cash_ratio > 1
        assert 0.0 < g.normalised <= 1.0  # normalised is always clipped to [0,1]
        assert g.ever_bankrupt is False

    def test_zero_cash_gives_zero(self):
        t = _trajectory(ending_cash=0.0, ever_bankrupt=False, mean_soil=0.8)
        g = ScalarFinalScoreGrader().grade(t)
        assert g.score == 0.0

    def test_bankruptcy_applies_penalty(self):
        t_no_bank = _trajectory(ending_cash=200_000.0, ever_bankrupt=False, mean_soil=0.6)
        t_bankrupt = _trajectory(ending_cash=200_000.0, ever_bankrupt=True, mean_soil=0.6)
        g_no = ScalarFinalScoreGrader().grade(t_no_bank)
        g_yes = ScalarFinalScoreGrader().grade(t_bankrupt)
        assert g_yes.score < g_no.score

    def test_low_soil_reduces_score(self):
        t_high = _trajectory(ending_cash=200_000.0, mean_soil=1.0)
        t_low  = _trajectory(ending_cash=200_000.0, mean_soil=0.41)
        g_high = ScalarFinalScoreGrader().grade(t_high)
        g_low  = ScalarFinalScoreGrader().grade(t_low)
        assert g_high.score > g_low.score


class TestBankruptcyAwareGrader:
    def test_negative_cash_bankrupt_gives_zero(self):
        t = _trajectory(ending_cash=-5_000.0, ever_bankrupt=True, mean_soil=0.5)
        g = BankruptcyAwareGrader().grade(t)
        assert g.score == 0.0

    def test_recovered_bankrupt_gets_partial(self):
        t = _trajectory(ending_cash=50_000.0, ever_bankrupt=True, mean_soil=0.5)
        g = BankruptcyAwareGrader().grade(t)
        assert 0.0 < g.score < 0.5

    def test_incomplete_episode_penalised(self):
        t_full = _trajectory(quarters=40)
        t_half = _trajectory(quarters=20)
        g_full = BankruptcyAwareGrader().grade(t_full)
        g_half = BankruptcyAwareGrader().grade(t_half)
        assert g_half.score < g_full.score


class TestStewardshipGrader:
    def test_high_soil_score_boosted(self):
        t_high_soil = _trajectory(ending_cash=180_000.0, mean_soil=1.15)
        t_low_soil  = _trajectory(ending_cash=180_000.0, mean_soil=0.42)
        g_high = StewardshipWeightedGrader().grade(t_high_soil)
        g_low  = StewardshipWeightedGrader().grade(t_low_soil)
        assert g_high.score > g_low.score

    def test_weights_sum_correctly(self):
        # When both components are 1.0, score should be <= 1.0
        t = _trajectory(ending_cash=300_000.0, mean_soil=1.2, ever_bankrupt=False)
        g = StewardshipWeightedGrader().grade(t)
        assert g.score <= 1.0


def test_grade_dispatch():
    """grade() function should dispatch to the right grader."""
    from grader import GRADERS
    t = _trajectory()
    for name in GRADERS:
        result = grade(t, name)
        assert hasattr(result, "score")
        assert isinstance(result.score, float)


def test_invalid_grader_raises():
    t = _trajectory()
    with pytest.raises(ValueError):
        grade(t, "nonexistent_grader")


def test_end_to_end_graded_episode():
    """Run a real episode and grade it — integration test."""
    from sim import FarmSimulator, FarmAction, PlotAction
    from config import NUM_PLOTS, TOTAL_QUARTERS
    from trajectory_logger import TrajectoryLogger

    spec = {"seed": 42, "starting_cash": 150_000.0, "task_id": "grader_test"}
    sim = FarmSimulator(spec)
    logger = TrajectoryLogger(spec)

    for q in range(TOTAL_QUARTERS):
        action = FarmAction("none", [PlotAction(i, "wheat", "medium", "ipm") for i in range(NUM_PLOTS)])
        result = sim.step(action)
        logger.record_step(
            quarter=q, action={"capital_action": "none", "plots": []},
            reward=result.reward, pnl=result.pnl, observation="",
            weather=result.weather.to_dict(), plot_pnl=result.plot_pnl,
            bankrupt=result.bankrupt, pest_pressure=result.pest_pressure,
        )
        if result.finished:
            break

    traj = logger.finalise(sim.state.to_dict(), sim.terminal_score())
    g = grade(traj.to_dict())

    assert isinstance(g.score, float)
    assert g.quarters_completed == TOTAL_QUARTERS
    assert g.ending_cash == pytest.approx(sim.state.cash, rel=0.01)
