"""Tests for soil health mechanics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from sim import FarmSimulator, FarmAction, PlotAction
from config import NUM_PLOTS, SOIL_MIN, SOIL_MAX, SOIL_DELTA


def _make_sim(seed=42, **kwargs):
    spec = {"seed": seed, "starting_cash": 150_000.0, **kwargs}
    return FarmSimulator(spec)


def test_soil_bounds_respected():
    """Soil sub-components must stay within [SOIL_MIN, SOIL_MAX]."""
    sim = _make_sim()
    # Run 40 quarters with greedy extraction
    for _ in range(40):
        action = FarmAction(
            capital_action="none",
            plots=[PlotAction(i, "wheat", "high", "spray") for i in range(NUM_PLOTS)],
        )
        sim.step(action)
        for p in sim.state.plots:
            assert SOIL_MIN <= p._organic_matter <= SOIL_MAX
            assert SOIL_MIN <= p._structure <= SOIL_MAX
            assert SOIL_MIN <= p._ph <= SOIL_MAX
            assert SOIL_MIN <= p._nutrient_balance <= SOIL_MAX


def test_restorative_crops_improve_soil():
    """Cover crop must improve soil health over 8 quarters."""
    sim = _make_sim()
    initial = [p.soil_health for p in sim.state.plots]

    for _ in range(8):
        action = FarmAction(
            capital_action="none",
            plots=[PlotAction(i, "cover_crop", "low", "none") for i in range(NUM_PLOTS)],
        )
        sim.step(action)

    final = [p.soil_health for p in sim.state.plots]
    for i, (ini, fin) in enumerate(zip(initial, final)):
        assert fin > ini, f"Plot {i}: soil should improve with cover_crop (was {ini:.3f}, now {fin:.3f})"


def test_extractive_crops_degrade_soil():
    """Repeated wheat should degrade soil health."""
    sim = _make_sim()
    initial = [p.soil_health for p in sim.state.plots]

    for _ in range(12):
        action = FarmAction(
            capital_action="none",
            plots=[PlotAction(i, "wheat", "high", "spray") for i in range(NUM_PLOTS)],
        )
        sim.step(action)

    final = [p.soil_health for p in sim.state.plots]
    for i, (ini, fin) in enumerate(zip(initial, final)):
        assert fin < ini, f"Plot {i}: soil should degrade with repeated wheat (was {ini:.3f}, now {fin:.3f})"


def test_soil_health_aggregate_weights():
    """soil_health = weighted sum of sub-components."""
    from sim import PlotState
    from config import SOIL_WEIGHT_OM, SOIL_WEIGHT_STRUCTURE, SOIL_WEIGHT_PH, SOIL_WEIGHT_NUTRIENT

    p = PlotState(plot_id=0)
    p._organic_matter = 0.70
    p._structure = 0.50
    p._ph = 0.60
    p._nutrient_balance = 0.45

    expected = (
        SOIL_WEIGHT_OM * 0.70
        + SOIL_WEIGHT_STRUCTURE * 0.50
        + SOIL_WEIGHT_PH * 0.60
        + SOIL_WEIGHT_NUTRIENT * 0.45
    )
    assert abs(p.soil_health - expected) < 1e-9


def test_repeat_penalty_applied():
    """Planting the same cash crop twice in a row should incur extra soil penalty."""
    sim1 = _make_sim(seed=1)
    sim2 = _make_sim(seed=1)

    # Both sims start identical; sim1 plants wheat twice (gets repeat penalty), sim2 alternates
    action_wheat = FarmAction("none", [PlotAction(0, "wheat", "medium", "ipm")] + [PlotAction(i, "fallow", "low", "none") for i in range(1, NUM_PLOTS)])

    sim1.step(action_wheat)
    soil_after_first_wheat_sim1 = sim1.state.plots[0].soil_health

    sim2.step(action_wheat)
    soil_after_first_wheat_sim2 = sim2.state.plots[0].soil_health

    # After first step both should be same
    assert abs(soil_after_first_wheat_sim1 - soil_after_first_wheat_sim2) < 0.05

    # Second step: sim1 plants wheat again (repeat), sim2 switches to field_beans
    sim1.step(action_wheat)
    action_beans = FarmAction("none", [PlotAction(0, "field_beans", "medium", "ipm")] + [PlotAction(i, "fallow", "low", "none") for i in range(1, NUM_PLOTS)])
    sim2.step(action_beans)

    assert sim1.state.plots[0].soil_health < sim2.state.plots[0].soil_health, \
        "Repeat crop should produce worse soil than rotation"
