"""
Greedy extractor baseline.

Strategy: always plant the highest-margin cash crop on every plot, use
high fertiliser, spray pest control, never invest in irrigation.

Expected behaviour:
  - High P&L early (first 2-3 years)
  - Progressive soil degradation from repeated high-extraction crops
  - Revenue collapse by year 7-10 as soil health falls below 0.4
  - Likely to survive on cash but finish with poor terminal score due to soil

Calibration target: wins early, fails on terminal_score.
"""
from __future__ import annotations

from typing import Any, Dict, List


def policy(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Always plant oilseed_rape (highest gross margin), high fertiliser, spray.
    Cycles wheat/OSR to avoid the hardest repeat penalty, but still extracts.
    """
    plots = state.get("plots", [])
    plot_plans: List[Dict[str, Any]] = []

    for i in range(4):
        # Alternate wheat and OSR to avoid exact repeat penalty on same crop,
        # but both are high-extraction.  Plots 0,2 → OSR; plots 1,3 → wheat.
        crop = "oilseed_rape" if i % 2 == 0 else "wheat"

        # Nudge: if soil_score very low, switch to barley (slightly less damage)
        if plots and i < len(plots):
            soil = plots[i].get("soil_score", 0.55)
            if soil < 0.30:
                crop = "barley"

        plot_plans.append({
            "crop": crop,
            "fertiliser": "high",
            "pest_control": "spray",
        })

    return {
        "capital_action": "none",
        "plots": plot_plans,
    }


BASELINE_NAME = "greedy_extractor"
