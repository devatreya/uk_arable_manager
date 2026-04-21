"""
Conservative rotation baseline.

Strategy: fixed 4-crop rotation across the 4 plots, medium fertiliser,
IPM pest control, no irrigation.

Rotation (one plot per position):
  Plot 0: wheat → field_beans → barley → cover_crop → repeat
  Plot 1: barley → cover_crop → wheat → field_beans → repeat
  Plot 2: field_beans → wheat → cover_crop → barley → repeat
  Plot 3: cover_crop → barley → field_beans → wheat → repeat

This ensures every plot gets a legume or cover break each 4-year cycle,
maintaining soil health above 0.5 throughout.

Expected behaviour:
  - Modest, stable profit throughout
  - Soil health maintained or slightly improving
  - Survives to full 40 quarters with decent terminal score
  - Underperforms greedy extractor early but outperforms it late
"""
from __future__ import annotations

from typing import Any, Dict, List

# 4-step rotation sequences per plot (indexed by quarter_in_cycle mod 4)
ROTATIONS = [
    ["wheat",        "field_beans", "barley",       "cover_crop"],   # plot 0
    ["barley",       "cover_crop",  "wheat",         "field_beans"],  # plot 1
    ["field_beans",  "wheat",       "cover_crop",    "barley"],       # plot 2
    ["cover_crop",   "barley",      "field_beans",   "wheat"],        # plot 3
]


def policy(state: Dict[str, Any]) -> Dict[str, Any]:
    quarter = state.get("quarter", 0)
    irrigation_owned = state.get("irrigation_owned", False)
    cash = state.get("cash", 150_000.0)
    starting_cash = state.get("starting_cash", 150_000.0)

    cycle_pos = quarter % 4
    plot_plans: List[Dict[str, Any]] = []

    # Buy irrigation once if we have a modest cash cushion (> 40% of starting)
    capital_action = "none"
    if not irrigation_owned and cash > starting_cash * 0.40:
        capital_action = "buy_irrigation"

    for i in range(4):
        crop = ROTATIONS[i][cycle_pos]

        # Use medium fertiliser on cash crops, low on cover/fallow
        fert = "medium" if crop not in ("cover_crop", "fallow") else "low"

        # IPM for cash crops, none for cover/fallow
        pest = "ipm" if crop not in ("cover_crop", "fallow") else "none"

        plot_plans.append({"crop": crop, "fertiliser": fert, "pest_control": pest})

    return {
        "capital_action": capital_action,
        "plots": plot_plans,
    }


BASELINE_NAME = "conservative_rotation"
