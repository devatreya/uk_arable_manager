"""
Weather-aware rotation baseline.

Strategy: adapts crop and input choices based on observed weather regime.

Key design decisions:
  - In dry regime: prioritise field_beans over wheat/barley (better drought tolerance
    AND soil restoration). Buy irrigation more aggressively (threshold 30% of cash).
  - In wet regime: use spray pest control, avoid OSR on wet plots (sclerotinia risk).
  - In normal regime: standard rotation with IPM and medium inputs.
  - Soil guardian: if any plot soil falls below 0.35, force cover_crop there next quarter.

Rotation base: same 4-plot, 4-year base as conservative_rotation.
OSR → field_beans substitution in dry (OSR is damaging AND drought-sensitive).
Wheat stays as wheat in dry (wheat→barley gives minimal benefit and risks repeat-crop penalty).

Expected behaviour:
  - Best overall terminal_score due to smarter soil management
  - Outperforms conservative in drought via earlier irrigation + OSR avoidance
  - Slightly better than conservative in normal conditions due to soil guardian
"""
from __future__ import annotations

from typing import Any, Dict, List

BASE_ROTATIONS = [
    ["wheat",        "field_beans", "barley",       "cover_crop"],
    ["barley",       "cover_crop",  "wheat",         "field_beans"],
    ["field_beans",  "wheat",       "cover_crop",    "barley"],
    ["cover_crop",   "barley",      "field_beans",   "wheat"],
]

# Only substitute OSR: it's most damaging AND poor under drought
DRY_SUBSTITUTIONS = {
    "oilseed_rape": "field_beans",
}

WET_SUBSTITUTIONS = {
    "oilseed_rape": "wheat",  # OSR gets sclerotinia in wet; swap to wheat
}

# Soil guardian threshold: force cover_crop below this level
SOIL_GUARDIAN_THRESHOLD = 0.35
SOIL_GUARDIAN_CRITICAL = 0.28   # force field_beans if soil is really low


def _recent_regime(state: Dict[str, Any], lookback: int = 2) -> str:
    """Return dominant regime from recent weather history."""
    history = state.get("weather_history", [])
    recent = history[-lookback:] if history else []
    if not recent:
        return state.get("weather_regime", "normal")
    regimes = [w.get("regime", "normal") if isinstance(w, dict) else w.regime for w in recent]
    dry_count = regimes.count("dry")
    wet_count = regimes.count("wet")
    if dry_count >= len(regimes) // 2 + 1:
        return "dry"
    if wet_count >= len(regimes) // 2 + 1:
        return "wet"
    return "normal"


def policy(state: Dict[str, Any]) -> Dict[str, Any]:
    quarter = state.get("quarter", 0)
    irrigation_owned = state.get("irrigation_owned", False)
    cash = state.get("cash", 150_000.0)
    starting_cash = state.get("starting_cash", 150_000.0)
    plots = state.get("plots", [])

    regime = _recent_regime(state)
    cycle_pos = quarter % 4

    # Irrigation: buy aggressively, especially in dry conditions
    capital_action = "none"
    if not irrigation_owned:
        threshold = 0.30 if regime == "dry" else 0.38
        if cash > starting_cash * threshold:
            capital_action = "buy_irrigation"

    plot_plans: List[Dict[str, Any]] = []

    for i in range(4):
        crop = BASE_ROTATIONS[i][cycle_pos]
        prev_crop = "fallow"
        soil = 0.55

        if plots and i < len(plots):
            pd = plots[i] if isinstance(plots[i], dict) else {}
            prev_crop = pd.get("previous_crop", "fallow")
            # Compute soil from sub-components if available
            om = pd.get("_organic_matter", 0.55)
            st = pd.get("_structure", 0.55)
            ph = pd.get("_ph", 0.55)
            nu = pd.get("_nutrient_balance", 0.55)
            soil = 0.45 * om + 0.20 * st + 0.15 * ph + 0.20 * nu

        # Regime substitution
        if regime == "dry":
            candidate = DRY_SUBSTITUTIONS.get(crop, crop)
        elif regime == "wet":
            candidate = WET_SUBSTITUTIONS.get(crop, crop)
        else:
            candidate = crop

        # Avoid repeat crop penalty: if substitution would repeat the previous crop,
        # fall back to field_beans (breaks the repeat AND is restorative)
        if candidate != "fallow" and candidate != "cover_crop":
            if candidate == prev_crop:
                candidate = "field_beans"

        # Soil guardian: override to restorative crop when soil is critically low
        if soil < SOIL_GUARDIAN_CRITICAL and candidate not in ("cover_crop", "fallow", "field_beans"):
            candidate = "cover_crop"
        elif soil < SOIL_GUARDIAN_THRESHOLD and candidate not in ("cover_crop", "fallow", "field_beans", "barley"):
            candidate = "field_beans"

        crop = candidate

        # Fertiliser
        if crop in ("cover_crop", "fallow"):
            fert = "low"
        elif regime == "dry":
            fert = "medium"   # high wasteful when rain-limited
        else:
            fert = "medium"

        # Pest control
        if crop in ("cover_crop", "fallow"):
            pest = "none"
        elif regime == "wet":
            pest = "spray"    # elevated pressure in wet
        else:
            pest = "ipm"

        plot_plans.append({"crop": crop, "fertiliser": fert, "pest_control": pest})

    return {
        "capital_action": capital_action,
        "plots": plot_plans,
    }


BASELINE_NAME = "weather_aware_rotation"
