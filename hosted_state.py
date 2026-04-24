from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from config import BANKRUPTCY_HARD_THRESHOLD, NUM_PLOTS, TOTAL_QUARTERS

_CURRENCY_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")
_QUARTER_RE = re.compile(
    r"Quarter:\s*(?P<quarter>\d+)/(?P<total>\d+)\s+Year:\s*(?P<year>\d+)/(?P<years>\d+)\s+Q-in-year:\s*(?P<quarter_in_year>\d+)"
)
_CASH_RE = re.compile(
    r"Cash:\s*£(?P<cash>[-0-9,\.]+)\s*\|\s*Starting cash:\s*£(?P<starting_cash>[-0-9,\.]+)"
)
_PLOT_STATE_RE = re.compile(
    r"plot_(?P<plot>\d+):\s*crop=(?P<crop>[a-z_]+)\s+prev=(?P<prev>[a-z_]+)\s+soil=(?P<soil>[0-9.]+)"
)
_SOIL_RE = re.compile(
    r"plot_(?P<plot>\d+):\s*OM=(?P<om>[0-9.]+)\s+pH=(?P<ph>[0-9.]+)\s+compaction=(?P<compaction>[a-z]+)\s+nutrients=(?P<nu>[0-9.]+)\s+soil_score=(?P<soil>[0-9.]+)"
)
_WEATHER_RE = re.compile(
    r"\[(?P<tag>[^\]]+)\]\s+regime=(?P<regime>[a-z]+)\s+rain_idx=(?P<rain>[0-9.]+)\s+temp_idx=(?P<temp>[0-9.]+)"
)
_COMMIT_HEADER_RE = re.compile(r"Quarter\s+(?P<quarter>\d+)/(?P<total>\d+)\s+complete")
_COMMIT_WEATHER_RE = re.compile(
    r"Weather:\s*regime=(?P<regime>[a-z]+)\s+rain=(?P<rain>[0-9.]+)\s+temp=(?P<temp>[0-9.]+)"
)
_PLOT_PNL_RE = re.compile(r"plot_(?P<plot>\d+):\s*£(?P<value>[-0-9,\.]+)")
_TOTAL_PNL_RE = re.compile(r"Total P&L:\s*£(?P<value>[-0-9,\.]+)")
_CASH_BALANCE_RE = re.compile(r"Cash balance:\s*£(?P<value>[-0-9,\.]+)")
_UPDATED_SOIL_RE = re.compile(r"plot_(?P<plot>\d+):\s*(?P<soil>[0-9.]+)$")
_TERMINAL_SCORE_RE = re.compile(r"Terminal score:\s*(?P<value>[-0-9.]+)")
_FINAL_CASH_RE = re.compile(r"Final cash:\s*£(?P<value>[-0-9,\.]+)")
_MEAN_FINAL_SOIL_RE = re.compile(r"Mean final soil:\s*(?P<value>[0-9.]+)")
_EVER_BANKRUPT_RE = re.compile(r"Ever bankrupt:\s*(?P<value>True|False)")
_PRICE_RE = re.compile(r"(?P<name>[a-z_]+):\s*£(?P<value>[0-9.]+)")
_FERT_COST_RE = re.compile(
    r"low=£(?P<low>[0-9.]+)\s+medium=£(?P<medium>[0-9.]+)\s+high=£(?P<high>[0-9.]+)"
)
_PEST_STATUS_RE = re.compile(r"plot_(?P<plot>\d+)=(?P<status>ELEVATED|normal)")


def _parse_currency(raw: str) -> float:
    match = _CURRENCY_RE.search(raw.replace("£", ""))
    if not match:
        raise ValueError(f"Could not parse currency from {raw!r}")
    return float(match.group(0).replace(",", ""))


def _parse_bool(raw: str) -> bool:
    return raw.strip() == "True"


def _mean_final_soil_from_state(state: dict[str, Any]) -> float:
    plots = state.get("plots", [])
    if not plots:
        return 0.55
    values: list[float] = []
    for plot in plots:
        values.append(
            0.45 * float(plot.get("_organic_matter", 0.55))
            + 0.20 * float(plot.get("_structure", 0.55))
            + 0.15 * float(plot.get("_ph", 0.55))
            + 0.20 * float(plot.get("_nutrient_balance", 0.55))
        )
    return sum(values) / len(values)


def initial_state_from_task(task_spec: dict[str, Any]) -> dict[str, Any]:
    starting_cash = float(task_spec.get("starting_cash", 150_000.0))
    initial_soil = list(task_spec.get("initial_soil_by_plot", [0.55] * NUM_PLOTS))
    initial_crop = list(task_spec.get("initial_crop_by_plot", ["fallow"] * NUM_PLOTS))
    weather_context = [
        {
            "quarter": int(ctx.get("quarter_offset", -(len(task_spec.get("recent_weather_context", [])) - index))),
            "regime": str(ctx.get("regime", "normal")),
            "rainfall_index": float(ctx.get("rainfall_index", 1.0)),
            "temperature_index": float(ctx.get("temperature_index", 1.0)),
        }
        for index, ctx in enumerate(task_spec.get("recent_weather_context", []))
    ]
    plots = []
    for index in range(NUM_PLOTS):
        soil = float(initial_soil[index]) if index < len(initial_soil) else 0.55
        crop = str(initial_crop[index]) if index < len(initial_crop) else "fallow"
        plots.append(
            {
                "plot_id": index,
                "current_crop": crop,
                "previous_crop": crop,
                "_organic_matter": soil,
                "_structure": soil,
                "_ph": soil,
                "_nutrient_balance": soil,
            }
        )
    return {
        "quarter": 0,
        "cash": starting_cash,
        "starting_cash": starting_cash,
        "irrigation_owned": False,
        "ever_bankrupt": False,
        "weather_regime": str(task_spec.get("initial_weather_regime", "normal")),
        "weather_history": weather_context,
        "plots": plots,
        "current_prices": {},
    }


def update_state_from_tool_output(
    tool_name: str,
    text: str,
    state: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    next_state = deepcopy(state)
    result: dict[str, Any] = {"state": next_state}

    if tool_name == "read_farm_state":
        _update_from_read_farm_state(text, next_state)
    elif tool_name == "read_soil_report":
        _update_from_soil_report(text, next_state)
    elif tool_name == "read_weather_history":
        _update_from_weather_history(text, next_state)
    elif tool_name == "read_price_board":
        _update_from_price_board(text, next_state)
    elif tool_name == "commit_plan":
        step = _update_from_commit_plan(text, next_state, payload or {})
        result["step"] = step
        result["episode_metrics"] = _episode_metrics_from_commit(text, next_state, step)

    return result


def _update_from_read_farm_state(text: str, state: dict[str, Any]) -> None:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        quarter_match = _QUARTER_RE.search(line)
        if quarter_match:
            state["quarter"] = max(0, int(quarter_match.group("quarter")) - 1)
            continue
        cash_match = _CASH_RE.search(line)
        if cash_match:
            state["cash"] = float(cash_match.group("cash").replace(",", ""))
            state["starting_cash"] = float(cash_match.group("starting_cash").replace(",", ""))
            continue
        if line.startswith("Irrigation owned:"):
            state["irrigation_owned"] = _parse_bool(line.split(":", 1)[1].strip())
            continue
        plot_match = _PLOT_STATE_RE.search(line)
        if plot_match:
            index = int(plot_match.group("plot")) - 1
            plot = state["plots"][index]
            plot["current_crop"] = plot_match.group("crop")
            plot["previous_crop"] = plot_match.group("prev")
            plot["_soil_score"] = float(plot_match.group("soil"))


def _update_from_soil_report(text: str, state: dict[str, Any]) -> None:
    for line in text.splitlines():
        match = _SOIL_RE.search(line.strip())
        if not match:
            continue
        index = int(match.group("plot")) - 1
        plot = state["plots"][index]
        organic_matter = max(0.0, min(1.3, float(match.group("om")) / 0.80))
        ph_component = max(0.0, min(1.3, (float(match.group("ph")) - 5.5) / 2.5))
        nutrient_balance = max(0.0, min(1.3, float(match.group("nu"))))
        soil_score = float(match.group("soil"))
        structure = (soil_score - 0.45 * organic_matter - 0.15 * ph_component - 0.20 * nutrient_balance) / 0.20
        plot["_organic_matter"] = round(max(0.0, min(1.3, organic_matter)), 4)
        plot["_ph"] = round(max(0.0, min(1.3, ph_component)), 4)
        plot["_nutrient_balance"] = round(max(0.0, min(1.3, nutrient_balance)), 4)
        plot["_structure"] = round(max(0.0, min(1.3, structure)), 4)
        plot["_soil_score"] = soil_score


def _update_from_weather_history(text: str, state: dict[str, Any]) -> None:
    weather_history: list[dict[str, Any]] = []
    ctx_index = -1
    for line in text.splitlines():
        match = _WEATHER_RE.search(line.strip())
        if not match:
            continue
        tag = match.group("tag")
        if tag.startswith("Q"):
            quarter = int(tag[1:]) - 1
        else:
            quarter = ctx_index
            ctx_index -= 1
        weather_history.append(
            {
                "quarter": quarter,
                "regime": match.group("regime"),
                "rainfall_index": float(match.group("rain")),
                "temperature_index": float(match.group("temp")),
            }
        )
    if weather_history:
        state["weather_history"] = weather_history
        state["weather_regime"] = weather_history[-1]["regime"]


def _update_from_price_board(text: str, state: dict[str, Any]) -> None:
    current_prices = dict(state.get("current_prices", {}))
    mode = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Crop prices"):
            mode = "crops"
            continue
        if stripped.startswith("Fertiliser add-on cost"):
            mode = "fertiliser"
            continue
        if stripped.startswith("Irrigation"):
            current_prices["irrigation_cost"] = _parse_currency(stripped)
            continue
        if mode == "crops":
            match = _PRICE_RE.search(stripped)
            if match:
                current_prices[match.group("name")] = float(match.group("value"))
        elif mode == "fertiliser":
            match = _FERT_COST_RE.search(stripped)
            if match:
                current_prices["fertiliser_low"] = float(match.group("low"))
                current_prices["fertiliser_medium"] = float(match.group("medium"))
                current_prices["fertiliser_high"] = float(match.group("high"))
    state["current_prices"] = current_prices


def _update_from_commit_plan(text: str, state: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    step: dict[str, Any] = {
        "pnl": 0.0,
        "terminal_score": None,
        "finished": False,
        "bankrupt": False,
        "plot_pnl": [0.0] * NUM_PLOTS,
        "weather": {},
        "pest_pressure": [False] * NUM_PLOTS,
        "irrigation_purchased": False,
    }
    updated_soils: dict[int, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        header_match = _COMMIT_HEADER_RE.search(stripped)
        if header_match:
            state["quarter"] = int(header_match.group("quarter"))
            continue
        weather_match = _COMMIT_WEATHER_RE.search(stripped)
        if weather_match:
            weather = {
                "quarter": max(0, int(state.get("quarter", 1)) - 1),
                "regime": weather_match.group("regime"),
                "rainfall_index": float(weather_match.group("rain")),
                "temperature_index": float(weather_match.group("temp")),
            }
            state["weather_regime"] = weather["regime"]
            history = list(state.get("weather_history", []))
            history.append(weather)
            state["weather_history"] = history[-16:]
            step["weather"] = weather
            continue
        if stripped == "Capital: Irrigation system installed.":
            step["irrigation_purchased"] = True
            state["irrigation_owned"] = True
            continue
        if stripped.startswith("Pest pressure:"):
            for pest_match in _PEST_STATUS_RE.finditer(stripped):
                step["pest_pressure"][int(pest_match.group("plot")) - 1] = pest_match.group("status") == "ELEVATED"
            continue
        plot_pnl_match = _PLOT_PNL_RE.search(stripped)
        if plot_pnl_match and stripped.startswith("plot_"):
            step["plot_pnl"][int(plot_pnl_match.group("plot")) - 1] = float(plot_pnl_match.group("value").replace(",", ""))
            continue
        total_match = _TOTAL_PNL_RE.search(stripped)
        if total_match:
            step["pnl"] = float(total_match.group("value").replace(",", ""))
            continue
        cash_match = _CASH_BALANCE_RE.search(stripped)
        if cash_match:
            state["cash"] = float(cash_match.group("value").replace(",", ""))
            continue
        if stripped.startswith("WARNING: Cash is NEGATIVE"):
            state["ever_bankrupt"] = True
            step["bankrupt"] = True
            continue
        updated_soil_match = _UPDATED_SOIL_RE.search(stripped)
        if updated_soil_match and stripped.startswith("plot_"):
            updated_soils[int(updated_soil_match.group("plot")) - 1] = float(updated_soil_match.group("soil"))
            continue
        terminal_match = _TERMINAL_SCORE_RE.search(stripped)
        if terminal_match:
            step["terminal_score"] = float(terminal_match.group("value"))
            step["finished"] = True
            continue
        final_cash_match = _FINAL_CASH_RE.search(stripped)
        if final_cash_match:
            state["cash"] = float(final_cash_match.group("value").replace(",", ""))
            continue
        bankrupt_match = _EVER_BANKRUPT_RE.search(stripped)
        if bankrupt_match:
            state["ever_bankrupt"] = _parse_bool(bankrupt_match.group("value"))

    for index in range(NUM_PLOTS):
        plot = state["plots"][index]
        planned = payload.get(f"plot_{index + 1}", {})
        plot["previous_crop"] = plot.get("current_crop", "fallow")
        plot["current_crop"] = planned.get("crop", plot.get("current_crop", "fallow"))
        if index in updated_soils:
            plot["_soil_score"] = updated_soils[index]

    if payload.get("capital_action") == "buy_irrigation":
        state["irrigation_owned"] = True

    if state.get("cash", 0.0) < 0.0:
        state["ever_bankrupt"] = True
        step["bankrupt"] = True

    step["finished"] = step["finished"] or int(state.get("quarter", 0)) >= TOTAL_QUARTERS
    return step


def _episode_metrics_from_commit(text: str, state: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    mean_final_soil_match = _MEAN_FINAL_SOIL_RE.search(text)
    mean_final_soil = (
        float(mean_final_soil_match.group("value"))
        if mean_final_soil_match
        else _mean_final_soil_from_state(state)
    )
    finished = bool(step.get("finished")) or state.get("cash", 0.0) < BANKRUPTCY_HARD_THRESHOLD
    return {
        "cash": float(state.get("cash", 0.0)),
        "starting_cash": float(state.get("starting_cash", 150_000.0)),
        "mean_final_soil": float(mean_final_soil),
        "quarter": int(state.get("quarter", 0)),
        "ever_bankrupt": bool(state.get("ever_bankrupt", False)),
        "finished": bool(finished),
        "terminal_score": step.get("terminal_score"),
    }
