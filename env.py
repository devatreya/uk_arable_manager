"""
ORS environment: UKArableManager
Wraps FarmSimulator and exposes the five required tools via the ORS SDK.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field

from ors import Environment, Split, TextBlock, ToolOutput, tool

from config import (
    CROPS,
    LOCAL_NORMALS,
    NUM_PLOTS,
    TASK_FILES,
    TOTAL_QUARTERS,
)
from sim import FarmAction, FarmSimulator, PlotAction


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tb(text: str) -> TextBlock:
    return TextBlock(text=text)


def _load_tasks(split: str) -> List[Dict[str, Any]]:
    path = TASK_FILES.get(split)
    if path is None or not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# ── Tool input schemas ────────────────────────────────────────────────────────

class ReadSoilInput(BaseModel):
    plots: List[int] = Field(
        default_factory=lambda: [0, 1, 2, 3],
        description="Plot IDs to query (0-3).",
    )


class ReadWeatherInput(BaseModel):
    lookback_quarters: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Number of recent quarters to include in history.",
    )


class PlotPlanInput(BaseModel):
    crop: str = Field(description=f"One of: {CROPS}")
    fertiliser: str = Field(description="low | medium | high")
    pest_control: str = Field(description="none | ipm | spray")


class CommitPlanInput(BaseModel):
    capital_action: str = Field(
        default="none",
        description="none | buy_irrigation  (irrigation costs £35,000 once)",
    )
    plot_1: PlotPlanInput
    plot_2: PlotPlanInput
    plot_3: PlotPlanInput
    plot_4: PlotPlanInput


# ── Environment ───────────────────────────────────────────────────────────────

class UKArableManager(Environment):
    """
    A 400-acre Cambridgeshire arable farm over 10 crop years (40 quarters).
    Four 100-acre plots.  Balance short-term profit against long-term soil health.
    """

    def __init__(self, task_spec: Dict[str, Any] = {}, secrets: dict = {}) -> None:
        super().__init__(task_spec, secrets)
        self.sim: Optional[FarmSimulator] = None

    # ── ORS lifecycle ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        self.sim = FarmSimulator(dict(self.task_spec))

    def teardown(self) -> None:
        self.sim = None

    def _ensure_sim(self) -> FarmSimulator:
        if self.sim is None:
            self.setup()
        assert self.sim is not None
        return self.sim

    # ── ORS class methods ─────────────────────────────────────────────────────

    @classmethod
    @cache
    def list_splits(cls) -> Sequence[Union[Split, str]]:
        return [
            Split(name="train",      type="train"),
            Split(name="validation", type="validation"),
            Split(name="test",       type="test"),
        ]

    @classmethod
    def list_tasks(cls, split: str) -> Sequence[Dict[str, Any]]:
        tasks = _load_tasks(split)
        if not tasks:
            # Fallback: one minimal task so the env is usable before build_tasks.py runs
            tasks = [_minimal_task(split, 0)]
        return tasks

    def get_prompt(self) -> List[TextBlock]:
        spec = dict(self.task_spec)
        task_id = spec.get("task_id", "unknown")
        starting_cash = float(spec.get("starting_cash", 150_000.0))
        scenario = spec.get("scenario_type", "standard")

        prompt = f"""You are managing a 400-acre arable farm in Cambridgeshire, East Anglia, over 10 crop years (40 quarterly turns).

FARM LAYOUT
- 4 plots of 100 acres each (plot_1 through plot_4)
- Task ID: {task_id}  |  Scenario: {scenario}
- Starting cash: £{starting_cash:,.0f}

YOUR GOAL
Maximise: terminal_score = max(0, ending_cash / starting_cash) × soil_factor × solvency_gate
  soil_factor   = clip(mean_final_soil, 0.4, 1.2), linearly scaled to [0, 1]
  solvency_gate = 1.0 if never bankrupt, else 0.2

AVAILABLE CROPS
  wheat          — £700 gross/acre, £420 cost/acre
  barley         — £620 gross/acre, £380 cost/acre
  oilseed_rape   — £760 gross/acre, £470 cost/acre
  field_beans    — £540 gross/acre, £300 cost/acre
  cover_crop     — £0 revenue,       £45 cost/acre  (restores soil +0.06/quarter)
  fallow         — £0 revenue,       £10 cost/acre  (restores soil +0.03/quarter)

SOIL HEALTH DYNAMICS
  Wheat −0.05/qt  Barley −0.04  OSR −0.06  Beans +0.02  Cover +0.06  Fallow +0.03
  Repeating the same crop on the same plot: −0.03 extra per quarter
  High fertiliser: yield +12%, soil −0.02/qt
  Dry weather without irrigation: soil −0.03/qt, yield ×0.78

QUARTERLY DECISION
For each plot choose:
  crop        — one of the 6 options above
  fertiliser  — low | medium | high
  pest_control — none | ipm | spray  (matters when pest pressure is elevated)

Capital: buy_irrigation once for £35,000 (boosts yield in dry years by +12%)

TOOL SEQUENCE (each quarter):
1. read_farm_state      — cash, quarter, irrigation status, current crops
2. read_soil_report     — organic matter, pH, compaction, nutrient balance per plot
3. read_weather_history — recent rainfall/temperature, local climatology
4. read_price_board     — current crop prices, fertiliser costs
5. commit_plan          — submit decisions for all 4 plots → advances time

Greedy extraction degrades soil; sustainable rotation across different crops wins over 40 quarters."""

        return [_tb(prompt)]

    # ── Tools ─────────────────────────────────────────────────────────────────

    @tool
    def read_farm_state(self) -> ToolOutput:
        """Return quarter index, year, cash, irrigation ownership, and current crop per plot."""
        sim = self._ensure_sim()
        s = sim.state
        plots_summary = {
            f"plot_{i+1}": {
                "current_crop": s.plots[i].current_crop,
                "previous_crop": s.plots[i].previous_crop,
                "soil_score": s.plots[i].soil_score(),
            }
            for i in range(NUM_PLOTS)
        }
        lines = [
            f"Quarter: {s.quarter + 1}/{TOTAL_QUARTERS}  Year: {s.year + 1}/10  Q-in-year: {s.quarter_in_year + 1}",
            f"Cash: £{s.cash:,.0f}  |  Starting cash: £{s.starting_cash:,.0f}",
            f"Irrigation owned: {s.irrigation_owned}",
            "Plots:",
        ]
        for i in range(NUM_PLOTS):
            p = s.plots[i]
            lines.append(
                f"  plot_{i+1}: crop={p.current_crop}  prev={p.previous_crop}  soil={p.soil_score():.3f}"
            )
        return ToolOutput(blocks=[_tb("\n".join(lines))], reward=None, finished=False)

    @tool
    def read_soil_report(self, params: ReadSoilInput) -> ToolOutput:
        """Return soil sub-component readings for requested plots."""
        sim = self._ensure_sim()
        s = sim.state
        lines = ["SOIL REPORT"]
        for i in sorted(set(params.plots)):
            if i < 0 or i >= NUM_PLOTS:
                continue
            p = s.plots[i]
            lines.append(
                f"  plot_{i+1}: OM={p.reported_organic_matter():.3f}  "
                f"pH={p.reported_ph():.2f}  "
                f"compaction={p.reported_compaction()}  "
                f"nutrients={p.reported_nutrient_balance():.3f}  "
                f"soil_score={p.soil_score():.3f}"
            )
        return ToolOutput(blocks=[_tb("\n".join(lines))], reward=None, finished=False)

    @tool
    def read_weather_history(self, params: ReadWeatherInput) -> ToolOutput:
        """Return recent realised weather and fixed local climatology."""
        sim = self._ensure_sim()
        history = sim.state.weather_history
        lookback = params.lookback_quarters
        recent = history[-lookback:] if history else []

        lines = [f"WEATHER HISTORY (last {lookback} quarters)"]
        if not recent:
            lines.append("  No history yet (first quarter).")
        else:
            for w in recent:
                lines.append(
                    f"  Q{w.quarter+1}: regime={w.regime}  rainfall_idx={w.rainfall_index:.3f}  "
                    f"temp_idx={w.temperature_index:.3f}"
                )

        lines.append("")
        lines.append("LOCAL CLIMATOLOGY (Cambridge NIAB normals)")
        for k, v in LOCAL_NORMALS.items():
            if k != "source":
                lines.append(f"  {k}: {v}")
        lines.append(f"  (source: {LOCAL_NORMALS['source']})")

        return ToolOutput(blocks=[_tb("\n".join(lines))], reward=None, finished=False)

    @tool
    def read_price_board(self) -> ToolOutput:
        """Return current-quarter crop prices and input costs."""
        sim = self._ensure_sim()
        prices = sim.get_current_prices()
        lines = ["PRICE BOARD (current quarter)"]
        lines.append("  Crop prices (£/acre gross):")
        for crop in CROPS:
            if prices.get(crop, 0) > 0:
                lines.append(f"    {crop}: £{prices[crop]:.2f}")
        lines.append("  Fertiliser add-on cost (£/acre):")
        lines.append(f"    low=£{prices.get('fertiliser_low', 20):.2f}  "
                     f"medium=£{prices.get('fertiliser_medium', 45):.2f}  "
                     f"high=£{prices.get('fertiliser_high', 75):.2f}")
        if not sim.state.irrigation_owned:
            lines.append(f"  Irrigation (one-time): £{prices.get('irrigation_cost', 35000):.2f}")
        return ToolOutput(blocks=[_tb("\n".join(lines))], reward=None, finished=False)

    @tool
    def commit_plan(self, params: CommitPlanInput) -> ToolOutput:
        """Submit quarterly plan for all 4 plots.  Advances simulation by one quarter.
        Returns updated observations, per-step reward, and finished flag."""
        sim = self._ensure_sim()
        s = sim.state

        if s.quarter >= TOTAL_QUARTERS:
            return ToolOutput(
                blocks=[_tb("Episode already finished.")],
                reward=0.0,
                finished=True,
            )

        plots_input = [params.plot_1, params.plot_2, params.plot_3, params.plot_4]
        plot_actions = []
        for i, pi in enumerate(plots_input):
            crop = pi.crop.strip().lower()
            fert = pi.fertiliser.strip().lower()
            pest = pi.pest_control.strip().lower()

            if crop not in CROPS:
                return ToolOutput(
                    blocks=[_tb(f"Invalid crop '{crop}' for plot_{i+1}. Choose from: {CROPS}")],
                    reward=0.0,
                    finished=False,
                )
            if fert not in ("low", "medium", "high"):
                return ToolOutput(
                    blocks=[_tb(f"Invalid fertiliser '{fert}'. Use: low | medium | high")],
                    reward=0.0,
                    finished=False,
                )
            if pest not in ("none", "ipm", "spray"):
                return ToolOutput(
                    blocks=[_tb(f"Invalid pest_control '{pest}'. Use: none | ipm | spray")],
                    reward=0.0,
                    finished=False,
                )
            plot_actions.append(PlotAction(plot_id=i, crop=crop, fertiliser=fert, pest_control=pest))

        action = FarmAction(
            capital_action=params.capital_action.strip().lower(),
            plots=plot_actions,
        )

        result = sim.step(action)

        lines = [
            f"── Quarter {s.quarter}/{TOTAL_QUARTERS} complete ─────────────────",
            f"Weather: regime={result.weather.regime}  "
            f"rain={result.weather.rainfall_index:.3f}  "
            f"temp={result.weather.temperature_index:.3f}",
        ]

        if result.irrigation_purchased:
            lines.append("Capital: Irrigation system installed.")

        pest_str = ", ".join(
            f"plot_{i+1}={'ELEVATED' if p else 'normal'}"
            for i, p in enumerate(result.pest_pressure)
        )
        lines.append(f"Pest pressure: {pest_str}")
        lines.append("")
        lines.append("Plot P&L this quarter:")
        for i, pnl in enumerate(result.plot_pnl):
            lines.append(f"  plot_{i+1}: £{pnl:,.0f}")
        lines.append(f"Total P&L: £{result.pnl:,.0f}")
        lines.append(f"Cash balance: £{s.cash:,.0f}")

        if result.bankrupt:
            lines.append("⚠  Cash is NEGATIVE — bankruptcy risk.")

        lines.append("")
        lines.append("Updated soil scores:")
        for i in range(NUM_PLOTS):
            lines.append(f"  plot_{i+1}: {s.plots[i].soil_score():.3f}")

        if result.finished:
            lines.append("")
            lines.append("═══ EPISODE COMPLETE ═══")
            lines.append(f"Terminal score: {result.terminal_score:.4f}")
            lines.append(f"Final cash: £{s.cash:,.0f}")
            mean_soil = sum(p.soil_health for p in s.plots) / NUM_PLOTS
            lines.append(f"Mean final soil health: {mean_soil:.3f}")
            lines.append(f"Ever bankrupt: {s.ever_bankrupt}")

        return ToolOutput(
            blocks=[_tb("\n".join(lines))],
            reward=result.reward,
            finished=result.finished,
        )


# ── Minimal fallback task ─────────────────────────────────────────────────────

def _minimal_task(split: str, index: int) -> Dict[str, Any]:
    return {
        "task_id": f"fallback_{split}_{index:04d}",
        "seed": index,
        "split": split,
        "scenario_type": "standard",
        "starting_cash": 150_000.0,
        "initial_weather_regime": "normal",
        "dry_bias": 0.0,
        "price_volatility": 0.08,
        "fertiliser_cost_multiplier": 1.0,
        "irrigation_cost_multiplier": 1.0,
        "initial_soil_by_plot": [0.55, 0.55, 0.55, 0.55],
        "initial_crop_by_plot": ["fallow", "fallow", "fallow", "fallow"],
    }
