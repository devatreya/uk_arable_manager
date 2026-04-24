"""
OpenReward environment: UKArableManager
Wraps FarmSimulator and exposes the five required tools via the openreward SDK.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field

from openreward.environments import Environment, Split, TextBlock, ToolOutput, tool

from config import (
    BANKRUPTCY_HARD_THRESHOLD,
    CLIMATE_NORMALS_PATH,
    CROPS,
    DATA_PROCESSED,
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


def _load_climate_normals() -> Dict[str, Any]:
    try:
        with open(CLIMATE_NORMALS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _runtime_log(event: str, **fields: Any) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    message = f"[uk_arable_manager] event={event}"
    if suffix:
        message = f"{message} {suffix}"
    print(message, flush=True)


def _state_metadata(sim: FarmSimulator) -> Dict[str, Any]:
    state = sim.state.to_dict()
    state["current_prices"] = sim.get_current_prices()
    return {"state": state}


def _episode_metrics_metadata(sim: FarmSimulator, terminal_score: Optional[float]) -> Dict[str, Any]:
    state = sim.state
    mean_final_soil = sum(plot.soil_health for plot in state.plots) / NUM_PLOTS
    finished = state.quarter >= TOTAL_QUARTERS or state.cash < BANKRUPTCY_HARD_THRESHOLD
    return {
        "cash": float(state.cash),
        "starting_cash": float(state.starting_cash),
        "mean_final_soil": float(mean_final_soil),
        "quarter": int(state.quarter),
        "ever_bankrupt": bool(state.ever_bankrupt),
        "finished": bool(finished),
        "terminal_score": terminal_score,
    }


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
        le=16,
        description="Number of recent quarters to include (includes pre-episode context).",
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

_CLIMATE_NORMALS: Dict[str, Any] = _load_climate_normals()
_runtime_log(
    "data_source_selected",
    data_processed=DATA_PROCESSED,
    train_tasks=TASK_FILES["train"],
)


class UKArableManager(Environment):
    """
    A 400-acre Cambridgeshire arable farm over 10 crop years (40 quarters).
    Four 100-acre plots.  Balance short-term profit against long-term soil health.
    """

    def __init__(self, task_spec: Dict[str, Any] = {}, secrets: dict = {}) -> None:
        super().__init__(task_spec, secrets)
        self.sim: Optional[FarmSimulator] = None

    # ── OpenReward lifecycle ──────────────────────────────────────────────────

    def setup(self) -> None:
        self.sim = FarmSimulator(dict(self.task_spec))
        _runtime_log(
            "setup",
            task_id=self.task_spec.get("task_id", "unknown"),
            split=self.task_spec.get("split", "train"),
        )

    def teardown(self) -> None:
        _runtime_log(
            "teardown",
            task_id=self.task_spec.get("task_id", "unknown"),
            split=self.task_spec.get("split", "train"),
            quarter=(self.sim.state.quarter if self.sim is not None else "na"),
        )
        self.sim = None

    def _ensure_sim(self) -> FarmSimulator:
        if self.sim is None:
            self.setup()
        assert self.sim is not None
        return self.sim

    # ── Class methods ─────────────────────────────────────────────────────────

    @classmethod
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
            tasks = [_minimal_task(split, 0)]
        _runtime_log(
            "list_tasks",
            split=split,
            count=len(tasks),
            task_file=TASK_FILES.get(split),
        )
        return tasks

    def get_prompt(self) -> List[TextBlock]:
        spec = dict(self.task_spec)
        sim  = self._ensure_sim()
        s    = sim.state

        task_id      = spec.get("task_id", "unknown")
        scenario     = spec.get("scenario_type", "standard")
        starting_cash = float(spec.get("starting_cash", 150_000.0))

        # ── Recent 2024-2025 weather context ──────────────────────────────────
        recent_ctx: List[Dict[str, Any]] = spec.get("recent_weather_context", [])
        ctx_lines: List[str] = []
        if recent_ctx:
            ctx_lines.append("RECENT WEATHER CONTEXT  (actual ERA5, 2024-2025)")
            ctx_lines.append(f"  {'Year':>4}  {'Q':>2}  {'Regime':<7}  {'Rain idx':>8}  {'Temp idx':>8}  {'Rain mm':>8}  {'Temp °C':>8}")
            for q in recent_ctx:
                ctx_lines.append(
                    f"  {q['year']:>4}  {q['quarter']:>2}  {q['regime']:<7}  "
                    f"{q['rainfall_index']:>8.3f}  {q['temperature_index']:>8.3f}  "
                    f"{q.get('rain_mm', 0):>8.1f}  {q.get('temp_c', 0):>8.2f}"
                )
        else:
            ctx_lines.append("RECENT WEATHER CONTEXT  (not available)")

        # ── 30-year climate normals ───────────────────────────────────────────
        normals = _CLIMATE_NORMALS
        normals_lines: List[str] = []
        qn = normals.get("quarterly_normals", {})
        if qn:
            period = normals.get("period", "1991-2020")
            annual = normals.get("annual_mean_rain_mm", 632)
            normals_lines.append(f"30-YEAR CLIMATE NORMALS  ({period}, ERA5, Cambridge 52.2°N 0.1°E)")
            normals_lines.append(f"  Annual mean rainfall: {annual:.0f} mm")
            normals_lines.append(f"  {'Season':<8}  {'Mean rain':>9}  {'±':>1}  {'Std':>6}  {'Mean temp':>9}  {'±':>1}  {'Std':>5}")
            season_names = {"1": "Q1 (Jan-Mar)", "2": "Q2 (Apr-Jun)", "3": "Q3 (Jul-Sep)", "4": "Q4 (Oct-Dec)"}
            for q_key in ("1", "2", "3", "4"):
                n = qn.get(q_key, {})
                normals_lines.append(
                    f"  {season_names[q_key]:<12}  {n.get('mean_rain_mm', 0):>9.1f}  ±  {n.get('std_rain_mm', 0):>5.1f} mm"
                    f"  {n.get('mean_temp_c', 0):>9.1f}  ±  {n.get('std_temp_c', 0):>4.1f} °C"
                )
            normals_lines.append("  Index = 1.0 means exactly the seasonal average for that quarter.")
        else:
            normals_lines.append(f"30-YEAR CLIMATE NORMALS  (Cambridge NIAB, Met Office)")
            for k, v in LOCAL_NORMALS.items():
                if k != "source":
                    normals_lines.append(f"  {k}: {v}")

        # ── Starting farm state ───────────────────────────────────────────────
        farm_lines = [
            f"STARTING FARM STATE",
            f"  Cash: £{starting_cash:,.0f}  |  Irrigation: {'installed' if s.irrigation_owned else 'not installed'}",
            f"  {'Plot':<8}  {'Crop':<15}  {'Soil score':>10}",
        ]
        initial_crops = spec.get("initial_crop_by_plot", [])
        initial_soil  = spec.get("initial_soil_by_plot", [])
        for i in range(NUM_PLOTS):
            crop = initial_crops[i] if i < len(initial_crops) else s.plots[i].current_crop
            soil = initial_soil[i]  if i < len(initial_soil)  else s.plots[i].soil_score()
            farm_lines.append(f"  plot_{i+1:<4}  {crop:<15}  {soil:>10.3f}")

        # ── Full prompt ───────────────────────────────────────────────────────
        prompt = f"""You are managing a 400-acre arable farm in Cambridgeshire, East Anglia, over 10 crop years (40 quarterly turns).

FARM LAYOUT
  4 plots of 100 acres each (plot_1 through plot_4)
  Task ID: {task_id}  |  Scenario: {scenario}

YOUR GOAL
  Maximise: terminal_score = max(0, ending_cash / starting_cash) × soil_factor × solvency_gate
    soil_factor   = clip(mean_final_soil, 0.4, 1.2), linearly scaled → [0, 1]
    solvency_gate = 1.0 if never bankrupt, else 0.2

{chr(10).join(ctx_lines)}

{chr(10).join(normals_lines)}

{chr(10).join(farm_lines)}

AVAILABLE CROPS
  wheat          — £700 gross/acre, £420 direct cost/acre  (soil −0.050/qtr)
  barley         — £620 gross/acre, £380 direct cost/acre  (soil −0.040/qtr)
  oilseed_rape   — £760 gross/acre, £470 direct cost/acre  (soil −0.060/qtr)
  field_beans    — £540 gross/acre, £300 direct cost/acre  (soil +0.020/qtr, fixes nitrogen)
  cover_crop     — £0 revenue,       £45 cost/acre         (soil +0.060/qtr)
  fallow         — £0 revenue,       £10 cost/acre         (soil +0.030/qtr)

SOIL HEALTH DYNAMICS
  Repeating the same crop on the same plot: −0.030 extra per quarter
  High fertiliser: yield +12%, soil −0.020/qtr
  Dry weather without irrigation: soil −0.018/qtr, yield ×0.92

FERTILISER OPTIONS  (add-on cost per acre)
  low: +£20, yield ×0.88  |  medium: +£45, yield ×1.00  |  high: +£75, yield ×1.12

PEST CONTROL OPTIONS  (add-on cost per acre, matters when pest pressure is elevated)
  none: £0, yield ×0.74 under pressure  |  ipm: £18, ×0.95  |  spray: £40, ×1.00

CAPITAL ACTION
  buy_irrigation — one-time £35,000; in dry quarters yield +18% above no-irrigation baseline

TOOL SEQUENCE  (each quarter)
  1. read_farm_state      — cash, quarter, irrigation, current crops and soil
  2. read_soil_report     — organic matter, pH, compaction, nutrient balance per plot
  3. read_weather_history — realised weather history + climate context
  4. read_price_board     — current crop prices, fertiliser/irrigation costs
  5. commit_plan          — submit decisions for all 4 plots → advances time by one quarter

Greedy extraction collapses soil health; diverse rotations with restorative crops win over 40 quarters."""

        return [_tb(prompt)]

    # ── Tools ─────────────────────────────────────────────────────────────────

    @tool
    def read_farm_state(self) -> ToolOutput:
        """Return quarter index, year, cash, irrigation ownership, and current crop per plot."""
        sim = self._ensure_sim()
        s = sim.state
        _runtime_log(
            "read_farm_state",
            task_id=self.task_spec.get("task_id", "unknown"),
            quarter=s.quarter,
        )
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
        metadata = _state_metadata(sim)
        metadata["tool"] = "read_farm_state"
        return ToolOutput(blocks=[_tb("\n".join(lines))], reward=0.0, finished=False, metadata=metadata)

    @tool
    def read_soil_report(self, params: ReadSoilInput) -> ToolOutput:
        """Return soil sub-component readings for requested plots."""
        sim = self._ensure_sim()
        s = sim.state
        _runtime_log(
            "read_soil_report",
            task_id=self.task_spec.get("task_id", "unknown"),
            quarter=s.quarter,
            plots=",".join(str(p) for p in params.plots),
        )
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
        metadata = {
            **_state_metadata(sim),
            "tool": "read_soil_report",
            "plots": sorted(set(int(i) for i in params.plots if 0 <= i < NUM_PLOTS)),
        }
        return ToolOutput(blocks=[_tb("\n".join(lines))], reward=0.0, finished=False, metadata=metadata)

    @tool
    def read_weather_history(self, params: ReadWeatherInput) -> ToolOutput:
        """Return recent realised weather (including pre-episode context) and 30-year climate normals."""
        sim = self._ensure_sim()
        history = sim.state.weather_history
        lookback = params.lookback_quarters
        _runtime_log(
            "read_weather_history",
            task_id=self.task_spec.get("task_id", "unknown"),
            quarter=sim.state.quarter,
            lookback=lookback,
        )
        recent = history[-lookback:] if history else []

        lines = [f"WEATHER HISTORY (last {lookback} entries, negative quarters = pre-episode context)"]
        if not recent:
            lines.append("  No history yet (first quarter).")
        else:
            for w in recent:
                tag = "ctx" if w.quarter < 0 else f"Q{w.quarter + 1:02d}"
                lines.append(
                    f"  [{tag}] regime={w.regime:<7}  rain_idx={w.rainfall_index:.3f}  temp_idx={w.temperature_index:.3f}"
                )

        lines.append("")
        qn = _CLIMATE_NORMALS.get("quarterly_normals", {})
        if qn:
            lines.append("30-YEAR SEASONAL NORMALS  (1991-2020 ERA5 Cambridge)")
            season_names = {"1": "Q1 Jan-Mar", "2": "Q2 Apr-Jun", "3": "Q3 Jul-Sep", "4": "Q4 Oct-Dec"}
            for q_key in ("1", "2", "3", "4"):
                n = qn.get(q_key, {})
                lines.append(
                    f"  {season_names[q_key]}: rain {n.get('mean_rain_mm',0):.0f}±{n.get('std_rain_mm',0):.0f}mm  "
                    f"temp {n.get('mean_temp_c',0):.1f}±{n.get('std_temp_c',0):.1f}°C"
                )
        else:
            lines.append("LOCAL CLIMATOLOGY (Cambridge NIAB normals)")
            for k, v in LOCAL_NORMALS.items():
                if k != "source":
                    lines.append(f"  {k}: {v}")

        metadata = {
            **_state_metadata(sim),
            "tool": "read_weather_history",
            "lookback_quarters": int(lookback),
            "recent_weather": [w.to_dict() for w in recent],
        }
        return ToolOutput(blocks=[_tb("\n".join(lines))], reward=0.0, finished=False, metadata=metadata)

    @tool
    def read_price_board(self) -> ToolOutput:
        """Return current-quarter crop prices and input costs."""
        sim = self._ensure_sim()
        prices = sim.get_current_prices()
        _runtime_log(
            "read_price_board",
            task_id=self.task_spec.get("task_id", "unknown"),
            quarter=sim.state.quarter,
        )
        lines = ["PRICE BOARD (current quarter)"]
        lines.append("  Crop prices (£/acre gross revenue):")
        for crop in CROPS:
            if prices.get(crop, 0) > 0:
                lines.append(f"    {crop}: £{prices[crop]:.2f}")
        lines.append("  Fertiliser add-on cost (£/acre):")
        lines.append(f"    low=£{prices.get('fertiliser_low', 20):.2f}  "
                     f"medium=£{prices.get('fertiliser_medium', 45):.2f}  "
                     f"high=£{prices.get('fertiliser_high', 75):.2f}")
        if not sim.state.irrigation_owned:
            lines.append(f"  Irrigation (one-time): £{prices.get('irrigation_cost', 35000):.2f}")
        metadata = {
            **_state_metadata(sim),
            "tool": "read_price_board",
            "prices": prices,
        }
        return ToolOutput(blocks=[_tb("\n".join(lines))], reward=0.0, finished=False, metadata=metadata)

    @tool
    def commit_plan(self, params: CommitPlanInput) -> ToolOutput:
        """Submit quarterly plan for all 4 plots.  Advances simulation by one quarter.
        Returns updated observations, per-step reward, and finished flag."""
        sim = self._ensure_sim()
        s = sim.state
        _runtime_log(
            "commit_plan_start",
            task_id=self.task_spec.get("task_id", "unknown"),
            quarter=s.quarter,
            capital_action=params.capital_action,
        )

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
            lines.append("WARNING: Cash is NEGATIVE — bankruptcy risk.")

        lines.append("")
        lines.append("Updated soil scores:")
        for i in range(NUM_PLOTS):
            lines.append(f"  plot_{i+1}: {s.plots[i].soil_score():.3f}")

        if result.finished:
            lines.append("")
            lines.append("EPISODE COMPLETE")
            lines.append(f"Terminal score:      {result.terminal_score:.4f}")
            lines.append(f"Final cash:          £{s.cash:,.0f}")
            mean_soil = sum(p.soil_health for p in s.plots) / NUM_PLOTS
            lines.append(f"Mean final soil:     {mean_soil:.3f}")
            lines.append(f"Ever bankrupt:       {s.ever_bankrupt}")

        metadata = {
            **_state_metadata(sim),
            "tool": "commit_plan",
            "step": {
                "reward": float(result.reward),
                "pnl": float(result.pnl),
                "terminal_score": result.terminal_score,
                "finished": bool(result.finished),
                "bankrupt": bool(result.bankrupt),
                "plot_pnl": [float(p) for p in result.plot_pnl],
                "weather": result.weather.to_dict(),
                "pest_pressure": [bool(p) for p in result.pest_pressure],
                "irrigation_purchased": bool(result.irrigation_purchased),
            },
            "episode_metrics": _episode_metrics_metadata(sim, result.terminal_score),
        }
        _runtime_log(
            "commit_plan_end",
            task_id=self.task_spec.get("task_id", "unknown"),
            quarter=s.quarter,
            finished=result.finished,
            reward=round(float(result.reward), 6),
            cash=round(float(s.cash), 2),
        )
        return ToolOutput(
            blocks=[_tb("\n".join(lines))],
            reward=result.reward,
            finished=result.finished,
            metadata=metadata,
        )


# ── Minimal fallback task ─────────────────────────────────────────────────────

def _minimal_task(split: str, index: int) -> Dict[str, Any]:
    return {
        "task_id": f"fallback_{split}_{index:04d}",
        "seed": index,
        "split": split,
        "scenario_type": "standard",
        "real_data_mode": False,
        "starting_cash": 150_000.0,
        "initial_weather_regime": "normal",
        "dry_bias": 0.0,
        "price_volatility": 0.08,
        "fertiliser_cost_multiplier": 1.0,
        "irrigation_cost_multiplier": 1.0,
        "initial_soil_by_plot": [0.55, 0.55, 0.55, 0.55],
        "initial_crop_by_plot": ["fallow", "fallow", "fallow", "fallow"],
        "recent_weather_context": [],
    }
