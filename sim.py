"""
Pure-Python deterministic farm simulator.
All randomness flows through a seeded numpy RNG.  No live network calls.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config import (
    ACRES_PER_PLOT,
    BANKRUPTCY_HARD_THRESHOLD,
    CROPS,
    DIRECT_COST,
    DROUGHT_THRESHOLD,
    FERTILISER_COST,
    FERTILISER_NUTRIENT_BOOST,
    FERTILISER_SOIL_PENALTY,
    FERTILISER_YIELD_MULT,
    GROSS_REVENUE,
    IRRIGATION_DRY_YIELD_BONUS,
    IRRIGATION_ONE_TIME_COST,
    LOCAL_NORMALS,
    NUM_PLOTS,
    PEST_CONTROL_COST,
    PEST_CONTROL_YIELD_MULT,
    PEST_DRY_BONUS,
    PEST_PRESSURE_BASE,
    PEST_WET_BONUS,
    PRICE_VOL_DEFAULT,
    REWARD_SCALE,
    SOIL_DELTA,
    SOIL_DRY_STRESS,
    SOIL_MAX,
    SOIL_MIN,
    SOIL_OM_SENSITIVITY,
    SOIL_NUTRIENT_SENSITIVITY,
    SOIL_PH_SENSITIVITY,
    SOIL_REPEAT_PENALTY,
    SOIL_STRUCTURE_SENSITIVITY,
    SOIL_WEIGHT_NUTRIENT,
    SOIL_WEIGHT_OM,
    SOIL_WEIGHT_PH,
    SOIL_WEIGHT_STRUCTURE,
    SOLVENCY_GATE_PENALTY,
    STARTING_CASH_DEFAULT,
    TERMINAL_SOIL_MAX,
    TERMINAL_SOIL_MIN,
    TOTAL_QUARTERS,
    WEATHER_PARAMS,
    WEATHER_TRANSITION,
)


# ── Plot state ────────────────────────────────────────────────────────────────

@dataclass
class PlotState:
    plot_id: int
    current_crop: str = "fallow"
    previous_crop: str = "fallow"
    # Latent sub-components, all normalised to ~[0, 1.3]
    _organic_matter: float = 0.55
    _structure: float = 0.55
    _ph: float = 0.55
    _nutrient_balance: float = 0.55

    @property
    def soil_health(self) -> float:
        return (
            SOIL_WEIGHT_OM * self._organic_matter
            + SOIL_WEIGHT_STRUCTURE * self._structure
            + SOIL_WEIGHT_PH * self._ph
            + SOIL_WEIGHT_NUTRIENT * self._nutrient_balance
        )

    # ── Reported observables ─────────────────────────────────────────────────

    def reported_organic_matter(self) -> float:
        """Organic matter fraction, e.g. 0.30–0.90."""
        return round(float(np.clip(self._organic_matter * 0.80, 0.10, 1.20)), 3)

    def reported_ph(self) -> float:
        """Real pH scale 5.5–8.0."""
        return round(5.5 + float(np.clip(self._ph, 0.0, 1.0)) * 2.5, 2)

    def reported_compaction(self) -> str:
        if self._structure > 0.70:
            return "low"
        if self._structure > 0.45:
            return "moderate"
        return "high"

    def reported_nutrient_balance(self) -> float:
        return round(float(np.clip(self._nutrient_balance, 0.0, 1.3)), 3)

    def soil_score(self) -> float:
        return round(float(self.soil_health), 3)

    # ── State mutation ───────────────────────────────────────────────────────

    def apply_soil_update(
        self,
        crop: str,
        fertiliser: str,
        rainfall_index: float,
        irrigation_owned: bool,
        rng: np.random.Generator,
    ) -> None:
        delta = SOIL_DELTA.get(crop, 0.0)

        if crop not in ("fallow", "cover_crop") and crop == self.previous_crop:
            delta += SOIL_REPEAT_PENALTY

        delta += FERTILISER_SOIL_PENALTY.get(fertiliser, 0.0)

        if rainfall_index < DROUGHT_THRESHOLD and not irrigation_owned:
            delta += SOIL_DRY_STRESS

        # Distribute delta to sub-components with different sensitivities
        noise = lambda s=0.006: float(rng.normal(0.0, s))

        self._organic_matter = float(np.clip(
            self._organic_matter + delta * SOIL_OM_SENSITIVITY + noise(), SOIL_MIN, SOIL_MAX
        ))
        self._structure = float(np.clip(
            self._structure + delta * SOIL_STRUCTURE_SENSITIVITY + noise(0.005), SOIL_MIN, SOIL_MAX
        ))
        self._ph = float(np.clip(
            self._ph + delta * SOIL_PH_SENSITIVITY + noise(0.003), SOIL_MIN, SOIL_MAX
        ))
        nutrient_delta = delta * SOIL_NUTRIENT_SENSITIVITY + FERTILISER_NUTRIENT_BOOST.get(fertiliser, 0.0)
        self._nutrient_balance = float(np.clip(
            self._nutrient_balance + nutrient_delta + noise(), SOIL_MIN, SOIL_MAX
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plot_id": self.plot_id,
            "current_crop": self.current_crop,
            "previous_crop": self.previous_crop,
            "_organic_matter": round(self._organic_matter, 4),
            "_structure": round(self._structure, 4),
            "_ph": round(self._ph, 4),
            "_nutrient_balance": round(self._nutrient_balance, 4),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PlotState:
        p = cls(plot_id=d["plot_id"])
        p.current_crop = d["current_crop"]
        p.previous_crop = d["previous_crop"]
        p._organic_matter = d["_organic_matter"]
        p._structure = d["_structure"]
        p._ph = d["_ph"]
        p._nutrient_balance = d["_nutrient_balance"]
        return p


# ── Weather record ────────────────────────────────────────────────────────────

@dataclass
class WeatherRecord:
    quarter: int
    regime: str
    rainfall_index: float
    temperature_index: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quarter": self.quarter,
            "regime": self.regime,
            "rainfall_index": self.rainfall_index,
            "temperature_index": self.temperature_index,
        }


# ── Farm state ────────────────────────────────────────────────────────────────

@dataclass
class FarmState:
    quarter: int = 0
    cash: float = STARTING_CASH_DEFAULT
    starting_cash: float = STARTING_CASH_DEFAULT
    irrigation_owned: bool = False
    ever_bankrupt: bool = False
    weather_regime: str = "normal"
    weather_history: List[WeatherRecord] = field(default_factory=list)
    plots: List[PlotState] = field(default_factory=lambda: [PlotState(i) for i in range(NUM_PLOTS)])
    current_prices: Dict[str, float] = field(default_factory=dict)

    @property
    def year(self) -> int:
        return self.quarter // 4

    @property
    def quarter_in_year(self) -> int:
        return self.quarter % 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quarter": self.quarter,
            "cash": round(self.cash, 2),
            "starting_cash": self.starting_cash,
            "irrigation_owned": self.irrigation_owned,
            "ever_bankrupt": self.ever_bankrupt,
            "weather_regime": self.weather_regime,
            "weather_history": [w.to_dict() for w in self.weather_history],
            "plots": [p.to_dict() for p in self.plots],
            "current_prices": self.current_prices,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FarmState:
        fs = cls()
        fs.quarter = d["quarter"]
        fs.cash = d["cash"]
        fs.starting_cash = d["starting_cash"]
        fs.irrigation_owned = d["irrigation_owned"]
        fs.ever_bankrupt = d.get("ever_bankrupt", False)
        fs.weather_regime = d["weather_regime"]
        fs.weather_history = [WeatherRecord(**w) for w in d["weather_history"]]
        fs.plots = [PlotState.from_dict(p) for p in d["plots"]]
        fs.current_prices = d.get("current_prices", {})
        return fs


# ── Actions ───────────────────────────────────────────────────────────────────

@dataclass
class PlotAction:
    plot_id: int
    crop: str
    fertiliser: str
    pest_control: str


@dataclass
class FarmAction:
    capital_action: str  # "none" | "buy_irrigation"
    plots: List[PlotAction]


# ── Step result ───────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    reward: float
    pnl: float
    terminal_score: Optional[float]
    finished: bool
    bankrupt: bool
    plot_pnl: List[float]
    weather: WeatherRecord
    pest_pressure: List[bool]
    irrigation_purchased: bool


# ── Simulator ─────────────────────────────────────────────────────────────────

class FarmSimulator:
    """
    Deterministic farm simulator.  Same task_spec + same seed reproduces identical episodes.
    """

    def __init__(self, task_spec: Dict[str, Any]) -> None:
        self.task_spec = task_spec
        seed = int(task_spec.get("seed", 42))
        self.rng = np.random.default_rng(seed)
        self.state = self._build_initial_state(task_spec)
        self._resample_prices()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _build_initial_state(self, spec: Dict[str, Any]) -> FarmState:
        fs = FarmState()
        fs.starting_cash = float(spec.get("starting_cash", STARTING_CASH_DEFAULT))
        fs.cash = fs.starting_cash
        fs.weather_regime = spec.get("initial_weather_regime", "normal")
        fs.irrigation_owned = False
        fs.ever_bankrupt = False

        initial_soil: List[float] = spec.get("initial_soil_by_plot", [0.55] * NUM_PLOTS)
        initial_crop: List[str] = spec.get("initial_crop_by_plot", ["fallow"] * NUM_PLOTS)

        for i in range(NUM_PLOTS):
            p = fs.plots[i]
            soil_val = float(initial_soil[i]) if i < len(initial_soil) else 0.55
            p._organic_matter = soil_val
            p._structure = soil_val
            p._ph = soil_val
            p._nutrient_balance = soil_val
            crop_val = initial_crop[i] if i < len(initial_crop) else "fallow"
            p.previous_crop = crop_val
            p.current_crop = crop_val

        return fs

    # ── Prices ───────────────────────────────────────────────────────────────

    def _resample_prices(self) -> None:
        vol = float(self.task_spec.get("price_volatility", PRICE_VOL_DEFAULT))
        fert_mult = float(self.task_spec.get("fertiliser_cost_multiplier", 1.0))
        irr_mult = float(self.task_spec.get("irrigation_cost_multiplier", 1.0))

        prices: Dict[str, float] = {}
        for crop in CROPS:
            base = GROSS_REVENUE[crop]
            if base > 0.0:
                noise = float(np.clip(self.rng.normal(1.0, vol), 0.60, 1.50))
                prices[crop] = round(base * noise, 2)
            else:
                prices[crop] = 0.0

        prices["fertiliser_low"] = round(20.0 * fert_mult, 2)
        prices["fertiliser_medium"] = round(45.0 * fert_mult, 2)
        prices["fertiliser_high"] = round(75.0 * fert_mult, 2)
        prices["irrigation_cost"] = round(IRRIGATION_ONE_TIME_COST * irr_mult, 2)

        self.state.current_prices = prices

    def get_current_prices(self) -> Dict[str, float]:
        return dict(self.state.current_prices)

    # ── Weather ──────────────────────────────────────────────────────────────

    def _step_weather(self) -> WeatherRecord:
        regime = self.state.weather_regime
        dry_bias = float(self.task_spec.get("dry_bias", 0.0))

        trans = dict(WEATHER_TRANSITION[regime])
        if dry_bias > 0.0:
            shift = dry_bias * 0.20
            trans["dry"] = min(0.90, trans["dry"] + shift)
            trans["normal"] = max(0.05, trans["normal"] - shift * 0.70)
            trans["wet"] = max(0.05, trans["wet"] - shift * 0.30)
            total = sum(trans.values())
            trans = {k: v / total for k, v in trans.items()}

        states = list(trans.keys())
        probs = [trans[s] for s in states]
        new_regime = str(self.rng.choice(states, p=probs))
        self.state.weather_regime = new_regime

        mu_r, sd_r = WEATHER_PARAMS[new_regime]["rainfall_index"]
        mu_t, sd_t = WEATHER_PARAMS[new_regime]["temperature_index"]
        rainfall = float(np.clip(self.rng.normal(mu_r, sd_r), 0.10, 2.50))
        temperature = float(np.clip(self.rng.normal(mu_t, sd_t), 0.50, 1.80))

        rec = WeatherRecord(
            quarter=self.state.quarter,
            regime=new_regime,
            rainfall_index=round(rainfall, 3),
            temperature_index=round(temperature, 3),
        )
        self.state.weather_history.append(rec)
        return rec

    # ── Pest pressure ────────────────────────────────────────────────────────

    def _compute_pest_pressure(self, regime: str) -> List[bool]:
        p = PEST_PRESSURE_BASE
        if regime == "wet":
            p += PEST_WET_BONUS
        elif regime == "dry":
            p += PEST_DRY_BONUS
        return [bool(self.rng.random() < p) for _ in range(NUM_PLOTS)]

    # ── Yield computation ────────────────────────────────────────────────────

    @staticmethod
    def _weather_yield_mult(rainfall: float, temperature: float) -> float:
        # r_eff: softer rainfall sensitivity (~15% reduction at 0.58 index vs ~27% before)
        # so that weather_yield_mult alone doesn't already doom the farm in drought
        r_eff = min(1.0, rainfall) * 0.40 + 0.60
        t_eff = max(0.80, 1.0 - abs(temperature - 1.0) * 0.40)
        return r_eff * t_eff

    def _compute_plot_pnl(
        self,
        plot: PlotState,
        action: PlotAction,
        weather: WeatherRecord,
        pest_elevated: bool,
        fert_mult: float,
    ) -> float:
        crop = action.crop
        s = self.state

        # Revenue
        base_price = s.current_prices.get(crop, GROSS_REVENUE[crop])
        y = 1.0
        y *= FERTILISER_YIELD_MULT[action.fertiliser]
        y *= self._weather_yield_mult(weather.rainfall_index, weather.temperature_index)

        if weather.rainfall_index < DROUGHT_THRESHOLD:
            if s.irrigation_owned:
                # Irrigation supplements moisture deficit: clear yield uplift
                y *= (1.0 + IRRIGATION_DRY_YIELD_BONUS)
            else:
                # Mild extra penalty for unirrigated crop under moisture stress
                # (most drought effect already in weather_yield_mult via low r_eff)
                y *= 0.92

        if pest_elevated:
            y *= PEST_CONTROL_YIELD_MULT.get(action.pest_control, 0.74)

        # Soil modifies yield: health 0.2→0.90 multiplier, 1.0→1.0, 1.3→1.12
        soil_y = 0.75 + 0.25 * min(1.0, plot.soil_health / 1.0)
        y *= soil_y

        revenue = base_price * y * ACRES_PER_PLOT

        # Costs
        direct = DIRECT_COST[crop] * ACRES_PER_PLOT
        fert_c = FERTILISER_COST[action.fertiliser] * fert_mult * ACRES_PER_PLOT
        pest_c = PEST_CONTROL_COST[action.pest_control] * ACRES_PER_PLOT
        total_cost = direct + fert_c + pest_c

        return round(revenue - total_cost, 2)

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self, action: FarmAction) -> StepResult:
        s = self.state
        if s.quarter >= TOTAL_QUARTERS:
            raise ValueError("Episode already finished (40 quarters elapsed)")

        irrigation_purchased = False
        irr_mult = float(self.task_spec.get("irrigation_cost_multiplier", 1.0))
        irr_cost = round(IRRIGATION_ONE_TIME_COST * irr_mult, 2)

        if action.capital_action == "buy_irrigation" and not s.irrigation_owned:
            s.cash -= irr_cost
            s.irrigation_owned = True
            irrigation_purchased = True

        weather = self._step_weather()
        pest_pressure = self._compute_pest_pressure(weather.regime)
        fert_mult = float(self.task_spec.get("fertiliser_cost_multiplier", 1.0))

        plot_pnl: List[float] = []
        plot_actions_by_id = {pa.plot_id: pa for pa in action.plots}

        for i in range(NUM_PLOTS):
            p = s.plots[i]
            pa = plot_actions_by_id.get(i)
            if pa is None:
                pa = PlotAction(plot_id=i, crop="fallow", fertiliser="low", pest_control="none")

            pnl = self._compute_plot_pnl(p, pa, weather, pest_pressure[i], fert_mult)
            plot_pnl.append(pnl)
            s.cash += pnl

            p.apply_soil_update(pa.crop, pa.fertiliser, weather.rainfall_index, s.irrigation_owned, self.rng)
            p.previous_crop = p.current_crop
            p.current_crop = pa.crop

        bankrupt = s.cash < 0.0
        if bankrupt:
            s.ever_bankrupt = True

        total_pnl = sum(plot_pnl)
        reward = round(total_pnl * REWARD_SCALE, 6)

        s.quarter += 1
        hard_stop = s.cash < BANKRUPTCY_HARD_THRESHOLD
        finished = (s.quarter >= TOTAL_QUARTERS) or hard_stop

        term_score: Optional[float] = None
        if finished:
            term_score = self.terminal_score()
            reward += term_score  # final step carries both shaping and terminal signal

        # Resample prices for next quarter
        if not finished:
            self._resample_prices()

        return StepResult(
            reward=round(reward, 6),
            pnl=round(total_pnl, 2),
            terminal_score=term_score,
            finished=finished,
            bankrupt=bankrupt,
            plot_pnl=plot_pnl,
            weather=weather,
            pest_pressure=pest_pressure,
            irrigation_purchased=irrigation_purchased,
        )

    # ── Terminal score ────────────────────────────────────────────────────────

    def terminal_score(self) -> float:
        s = self.state
        cash_ratio = max(0.0, s.cash / s.starting_cash)
        mean_soil = float(np.mean([p.soil_health for p in s.plots]))
        soil_clipped = float(np.clip(mean_soil, TERMINAL_SOIL_MIN, TERMINAL_SOIL_MAX))
        soil_factor = (soil_clipped - TERMINAL_SOIL_MIN) / (TERMINAL_SOIL_MAX - TERMINAL_SOIL_MIN)
        solvency = 1.0 if not s.ever_bankrupt else SOLVENCY_GATE_PENALTY
        return round(cash_ratio * soil_factor * solvency, 4)

    def snapshot(self) -> Dict[str, Any]:
        return self.state.to_dict()
