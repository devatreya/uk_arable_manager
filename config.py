"""
Central configuration for uk_arable_manager.
All economic, biological, and environmental constants live here.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# ── Farm layout ──────────────────────────────────────────────────────────────
FARM_ACRES = 400
NUM_PLOTS = 4
ACRES_PER_PLOT = FARM_ACRES // NUM_PLOTS  # 100

QUARTERS_PER_YEAR = 4
YEARS = 10
TOTAL_QUARTERS = YEARS * QUARTERS_PER_YEAR  # 40

PLOT_IDS = list(range(NUM_PLOTS))

# ── Crops ────────────────────────────────────────────────────────────────────
CROPS = ["wheat", "barley", "oilseed_rape", "field_beans", "cover_crop", "fallow"]
CASH_CROPS = ["wheat", "barley", "oilseed_rape", "field_beans"]
RESTORATIVE_CROPS = ["field_beans", "cover_crop", "fallow"]

# ── Gross revenue per acre (£) ───────────────────────────────────────────────
GROSS_REVENUE: Dict[str, float] = {
    "wheat": 700.0,
    "barley": 620.0,
    "oilseed_rape": 760.0,
    "field_beans": 540.0,
    "cover_crop": 0.0,
    "fallow": 0.0,
}

# ── Direct cost per acre (£, pre-addons) ────────────────────────────────────
DIRECT_COST: Dict[str, float] = {
    "wheat": 420.0,
    "barley": 380.0,
    "oilseed_rape": 470.0,
    "field_beans": 300.0,
    "cover_crop": 45.0,
    "fallow": 10.0,
}

# ── Fertiliser add-on cost per acre (£) ────────────────────────────────────
FERTILISER_COST: Dict[str, float] = {
    "low": 20.0,
    "medium": 45.0,
    "high": 75.0,
}

FERTILISER_YIELD_MULT: Dict[str, float] = {
    "low": 0.88,
    "medium": 1.00,
    "high": 1.12,
}

FERTILISER_SOIL_PENALTY: Dict[str, float] = {
    "low": 0.000,
    "medium": 0.000,
    "high": -0.020,
}

FERTILISER_NUTRIENT_BOOST: Dict[str, float] = {
    "low": -0.005,
    "medium": 0.010,
    "high": 0.030,
}

# ── Pest control ─────────────────────────────────────────────────────────────
PEST_CONTROL_COST: Dict[str, float] = {
    "none": 0.0,
    "ipm": 18.0,
    "spray": 40.0,
}

# Yield multiplier when pest pressure is elevated
PEST_CONTROL_YIELD_MULT: Dict[str, float] = {
    "none": 0.74,
    "ipm": 0.95,
    "spray": 1.00,
}

# ── Irrigation ───────────────────────────────────────────────────────────────
IRRIGATION_ONE_TIME_COST = 35_000.0
IRRIGATION_DRY_YIELD_BONUS = 0.18  # yield uplift above no-irrigation baseline in dry conditions

# ── Soil sub-component weights (latent aggregate) ────────────────────────────
SOIL_WEIGHT_OM = 0.45
SOIL_WEIGHT_STRUCTURE = 0.20
SOIL_WEIGHT_PH = 0.15
SOIL_WEIGHT_NUTRIENT = 0.20

# ── Soil delta rules (applied to aggregate, then distributed) ────────────────
SOIL_DELTA: Dict[str, float] = {
    "wheat": -0.050,
    "barley": -0.040,
    "oilseed_rape": -0.060,
    "field_beans": +0.020,
    "cover_crop": +0.060,
    "fallow": +0.030,
}

SOIL_REPEAT_PENALTY = -0.030
SOIL_DRY_STRESS = -0.018  # soil health penalty per dry quarter without irrigation
SOIL_MIN = 0.20
SOIL_MAX = 1.30

# Per-sub-component delta multipliers (relative sensitivity)
SOIL_OM_SENSITIVITY = 1.20
SOIL_STRUCTURE_SENSITIVITY = 0.90
SOIL_PH_SENSITIVITY = 0.60
SOIL_NUTRIENT_SENSITIVITY = 1.00

# ── Weather ──────────────────────────────────────────────────────────────────
WEATHER_REGIMES = ["normal", "dry", "wet"]

WEATHER_TRANSITION: Dict[str, Dict[str, float]] = {
    "normal": {"normal": 0.70, "dry": 0.18, "wet": 0.12},
    "dry":    {"normal": 0.35, "dry": 0.55, "wet": 0.10},
    "wet":    {"normal": 0.45, "dry": 0.05, "wet": 0.50},
}

WEATHER_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "normal": {"rainfall_index": (1.00, 0.10), "temperature_index": (1.00, 0.08)},
    "dry":    {"rainfall_index": (0.58, 0.12), "temperature_index": (1.12, 0.10)},
    "wet":    {"rainfall_index": (1.48, 0.15), "temperature_index": (0.91, 0.09)},
}

LOCAL_NORMALS = {
    "mean_annual_rainfall_mm": 565,
    "mean_summer_temp_c": 17.2,
    "mean_winter_temp_c": 4.1,
    "frost_days_per_year": 42,
    "source": "Met Office Cambridge NIAB historical normals",
}

DROUGHT_THRESHOLD = 0.70

# ── Pest pressure ────────────────────────────────────────────────────────────
PEST_PRESSURE_BASE = 0.28
PEST_WET_BONUS = 0.15
PEST_DRY_BONUS = 0.05

# ── Economics ────────────────────────────────────────────────────────────────
STARTING_CASH_DEFAULT = 150_000.0
PRICE_VOL_DEFAULT = 0.08
BANKRUPTCY_HARD_THRESHOLD = -60_000.0

# ── Reward ───────────────────────────────────────────────────────────────────
REWARD_SCALE = 1e-4

TERMINAL_SOIL_MIN = 0.40
TERMINAL_SOIL_MAX = 1.20
SOLVENCY_GATE_PENALTY = 0.20

# ── Split scales ─────────────────────────────────────────────────────────────
SPLIT_SCALES: Dict[str, Dict[str, int]] = {
    "dry_run": {"train": 24, "validation": 8,  "test": 8},
    "medium":  {"train": 64, "validation": 16, "test": 16},
    "full":    {"train": 128, "validation": 32, "test": 32},
}
DEFAULT_SCALE = "dry_run"

# ── Scenario mix proportions ─────────────────────────────────────────────────
SCENARIO_MIX = {
    "standard":         0.50,
    "drought_stressed": 0.25,
    "input_cost_shock": 0.15,
    "recovery":         0.10,
}

# ── Data paths ───────────────────────────────────────────────────────────────
import pathlib

ROOT = pathlib.Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

TASK_FILES = {
    "train":      DATA_PROCESSED / "scenario_tasks_train.json",
    "validation": DATA_PROCESSED / "scenario_tasks_validation.json",
    "test":       DATA_PROCESSED / "scenario_tasks_test.json",
}

# Real historical data (written by fetch_weather.py / fetch_prices.py)
QUARTERLY_WEATHER_PATH = DATA_PROCESSED / "quarterly_weather.json"
QUARTERLY_PRICES_PATH  = DATA_PROCESSED / "quarterly_prices.json"

# Valid simulation start-year range: 10-year window must fit inside 2000-2023.
# Effective start range is 2007-2014: avoids very low pre-reform wheat prices
# (2000-2006) while covering the 2007-08 commodity spike, 2009 crash, 2012 drought.
REAL_DATA_START_YEAR = 2000
REAL_DATA_END_YEAR   = 2023
SIM_MAX_START_YEAR   = REAL_DATA_END_YEAR - 10 + 1  # 2014
