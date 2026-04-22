"""
Fetch real historical weather for Cambridge from Open-Meteo Historical Archive API.
Data source: ERA5 reanalysis via Open-Meteo (https://open-meteo.com/)
Location: 52.20°N 0.10°E — Cambridge area, near NIAB trial site
No API key required.

Fetches 1991-2025 daily precipitation and temperature.

Outputs:
  data/raw/cambridge_niab_quarterly.csv    — full 1991-2025 quarterly records
  data/processed/quarterly_weather.json   — {year: {quarter: {...}}}
  data/processed/climate_normals.json     — 30-year (1991-2020) seasonal stats per Q1..Q4
  data/processed/recent_weather.json      — last 2 years actual (2024-2025) for agent context
"""
from __future__ import annotations

import csv
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED, DATA_RAW, DROUGHT_THRESHOLD

LAT, LON = 52.20, 0.10
FETCH_START   = "1991-01-01"
FETCH_END     = "2025-12-31"
NORMAL_START  = 1991           # WMO 30-year climate normal period
NORMAL_END    = 2020
RECENT_YEARS  = [2024, 2025]   # most recent 2 full years for agent context
WET_THRESHOLD = 1.35


# ── Fetch ─────────────────────────────────────────────────────────────────────

def _fetch_daily() -> dict:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={FETCH_START}&end_date={FETCH_END}"
        "&daily=precipitation_sum,temperature_2m_mean"
        "&timezone=Europe%2FLondon"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "uk-arable-manager/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


# ── Aggregate daily → quarterly ───────────────────────────────────────────────

def _aggregate(data: dict) -> list[dict]:
    dates = data["daily"]["time"]
    rain  = data["daily"]["precipitation_sum"]
    temp  = data["daily"]["temperature_2m_mean"]

    buckets: dict = {}
    for d, r, t in zip(dates, rain, temp):
        y, m = int(d[:4]), int(d[5:7])
        q = (m - 1) // 3 + 1
        key = (y, q)
        buckets.setdefault(key, {"rain": [], "temp": []})
        if r is not None:
            buckets[key]["rain"].append(float(r))
        if t is not None:
            buckets[key]["temp"].append(float(t))

    records: list[dict] = []
    for (y, q), v in sorted(buckets.items()):
        records.append({
            "year":    y,
            "quarter": q,
            "rain_mm": round(sum(v["rain"]), 1)                  if v["rain"] else 0.0,
            "temp_c":  round(sum(v["temp"]) / len(v["temp"]), 2) if v["temp"] else 10.0,
        })
    return records


# ── Climate normals (1991-2020) ───────────────────────────────────────────────

def _compute_normals(records: list[dict]) -> dict:
    """
    Compute 30-year seasonal stats (mean + std) per quarter-of-year from the
    1991-2020 WMO normal period.

    Indices are normalised against each quarter's own seasonal mean so that
    index = 1.0 always means "exactly average for this time of year."
    """
    normal_recs = [r for r in records if NORMAL_START <= r["year"] <= NORMAL_END]

    stats: dict = {}
    for q_of_year in range(1, 5):
        subset = [r for r in normal_recs if r["quarter"] == q_of_year]
        rains = [r["rain_mm"] for r in subset]
        temps = [r["temp_c"]  for r in subset]

        def _mean(v): return sum(v) / len(v)
        def _std(v):
            m = _mean(v)
            return math.sqrt(sum((x - m)**2 for x in v) / max(len(v) - 1, 1))

        stats[str(q_of_year)] = {
            "mean_rain_mm": round(_mean(rains), 2),
            "std_rain_mm":  round(_std(rains),  2),
            "mean_temp_c":  round(_mean(temps),  2),
            "std_temp_c":   round(_std(temps),   2),
            "n_years":      len(subset),
        }

    annual_mean_rain = sum(stats[str(q)]["mean_rain_mm"] for q in range(1, 5))

    return {
        "period":            f"{NORMAL_START}-{NORMAL_END}",
        "location":          f"Cambridge {LAT}°N {LON}°E",
        "source":            "ERA5 via Open-Meteo",
        "annual_mean_rain_mm": round(annual_mean_rain, 1),
        "quarterly_normals": stats,
    }


# ── Normalise records → indices ───────────────────────────────────────────────

def _add_indices(records: list[dict], normals: dict) -> None:
    """
    Append rainfall_index and temperature_index to each record.
    Normalised against the seasonal mean (quarterly_normals) so index=1.0
    means exactly average for that quarter of year.
    """
    qn = normals["quarterly_normals"]
    for r in records:
        norm = qn[str(r["quarter"])]
        ri = r["rain_mm"] / norm["mean_rain_mm"] if norm["mean_rain_mm"] > 0 else 1.0
        ti = r["temp_c"]  / norm["mean_temp_c"]  if abs(norm["mean_temp_c"]) > 0.5 else 1.0
        r["rainfall_index"]    = round(ri, 4)
        r["temperature_index"] = round(ti, 4)
        r["regime"] = (
            "dry"    if ri < DROUGHT_THRESHOLD else
            "wet"    if ri > WET_THRESHOLD     else
            "normal"
        )


# ── Extract recent 2-year context ─────────────────────────────────────────────

def _extract_recent(records: list[dict]) -> list[dict]:
    return [
        {k: r[k] for k in ("year", "quarter", "rain_mm", "temp_c",
                            "rainfall_index", "temperature_index", "regime")}
        for r in records if r["year"] in RECENT_YEARS
    ]


# ── Save outputs ──────────────────────────────────────────────────────────────

def _save_raw_csv(records: list[dict]) -> None:
    path = DATA_RAW / "cambridge_niab_quarterly.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["year", "quarter", "rain_mm", "temp_c",
              "rainfall_index", "temperature_index", "regime"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"  Raw CSV → {path}  ({len(records)} rows, "
          f"{records[0]['year']}–{records[-1]['year']})")


def _save_quarterly_json(records: list[dict]) -> None:
    nested: dict = {}
    for r in records:
        nested.setdefault(str(r["year"]), {})[str(r["quarter"])] = {
            "rainfall_index":    r["rainfall_index"],
            "temperature_index": r["temperature_index"],
            "regime":            r["regime"],
        }
    path = DATA_PROCESSED / "quarterly_weather.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(nested, f, indent=2)
    print(f"  Quarterly weather JSON → {path}")


def _save_climate_normals(normals: dict) -> None:
    path = DATA_PROCESSED / "climate_normals.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(normals, f, indent=2)
    print(f"  Climate normals JSON → {path}")
    qn = normals["quarterly_normals"]
    for q in range(1, 5):
        n = qn[str(q)]
        print(f"    Q{q}: rain {n['mean_rain_mm']:.0f}±{n['std_rain_mm']:.0f}mm  "
              f"temp {n['mean_temp_c']:.1f}±{n['std_temp_c']:.1f}°C")


def _save_recent_weather(recent: list[dict]) -> None:
    path = DATA_PROCESSED / "recent_weather.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"years": RECENT_YEARS, "quarters": recent}, f, indent=2)
    print(f"  Recent weather ({RECENT_YEARS}) → {path}  ({len(recent)} quarters)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Fetching Cambridge weather from Open-Meteo "
          f"({FETCH_START} → {FETCH_END}) …")
    try:
        data = _fetch_daily()
    except urllib.error.URLError as e:
        print(f"ERROR: Open-Meteo unreachable: {e}")
        sys.exit(1)

    records = _aggregate(data)
    normals = _compute_normals(records)
    _add_indices(records, normals)

    _save_raw_csv(records)
    _save_quarterly_json(records)
    _save_climate_normals(normals)
    _save_recent_weather(_extract_recent(records))

    regs = [r["regime"] for r in records]
    print(f"\n  Full period: {len(records)} quarters | "
          f"dry={regs.count('dry')}  normal={regs.count('normal')}  wet={regs.count('wet')}")
    print("Weather data ready.")


if __name__ == "__main__":
    main()
