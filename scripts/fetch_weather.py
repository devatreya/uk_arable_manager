"""
Fetch real historical weather for Cambridge from Open-Meteo Historical Archive API.
Data source: ERA5 reanalysis aggregated by Open-Meteo (https://open-meteo.com/)
Location: 52.20°N 0.10°E — Cambridge area, near NIAB trial site
Period: 2000-2023 (24 years × 4 quarters = 96 quarterly records)
No API key required.

Outputs:
  data/raw/cambridge_niab_quarterly.csv   — raw quarterly totals + indices
  data/processed/quarterly_weather.json  — nested {year: {quarter: {...}}} for sim.py
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED, DATA_RAW, DROUGHT_THRESHOLD

LAT, LON = 52.20, 0.10
START_DATE = "2000-01-01"
END_DATE   = "2023-12-31"
WET_THRESHOLD = 1.35  # rainfall_index above this = wet regime


def _fetch_daily() -> dict:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        "&daily=precipitation_sum,temperature_2m_mean"
        "&timezone=Europe%2FLondon"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "uk-arable-manager/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def _aggregate_to_quarters(data: dict) -> list[dict]:
    dates = data["daily"]["time"]
    rain  = data["daily"]["precipitation_sum"]
    temp  = data["daily"]["temperature_2m_mean"]

    # Accumulate daily → quarterly
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
            "rain_mm": round(sum(v["rain"]), 1)              if v["rain"] else 0.0,
            "temp_c":  round(sum(v["temp"]) / len(v["temp"]), 2) if v["temp"] else 10.0,
        })

    # Normalise: mean over the full period = 1.0
    rm = sum(r["rain_mm"] for r in records) / len(records)
    tm = sum(r["temp_c"]  for r in records) / len(records)
    for r in records:
        ri = round(r["rain_mm"] / rm, 4) if rm > 0 else 1.0
        ti = round(r["temp_c"]  / tm, 4) if tm != 0 else 1.0
        r["rainfall_index"]    = ri
        r["temperature_index"] = ti
        r["regime"] = (
            "dry"    if ri < DROUGHT_THRESHOLD else
            "wet"    if ri > WET_THRESHOLD     else
            "normal"
        )

    return records


def _save_raw_csv(records: list[dict]) -> None:
    path = DATA_RAW / "cambridge_niab_quarterly.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["year", "quarter", "rain_mm", "temp_c",
              "rainfall_index", "temperature_index", "regime"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"  Raw CSV → {path}  ({len(records)} rows)")


def _save_processed_json(records: list[dict]) -> None:
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

    reg = [r["regime"] for r in records]
    n   = len(records)
    rain_mean = sum(r["rain_mm"] for r in records) / n
    print(f"  Processed JSON → {path}")
    print(f"  {n} quarters | mean rainfall {rain_mean:.0f} mm/quarter")
    print(f"  Regime split: dry={reg.count('dry')} "
          f"normal={reg.count('normal')} wet={reg.count('wet')}")


def main() -> None:
    print(f"Fetching Cambridge weather from Open-Meteo ({START_DATE} → {END_DATE}) …")
    try:
        data = _fetch_daily()
    except urllib.error.URLError as e:
        print(f"ERROR: Could not reach Open-Meteo API: {e}")
        sys.exit(1)
    records = _aggregate_to_quarters(data)
    _save_raw_csv(records)
    _save_processed_json(records)
    print("Weather data ready.")


if __name__ == "__main__":
    main()
