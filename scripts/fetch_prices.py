"""
Fetch / stub UK arable crop and fertiliser price data.

Real data sources:
  - AHDB Cereals & Oilseeds ex-farm prices:
      https://ahdb.org.uk/cereals-oilseeds/market-information/prices
  - AHDB Fertiliser prices:
      https://ahdb.org.uk/arable/nitrogen-fertiliser-prices
  - Defra Agricultural Price Indices:
      https://www.gov.uk/government/collections/agricultural-price-indices

NOTE: Real download not yet implemented.  Stub files contain synthetic
      values calibrated to 2018-2024 AHDB averages.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_RAW


def write_wheat_stub() -> None:
    path = DATA_RAW / "wheat_exfarm.csv"
    if path.exists():
        print(f"  {path} already exists, skipping.")
        return
    rng = random.Random(101)
    rows = [["year", "quarter", "price_per_tonne_gbp", "source"]]
    base = 180.0
    for year in range(2010, 2024):
        for q in range(1, 5):
            p = round(base * rng.uniform(0.82, 1.22), 2)
            rows.append([year, q, p, "synthetic_AHDB_calibrated"])
            base = p * rng.uniform(0.97, 1.03)
    _write_csv(path, rows)


def write_barley_stub() -> None:
    path = DATA_RAW / "barley_exfarm.csv"
    if path.exists():
        print(f"  {path} already exists, skipping.")
        return
    rng = random.Random(102)
    rows = [["year", "quarter", "price_per_tonne_gbp", "source"]]
    base = 160.0
    for year in range(2010, 2024):
        for q in range(1, 5):
            p = round(base * rng.uniform(0.82, 1.20), 2)
            rows.append([year, q, p, "synthetic_AHDB_calibrated"])
            base = p * rng.uniform(0.97, 1.03)
    _write_csv(path, rows)


def write_osr_stub() -> None:
    path = DATA_RAW / "osr_delivered.csv"
    if path.exists():
        print(f"  {path} already exists, skipping.")
        return
    rng = random.Random(103)
    rows = [["year", "quarter", "price_per_tonne_gbp", "source"]]
    base = 360.0
    for year in range(2010, 2024):
        for q in range(1, 5):
            p = round(base * rng.uniform(0.80, 1.25), 2)
            rows.append([year, q, p, "synthetic_AHDB_calibrated"])
            base = p * rng.uniform(0.96, 1.04)
    _write_csv(path, rows)


def write_fertiliser_stub() -> None:
    path = DATA_RAW / "fertiliser_weekly.csv"
    if path.exists():
        print(f"  {path} already exists, skipping.")
        return
    rng = random.Random(104)
    rows = [["year", "week", "an_34_5_per_tonne_gbp", "source"]]
    base = 250.0
    for year in range(2010, 2024):
        for week in range(1, 53, 4):
            p = round(base * rng.uniform(0.75, 1.40), 2)
            rows.append([year, week, p, "synthetic_AHDB_calibrated"])
            base = p * rng.uniform(0.96, 1.04)
    _write_csv(path, rows)


def write_field_beans_stub() -> None:
    path = DATA_RAW / "field_beans_anchor.json"
    if path.exists():
        print(f"  {path} already exists, skipping.")
        return
    data = {
        "source": "synthetic_AHDB_calibrated",
        "note": "Field bean ex-farm prices are less liquid; using anchor price with modest vol",
        "anchor_price_per_tonne_gbp": 220.0,
        "typical_range": [170.0, 280.0],
        "price_volatility_annual_pct": 18.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote stub → {path}")


def _write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  Wrote stub → {path}")


def main(real: bool = False) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if real:
        print("Real AHDB download not yet implemented.  Writing stubs.")
    write_wheat_stub()
    write_barley_stub()
    write_osr_stub()
    write_fertiliser_stub()
    write_field_beans_stub()
    print("Price data ready.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    main(real=args.real)
