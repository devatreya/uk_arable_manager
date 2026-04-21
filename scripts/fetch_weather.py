"""
Fetch / stub Cambridge NIAB weather data.

Real data: Met Office / CEDA MIDAS weather station data for Cambridge NIAB.
Fallback: synthetic placeholder files so the environment works immediately.

Usage:
  python scripts/fetch_weather.py          # writes stub files if missing
  python scripts/fetch_weather.py --real   # attempts real download (not implemented yet)

Data sources:
  - Met Office Hadobs: https://www.metoffice.gov.uk/hadobs/haduk-grid/
  - CEDA archive: https://data.ceda.ac.uk/badc/ukmo-midas-open/
  - Cambridge NIAB station ID: varies by dataset; lat 52.24N lon 0.10E

NOTE: Real download is not implemented.  The placeholder files contain
      synthetic values calibrated to East Anglia climatology.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_RAW


def write_monthly_stub() -> None:
    path = DATA_RAW / "cambridge_niab_monthly.csv"
    if path.exists():
        print(f"  {path} already exists, skipping.")
        return

    # Synthetic monthly data: year, month, rainfall_mm, temp_mean_c
    # Calibrated to Cambridge NIAB normals (565mm annual, 10.0°C mean)
    monthly_normals = {
        1: (45, 4.0), 2: (32, 4.2), 3: (38, 6.5), 4: (42, 9.2),
        5: (46, 12.5), 6: (52, 15.8), 7: (48, 17.2), 8: (55, 17.0),
        9: (52, 14.5), 10: (56, 11.2), 11: (50, 7.0), 12: (49, 4.5),
    }

    rows = [["year", "month", "rainfall_mm", "temp_mean_c", "source"]]
    import random
    rng = random.Random(9999)
    for year in range(2000, 2024):
        for month, (r_norm, t_norm) in monthly_normals.items():
            r = round(r_norm * rng.uniform(0.5, 1.6), 1)
            t = round(t_norm + rng.gauss(0, 0.8), 2)
            rows.append([year, month, r, t, "synthetic_placeholder"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  Wrote stub → {path}")


def write_normals_stub() -> None:
    path = DATA_RAW / "cambridge_niab_normals.csv"
    if path.exists():
        print(f"  {path} already exists, skipping.")
        return

    rows = [
        ["month", "rainfall_mm_normal", "temp_mean_c_normal", "source"],
        [1,  45, 4.0, "Met Office 1991-2020 normals (synthetic)"],
        [2,  32, 4.2, ""],
        [3,  38, 6.5, ""],
        [4,  42, 9.2, ""],
        [5,  46, 12.5, ""],
        [6,  52, 15.8, ""],
        [7,  48, 17.2, ""],
        [8,  55, 17.0, ""],
        [9,  52, 14.5, ""],
        [10, 56, 11.2, ""],
        [11, 50, 7.0, ""],
        [12, 49, 4.5, ""],
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  Wrote stub → {path}")


def main(real: bool = False) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if real:
        print("Real download not yet implemented.  Writing stubs instead.")
    write_monthly_stub()
    write_normals_stub()
    print("Weather data ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Attempt real data download")
    args = parser.parse_args()
    main(real=args.real)
