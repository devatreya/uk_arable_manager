"""
Build a quarterly price board from raw crop/fertiliser CSVs.

Output: data/processed/price_board.json
  { "year_quarter": {"wheat": float, "barley": float, ...}, ... }

Prices are calibrated to gross revenue per acre (yield × price/tonne).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED, DATA_RAW, GROSS_REVENUE


# Approximate yields in t/ha → t/acre (*0.405) for price calibration
YIELD_T_PER_ACRE = {
    "wheat": 3.2,
    "barley": 2.8,
    "oilseed_rape": 1.4,
    "field_beans": 1.6,
}


def main() -> None:
    board: dict = defaultdict(dict)

    # Wheat
    wpath = DATA_RAW / "wheat_exfarm.csv"
    if wpath.exists():
        with open(wpath) as f:
            for row in csv.DictReader(f):
                key = f"{row['year']}_Q{row['quarter']}"
                board[key]["wheat"] = round(float(row["price_per_tonne_gbp"]) * YIELD_T_PER_ACRE["wheat"], 2)

    # Barley
    bpath = DATA_RAW / "barley_exfarm.csv"
    if bpath.exists():
        with open(bpath) as f:
            for row in csv.DictReader(f):
                key = f"{row['year']}_Q{row['quarter']}"
                board[key]["barley"] = round(float(row["price_per_tonne_gbp"]) * YIELD_T_PER_ACRE["barley"], 2)

    # OSR
    opath = DATA_RAW / "osr_delivered.csv"
    if opath.exists():
        with open(opath) as f:
            for row in csv.DictReader(f):
                key = f"{row['year']}_Q{row['quarter']}"
                board[key]["oilseed_rape"] = round(float(row["price_per_tonne_gbp"]) * YIELD_T_PER_ACRE["oilseed_rape"], 2)

    if not board:
        print("  No raw price files found.  Run fetch_prices.py first.")
        return

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "price_board.json"
    with open(out_path, "w") as f:
        json.dump(dict(board), f, indent=2)
    print(f"  Wrote {len(board)} quarter records → {out_path}")


if __name__ == "__main__":
    main()
