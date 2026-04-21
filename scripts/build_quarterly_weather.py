"""
Build a quarterly weather lookup table from raw monthly CSV.

Output: data/processed/quarterly_weather.json
  { "year_quarter": {"rainfall_index": float, "temperature_index": float}, ... }

Indices are normalised relative to local normals.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED, DATA_RAW, LOCAL_NORMALS


MONTHLY_NORMALS = {
    1: (45, 4.0), 2: (32, 4.2), 3: (38, 6.5), 4: (42, 9.2),
    5: (46, 12.5), 6: (52, 15.8), 7: (48, 17.2), 8: (55, 17.0),
    9: (52, 14.5), 10: (56, 11.2), 11: (50, 7.0), 12: (49, 4.5),
}

# Map calendar months to farm quarters (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec)
MONTH_TO_Q = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}


def main() -> None:
    raw_path = DATA_RAW / "cambridge_niab_monthly.csv"
    if not raw_path.exists():
        print(f"  {raw_path} not found — run fetch_weather.py first")
        return

    # Aggregate monthly → quarterly
    from collections import defaultdict
    quarterly: dict = defaultdict(lambda: {"rain": [], "temp": []})

    with open(raw_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["year"])
            month = int(row["month"])
            q = MONTH_TO_Q[month]
            key = f"{year}_Q{q}"
            quarterly[key]["rain"].append(float(row["rainfall_mm"]))
            quarterly[key]["temp"].append(float(row["temp_mean_c"]))

    # Quarterly normals
    q_normals_rain = {}
    q_normals_temp = {}
    for month, (r, t) in MONTHLY_NORMALS.items():
        q = MONTH_TO_Q[month]
        q_normals_rain[q] = q_normals_rain.get(q, 0) + r
        q_normals_temp[q] = q_normals_temp.get(q, [])
        q_normals_temp[q].append(t)
    q_normals_temp = {q: sum(v) / len(v) for q, v in q_normals_temp.items()}

    output = {}
    for key, vals in sorted(quarterly.items()):
        year_str, q_str = key.split("_")
        q = int(q_str[1])
        r_total = sum(vals["rain"])
        t_mean = sum(vals["temp"]) / len(vals["temp"])
        r_norm = q_normals_rain.get(q, 1.0)
        t_norm = q_normals_temp.get(q, 10.0)
        output[key] = {
            "rainfall_index": round(r_total / r_norm, 3) if r_norm > 0 else 1.0,
            "temperature_index": round(t_mean / t_norm, 3) if t_norm > 0 else 1.0,
        }

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "quarterly_weather.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Wrote {len(output)} quarter records → {out_path}")


if __name__ == "__main__":
    main()
