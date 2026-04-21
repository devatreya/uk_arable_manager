"""
Build UK agricultural commodity price series for 2000-2023.

Primary data: real AHDB / DEFRA annual price averages embedded below.
  Source: AHDB Cereals & Oilseeds market data and DEFRA Agricultural Price Indices.
  Values: approximate annual UK ex-farm averages (£/tonne).

GBP/USD FX: fetched from ECB Statistical Data Warehouse (free, no key).
  Used only to cross-check; the primary series are already in GBP.

Outputs:
  data/raw/prices_annual_gbp.csv       — absolute annual prices (£/tonne)
  data/processed/quarterly_prices.json — {year: {q: {wheat_mult, barley_mult, …}}}

Quarterly prices are computed as:
  annual_price × seasonal_factor[q]
Multipliers are relative to each commodity's own 2000-2023 mean so the
simulator's calibrated GROSS_REVENUE constants stay valid at the mean.
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED, DATA_RAW

# ── Real historical UK prices (£/tonne, annual average) ─────────────────────
# Wheat:  AHDB feed wheat ex-farm East Anglia / UK average
# OSR:    AHDB oilseed rape delivered UK mill
# AN:     AHDB/DEFRA ammonium nitrate 34.5% N (bags)
# Barley: wheat × BARLEY_WHEAT_RATIO (AHDB ex-farm barley tracks wheat closely)
# Beans:  wheat × FIELDBEANS_WHEAT_RATIO

REAL_WHEAT: Dict[int, float] = {
    2000: 72,   2001: 82,   2002: 88,   2003: 92,   2004: 74,   2005: 78,
    2006: 103,  2007: 145,  2008: 175,  2009:  96,  2010: 133,  2011: 166,
    2012: 197,  2013: 160,  2014: 121,  2015: 106,  2016: 118,  2017: 147,
    2018: 177,  2019: 165,  2020: 167,  2021: 197,  2022: 288,  2023: 202,
}

REAL_OSR: Dict[int, float] = {
    2000: 148,  2001: 162,  2002: 183,  2003: 198,  2004: 182,  2005: 170,
    2006: 203,  2007: 273,  2008: 378,  2009: 248,  2010: 292,  2011: 393,
    2012: 451,  2013: 418,  2014: 332,  2015: 268,  2016: 331,  2017: 343,
    2018: 328,  2019: 362,  2020: 393,  2021: 472,  2022: 648,  2023: 423,
}

REAL_AN: Dict[int, float] = {
    2000: 132,  2001: 135,  2002: 147,  2003: 132,  2004: 142,  2005: 178,
    2006: 198,  2007: 198,  2008: 323,  2009: 197,  2010: 267,  2011: 328,
    2012: 322,  2013: 293,  2014: 282,  2015: 246,  2016: 202,  2017: 241,
    2018: 262,  2019: 250,  2020: 220,  2021: 409,  2022: 888,  2023: 432,
}

BARLEY_WHEAT_RATIO     = 0.87
FIELDBEANS_WHEAT_RATIO = 0.85
YEARS = list(range(2000, 2024))

# ── Seasonal within-year multipliers (Q1..Q4 must sum to 4.0) ────────────────
# Harvest is Q3 (Jul–Sep) → lowest prices; spring/forward-buying peaks in Q1/Q4.
SEASONAL: Dict[str, list] = {
    "wheat":       [1.04, 1.01, 0.93, 1.02],
    "barley":      [1.03, 1.01, 0.93, 1.03],
    "osr":         [1.02, 1.04, 0.93, 1.01],
    "field_beans": [1.04, 1.00, 0.93, 1.03],
    "fertiliser":  [1.08, 0.97, 0.87, 1.08],  # peak in Q1/Q4 (forward buying)
}

# ── ECB FX for audit / raw CSV (prices already in GBP so FX is informational) ─

FALLBACK_GBP_PER_USD: Dict[int, float] = {
    2000: 0.660, 2001: 0.695, 2002: 0.667, 2003: 0.613,
    2004: 0.546, 2005: 0.550, 2006: 0.543, 2007: 0.500,
    2008: 0.536, 2009: 0.640, 2010: 0.647, 2011: 0.623,
    2012: 0.631, 2013: 0.640, 2014: 0.607, 2015: 0.653,
    2016: 0.740, 2017: 0.776, 2018: 0.750, 2019: 0.784,
    2020: 0.780, 2021: 0.728, 2022: 0.812, 2023: 0.802,
}


def _fetch_ecb_fx() -> Dict[int, float]:
    """Attempt to get GBP/USD from ECB; fall back to hardcoded on any error."""
    ECB = "https://data-api.ecb.europa.eu/service/data/EXR"

    def _parse(raw: dict) -> Dict[int, float]:
        ds = raw["dataSets"][0]["series"]
        years_meta = raw["structure"]["dimensions"]["observation"][0]["values"]
        obs = next(iter(ds.values()))["observations"]
        result: Dict[int, float] = {}
        for idx_str, val_list in obs.items():
            try:
                yr = int(years_meta[int(idx_str)]["id"])
                if val_list and val_list[0] is not None:
                    result[yr] = float(val_list[0])
            except (IndexError, ValueError, TypeError):
                pass
        return result

    try:
        def _get(series: str) -> Dict[int, float]:
            url = f"{ECB}/{series}?format=jsondata&startPeriod=2000&endPeriod=2023"
            req = urllib.request.Request(url, headers={"User-Agent": "uk-arable-manager/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return _parse(json.loads(r.read()))

        usd_per_eur = _get("A.USD.EUR.SP00.A")
        gbp_per_eur = _get("A.GBP.EUR.SP00.A")
        result: Dict[int, float] = {}
        for yr in YEARS:
            u = usd_per_eur.get(yr)
            g = gbp_per_eur.get(yr)
            if u and g and u > 0:
                result[yr] = round(g / u, 5)
        if len(result) >= 20:
            print(f"  ECB FX: fetched {len(result)} years of GBP/USD")
            return result
    except Exception as e:
        print(f"  ECB FX unavailable ({e}), using hardcoded GBP/USD rates")
    return dict(FALLBACK_GBP_PER_USD)


# ── Build annual GBP price table ─────────────────────────────────────────────

def _build_annual(gbp_per_usd: Dict[int, float]) -> Dict[int, Dict[str, float]]:
    annual: Dict[int, Dict[str, float]] = {}
    for yr in YEARS:
        w = REAL_WHEAT[yr]
        annual[yr] = {
            "wheat":         round(w, 2),
            "barley":        round(w * BARLEY_WHEAT_RATIO, 2),
            "osr":           round(REAL_OSR[yr], 2),
            "field_beans":   round(w * FIELDBEANS_WHEAT_RATIO, 2),
            "an_fertiliser": round(REAL_AN[yr], 2),
        }
    return annual


# ── Multiplier computation ────────────────────────────────────────────────────

def _compute_multipliers(annual: Dict[int, Dict]) -> dict:
    commodities = ["wheat", "barley", "osr", "field_beans", "an_fertiliser"]
    means = {c: sum(annual[yr][c] for yr in YEARS) / len(YEARS) for c in commodities}

    quarterly: Dict[str, Dict[str, Dict]] = {}
    for yr in YEARS:
        quarterly[str(yr)] = {}
        for q in range(1, 5):
            quarterly[str(yr)][str(q)] = {
                "wheat_mult":       round(annual[yr]["wheat"]         / means["wheat"]         * SEASONAL["wheat"][q-1],       4),
                "barley_mult":      round(annual[yr]["barley"]        / means["barley"]        * SEASONAL["barley"][q-1],      4),
                "osr_mult":         round(annual[yr]["osr"]           / means["osr"]           * SEASONAL["osr"][q-1],         4),
                "field_beans_mult": round(annual[yr]["field_beans"]   / means["field_beans"]   * SEASONAL["field_beans"][q-1], 4),
                "fertiliser_mult":  round(annual[yr]["an_fertiliser"] / means["an_fertiliser"] * SEASONAL["fertiliser"][q-1],  4),
            }

    return {
        "period_means_gbp_per_tonne": {k: round(v, 2) for k, v in means.items()},
        "source": "AHDB Cereals & Oilseeds / DEFRA Agricultural Price Indices (annual averages)",
        "quarterly": quarterly,
    }


# ── Save outputs ──────────────────────────────────────────────────────────────

def _save_raw_csv(annual: Dict[int, Dict], gbp_per_usd: Dict[int, float]) -> None:
    path = DATA_RAW / "prices_annual_gbp.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["year", "gbp_per_usd", "wheat", "barley", "osr", "field_beans", "an_fertiliser"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for yr in YEARS:
            row = {"year": yr, "gbp_per_usd": round(gbp_per_usd.get(yr, 0), 5)}
            row.update(annual[yr])
            w.writerow(row)
    print(f"  Raw annual CSV → {path}")


def _save_processed_json(data: dict) -> None:
    path = DATA_PROCESSED / "quarterly_prices.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Processed price JSON → {path}")
    m = data["period_means_gbp_per_tonne"]
    print(f"  Period means (£/t): wheat={m['wheat']:.0f}  barley={m['barley']:.0f}  "
          f"osr={m['osr']:.0f}  fert={m['an_fertiliser']:.0f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Building UK agricultural price series (AHDB/DEFRA sourced) …")
    print("  Fetching GBP/USD from ECB …")
    gbp_per_usd = _fetch_ecb_fx()
    annual = _build_annual(gbp_per_usd)
    price_data = _compute_multipliers(annual)
    _save_raw_csv(annual, gbp_per_usd)
    _save_processed_json(price_data)
    print("Price data ready.")


if __name__ == "__main__":
    main()
