# Data Sources

## Overview

The simulator uses frozen local data snapshots so the environment works offline at runtime.
No live network calls are made during episode execution.

---

## Weather Data

### Target source
- **Met Office MIDAS-Open** — Cambridge NIAB station (approx. 52.24°N, 0.10°E)
- Archive: https://data.ceda.ac.uk/badc/ukmo-midas-open/
- Requires CEDA account for download

### Local climatology normals
- **Met Office 1991–2020 UK Climate Normals**
- Cambridge NIAB profile embedded in `config.LOCAL_NORMALS`:
  - Mean annual rainfall: 565 mm
  - Mean summer temp: 17.2°C
  - Mean winter temp: 4.1°C
  - Frost days/year: 42

### Current status
- `data/raw/cambridge_niab_monthly.csv` — **synthetic placeholder** calibrated to above normals
- `data/raw/cambridge_niab_normals.csv` — **synthetic placeholder**
- Real download: `python scripts/fetch_weather.py --real` (not yet implemented)
- Run `python scripts/fetch_weather.py` to generate placeholder files

### Regional profile reference
- Defra East of England Agricultural Statistics: https://www.gov.uk/government/collections/agriculture-in-the-english-regions
- Cambridgeshire farming profile: predominantly arable, large field sizes, sandy/chalky loam soils

---

## Crop Price Data

### Target sources
- **AHDB Cereals & Oilseeds ex-farm prices**:
  https://ahdb.org.uk/cereals-oilseeds/market-information/prices
- **AHDB Fertiliser prices** (UK ammonium nitrate 34.5%):
  https://ahdb.org.uk/arable/nitrogen-fertiliser-prices
- **Defra Agricultural Price Indices**:
  https://www.gov.uk/government/collections/agricultural-price-indices

### Calibration basis (economic assumptions)
Gross revenue per acre is derived from:
  - Wheat:        ~200 t/ha yield × £180/t ≈ £700/acre (approximate 2018-2022 average)
  - Barley:       ~175 t/ha × £160/t ≈ £620/acre
  - Oilseed rape: ~140 t/ha × £360/t ≈ £760/acre
  - Field beans:  ~155 t/ha × £220/t ≈ £540/acre

Note: t/ha figures converted to t/acre (× 0.405).  These are indicative benchmarks
for a Cambridgeshire arable farm, not precise current-market values.

### Fertiliser cost calibration
- AN 34.5% typical price range: £250–£600/t depending on year
- Low fertiliser (20 kg N/ha equivalent): ~£20/acre
- Medium (100 kg N/ha): ~£45/acre
- High (160 kg N/ha): ~£75/acre

### Current status
- `data/raw/wheat_exfarm.csv` — **synthetic placeholder**
- `data/raw/barley_exfarm.csv` — **synthetic placeholder**
- `data/raw/osr_delivered.csv` — **synthetic placeholder**
- `data/raw/fertiliser_weekly.csv` — **synthetic placeholder**
- `data/raw/field_beans_anchor.json` — **synthetic placeholder**
- Real download: `python scripts/fetch_prices.py --real` (not yet implemented)

---

## Soil Data

No external soil data files are used at runtime.  Soil sub-component parameters
are derived from:
- **ADAS & AHDB Soil Nutrient Management for Arable Crops** guidance
- Typical East Anglian loam soil starting conditions (organic matter ~2.5–3.5%)

---

## Crop Statistics Reference

Used for calibration but not loaded at runtime:
- **Defra June Agricultural Survey** (England crop area and yield):
  https://www.gov.uk/government/collections/agricultural-and-horticultural-census
- East of England accounts for ~25% of English wheat and barley area

---

## Updating to Real Data

1. Obtain CEDA account → download MIDAS-Open for Cambridge NIAB
2. Implement `scripts/fetch_weather.py --real`
3. Obtain AHDB price CSVs (publicly available or via AHDB Datum subscription)
4. Implement `scripts/fetch_prices.py --real`
5. Run `python scripts/build_quarterly_weather.py` and `python scripts/build_price_board.py`
6. The `FarmSimulator` will pick up the updated files automatically via `config.TASK_FILES`

The environment is designed so that switching from synthetic to real data
requires only updating the raw files — the simulator logic and ORS interface
remain unchanged.
