# uk_arable_manager

A UK arable farm management benchmark built on the [Open Reward Standard (ORS)](https://openrewardstandard.io/).

**Target model**: `o4-mini-2025-04-16` via OpenAI Reinforcement Fine-Tuning (RFT).

---

## What This Is

A General Reasoning RL environment where the agent manages a 400-acre Cambridgeshire arable farm over 10 crop years (40 quarterly turns). The agent must balance short-term profit against long-term soil stewardship and solvency.

**Core tension**: greedy extraction looks good early and fails later. Sustainable rotation wins over the full 40-quarter horizon.

---

## Architecture

| Layer | File(s) | Description |
|-------|---------|-------------|
| ORS environment | `env.py`, `app.py` | ORS-compliant environment server |
| Simulator | `sim.py`, `config.py` | Deterministic farm simulator |
| Grader | `grader.py` | Three grader variants |
| Trajectory logging | `trajectory_logger.py` | Full episode recording |
| RFT bridge | `rft_bridge.py`, `openai_rft_config.py` | ORS → OpenAI RFT artifact pipeline |
| Rollout client | `rollout_client.py` | Local and server-based rollouts |
| Baselines | `baselines/` | Three scripted policies |
| Eval | `eval/` | Baseline runner and result summariser |
| Scripts | `scripts/` | Data prep, task gen, RFT pipeline |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data stubs and tasks
```bash
python scripts/fetch_weather.py
python scripts/fetch_prices.py
python scripts/build_tasks.py          # dry_run: 24/8/8 tasks
```

### 3. Run tests
```bash
python -m pytest tests/ -v
```

### 4. Run scripted baselines
```bash
python eval/run_baselines.py --split validation
python eval/summarize_results.py --split validation
```

### 5. Start ORS server
```bash
python app.py
# Server runs on http://localhost:8080
```

---

## ORS Environment

**Environment name**: `UKArableManager`

**Tools**:
| Tool | Advances time? | Returns reward? |
|------|---------------|----------------|
| `read_farm_state` | No | No |
| `read_soil_report` | No | No |
| `read_weather_history` | No | No |
| `read_price_board` | No | No |
| `commit_plan` | **Yes (+1 quarter)** | **Yes** |

**Episode structure**:
- 1 episode = 1 task = 40 quarters
- Each quarter: read state → commit plan → receive reward
- `finished=True` when quarter 40 completes or bankruptcy threshold crossed

**Splits**: `train`, `validation`, `test`

---

## Farm Specification

| Property | Value |
|----------|-------|
| Location | Cambridgeshire / East Anglia |
| Total area | 400 acres |
| Plots | 4 × 100 acres |
| Horizon | 40 quarters (10 years) |
| Crops | wheat, barley, oilseed_rape, field_beans, cover_crop, fallow |

### Reward

- **Per-step**: `scaled_pnl = quarterly_p&l × 1e-4`
- **Terminal** (added to final step):
  ```
  terminal_score = max(0, ending_cash / starting_cash)
                 × clip(mean_final_soil, 0.4, 1.2)   [normalised to 0-1]
                 × solvency_gate                       [1.0 or 0.2]
  ```

---

## OpenAI RFT Pipeline

### What this repo provides
1. Task generation → deterministic train/val/test splits
2. Episode trajectories → logged by scripted baselines or model rollouts
3. RFT JSONL artifacts → `artifacts/rft/rft_{split}.jsonl`
4. Grader script → `artifacts/rft/grader.py`
5. Job manifest → `RFTJobManifest.to_api_payload()`
6. Upload + job creation → `rft_bridge.upload_and_create_job()`

### Full pipeline
```bash
# Generate artifacts
python scripts/build_rft_artifacts.py

# Upload to OpenAI and create RFT job
export OPENAI_API_KEY=sk-...
python scripts/build_rft_artifacts.py --upload

# Post-training eval
python app.py &
python scripts/run_rft_eval.py --split validation --fine-tuned-model ft:o4-mini:...
```

**See `docs/RFT_INTEGRATION.md` for full details and honest boundary documentation.**

---

## Scripted Baselines

| Baseline | Strategy | Expected terminal_score |
|----------|----------|------------------------|
| `greedy_extractor` | Always highest-margin crop, high fert, spray | 0.05–0.25 |
| `conservative_rotation` | Fixed 4-year rotation, medium inputs | 0.30–0.55 |
| `weather_aware_rotation` | Rotation + weather-adaptive choices | 0.45–0.70 |

---

## Honest Scope

**Implemented in this repo**:
- Full ORS environment (farm sim + tools + reward)
- Train/validation/test task generation
- Scripted baselines with end-to-end trajectories
- Grader logic (3 variants, tested)
- OpenAI RFT artifact generation (JSONL + grader script + manifest)
- Upload + job creation (requires OPENAI_API_KEY)
- Post-training evaluation harness

**Handled by OpenAI**:
- Policy gradient updates (the actual RL training)
- Rollout sampling during training
- GPU compute and job orchestration

**Not yet implemented**:
- Real weather/price data download (stubs provided, see `docs/DATA_SOURCES.md`)
- Live AHDB price integration

---

## Project Structure

```
uk_arable_manager/
├── app.py                    # ORS server entry point
├── env.py                    # ORS environment (tools, splits, prompt)
├── sim.py                    # Pure-Python deterministic simulator
├── config.py                 # All constants and parameters
├── grader.py                 # Three grader variants
├── trajectory_logger.py      # Episode recording
├── rft_bridge.py             # ORS → OpenAI RFT artifact pipeline
├── openai_rft_config.py      # Isolated OpenAI API assumptions
├── rollout_client.py         # Local and HTTP rollout clients
├── data/
│   ├── raw/                  # Stub weather and price CSVs
│   └── processed/            # Generated task JSON files
├── scripts/
│   ├── build_tasks.py        # Generate task splits
│   ├── build_rft_artifacts.py# Build RFT JSONL + grader
│   ├── run_rft_eval.py       # Post-training model evaluation
│   ├── fetch_weather.py      # Weather data stubs
│   ├── fetch_prices.py       # Price data stubs
│   ├── build_quarterly_weather.py
│   └── build_price_board.py
├── baselines/
│   ├── greedy_extractor.py
│   ├── conservative_rotation.py
│   └── weather_aware_rotation.py
├── eval/
│   ├── run_baselines.py      # Run all baselines on a split
│   └── summarize_results.py  # Compare and plot results
├── tests/
│   ├── test_soil.py
│   ├── test_reward.py
│   ├── test_episode.py
│   ├── test_task_generation.py
│   └── test_grader.py
├── docs/
│   ├── DATA_SOURCES.md
│   ├── TRAINING_PLAN.md
│   └── RFT_INTEGRATION.md
└── requirements.txt
```
