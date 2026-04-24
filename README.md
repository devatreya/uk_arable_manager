# uk_arable_manager

A UK arable farm management benchmark built on the [Open Reward Standard (ORS)](https://openrewardstandard.io/), trained with `Qwen/Qwen2.5-7B-Instruct` using OpenPipe ART on Modal.

## What This Is

The agent manages a 400-acre Cambridgeshire farm over 10 crop years (40 quarterly turns). Each quarter it inspects the farm with tools, commits a full-plot plan, and trades off short-term profit against long-term soil health and solvency.

Greedy extraction looks good early and loses late. Durable rotations, restorative crops, and disciplined capital decisions win over the full horizon.

## Architecture

| Layer | File(s) | Description |
|-------|---------|-------------|
| ORS environment | `env.py`, `app.py` | OpenReward environment server |
| Simulator | `sim.py`, `config.py` | Deterministic farm dynamics |
| Baselines | `baselines/`, `eval/` | Scripted agronomy policies and offline evaluation |
| Local rollout | `rollout_client.py`, `trajectory_logger.py` | Deterministic local rollouts and trajectory logs |
| SFT data prep | `scripts/prepare_sft_data.py` | Top-quartile baseline replay into chat/tool traces |
| ART rollouts | `pipeline/farm_session.py`, `pipeline/art_rollout.py` | In-process farm sessions for model rollouts |
| Modal training | `pipeline/train_sft.py`, `pipeline/train_rl.py` | Qwen SFT and GRPO on `H100:2` |
| Model eval | `pipeline/eval_compare.py` | Compare trained model against scripted baselines |

## Farm Interface

Environment name: `UKArableManager`

Tools:

| Tool | Advances time? | Returns reward? |
|------|---------------|----------------|
| `read_farm_state` | No | No |
| `read_soil_report` | No | No |
| `read_weather_history` | No | No |
| `read_price_board` | No | No |
| `commit_plan` | Yes | Yes |

Splits: `train`, `validation`, `test`

## Training Stack

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- RL stack: `openpipe-art`
- Remote training: Modal `H100:2`
- Tracking: Weights & Biases
- Hosted env smoke test: OpenReward

## Quick Start

### 1. Create a Python 3.12 environment

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

### 2. Build tasks and baselines

```bash
python scripts/fetch_weather.py
python scripts/fetch_prices.py
python scripts/build_tasks.py --scale medium
python eval/run_baselines.py --split train
python eval/run_baselines.py --split validation
```

### 3. Build the SFT warm-start dataset

```bash
python scripts/prepare_sft_data.py
```

For a cheap smoke:

```bash
python scripts/prepare_sft_data.py --top-quantile 0.25 --max-tasks-per-split 4
```

### 4. Configure training credentials

Export these locally before running any Modal job:

```bash
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
export HF_TOKEN=...
export WANDB_API_KEY=...
export OPENREWARD_API_KEY=...
```

### 5. Verify Modal GPU provisioning

```bash
modal run scripts/modal_hello.py
```

### 6. Run SFT on Modal

```bash
modal run pipeline/train_sft.py
```

### 7. Run RL on Modal

```bash
modal run pipeline/train_rl.py
```

### 8. Compare the trained model against baselines

```bash
modal run pipeline/eval_compare.py --split validation --max-tasks 8
```

## OpenReward Smoke Test

The deployed environment can be checked independently:

```bash
python scripts/test_deployed_env.py
```

## Reward Summary

- Dense reward: quarterly P&L scaled by `1e-4`
- Dense shaping: soil preservation bonus per quarter
- Terminal reward: `cash_ratio × soil_factor × solvency_gate`
- Completion bonus: awarded only for reaching quarter 40 without bankruptcy

## Project Structure

```text
uk_arable_manager/
├── app.py
├── env.py
├── sim.py
├── config.py
├── grader.py
├── rollout_client.py
├── trajectory_logger.py
├── pipeline/
│   ├── art_rollout.py
│   ├── config.py
│   ├── eval_compare.py
│   ├── farm_session.py
│   ├── modal_common.py
│   ├── train_rl.py
│   └── train_sft.py
├── scripts/
│   ├── prepare_sft_data.py
│   ├── modal_hello.py
│   ├── build_tasks.py
│   └── test_deployed_env.py
├── eval/
├── baselines/
├── tests/
└── docs/
```

## Docs

- `docs/TRAINING_PLAN.md`
- `docs/MODAL_ART_PIPELINE.md`
- `docs/DATA_SOURCES.md`
