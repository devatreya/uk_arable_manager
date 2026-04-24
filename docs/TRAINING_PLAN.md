# Training Plan

## Objective

Train `Qwen/Qwen2.5-7B-Instruct` to beat the scripted `weather_aware_rotation` baseline on the UK arable farm benchmark while preserving soil health.

## Phase 1: Generate Tasks

```bash
python scripts/build_tasks.py --scale medium
```

Current medium scale:

| Split | Tasks |
|-------|-------|
| train | 64 |
| validation | 16 |
| test | 16 |

## Phase 2: Establish Baselines

```bash
python eval/run_baselines.py --split train
python eval/run_baselines.py --split validation
python eval/run_baselines.py --split test
```

Track these headline metrics:

- mean terminal cash
- mean final soil health
- bankruptcy rate
- completion rate
- mean total episode reward
- mean terminal score

## Phase 3: Build SFT Warm-Start Data

```bash
python scripts/prepare_sft_data.py
```

This script:

1. ranks tasks with `weather_aware_rotation`
2. keeps the top quantile
3. replays them through the in-process farm environment
4. writes chat/tool traces to `artifacts/sft/train.jsonl` and `artifacts/sft/validation.jsonl`

Cheap smoke:

```bash
python scripts/prepare_sft_data.py --top-quantile 0.25 --max-tasks-per-split 4
```

## Phase 4: Modal Preflight

Export credentials:

```bash
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
export HF_TOKEN=...
export WANDB_API_KEY=...
export OPENREWARD_API_KEY=...
```

Verify GPU provisioning:

```bash
modal run scripts/modal_hello.py
```

## Phase 5: SFT on `H100:2`

```bash
modal run pipeline/train_sft.py
```

Useful overrides:

```bash
modal run pipeline/train_sft.py --epochs 1 --batch-size 2
modal run pipeline/train_sft.py --dataset-path artifacts/sft/train.jsonl
```

Result summary is written to the Modal results volume under `training/sft_result.json`.

## Phase 6: RL on `H100:2`

```bash
modal run pipeline/train_rl.py
```

Cheap RL smoke:

```bash
modal run pipeline/train_rl.py \
  --train-steps 1 \
  --groups-per-step 1 \
  --trajectories-per-group 1 \
  --max-train-tasks 1 \
  --max-validation-tasks 1 \
  --eval-every 1
```

RL summary is written to the Modal results volume under `training/rl_result.json`.

## Phase 7: Evaluation

```bash
modal run pipeline/eval_compare.py --split validation --max-tasks 8
modal run pipeline/eval_compare.py --split test --max-tasks 16
```

Comparison summaries are written to the Modal results volume under `eval/`.

## Success Criteria

- beats or matches `weather_aware_rotation` on mean terminal score
- maintains mean final soil health at or above the baseline
- keeps bankruptcy rate acceptably low
- reliably reaches quarter 40 on held-out tasks
