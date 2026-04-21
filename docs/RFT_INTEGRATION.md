# OpenAI RFT Integration

## What This Repo Does vs What OpenAI Does

This document is honest about the boundary.

### This repo is responsible for:
1. **ORS environment** — farm simulation, tools, rewards, episode lifecycle
2. **Task generation** — deterministic train/validation/test splits
3. **Episode execution** — local rollouts via `LocalRollout` or ORS server
4. **Trajectory logging** — complete per-step records
5. **Grading** — three grader variants applied to logged trajectories
6. **RFT artifact generation** — training/validation JSONL in OpenAI format
7. **Grader script** — uploaded to OpenAI for reward computation
8. **Job manifest** — `RFTJobManifest.to_api_payload()` is the exact launch payload
9. **Upload** — `rft_bridge.upload_and_create_job()` handles file upload + job creation
10. **Post-training eval** — `scripts/run_rft_eval.py` runs o4-mini against ORS environment

### OpenAI manages:
- Policy gradient updates (the actual fine-tuning)
- Rollout sampling during RFT training
- Job orchestration and GPU compute

---

## Data Flow: ORS Episodes → RFT Artifacts

```
ORS Environment (env.py + sim.py)
         │
         │ episodes run locally via eval/run_baselines.py
         ▼
eval/trajectories/<split>/<baseline>/<task_id>.json
         │
         │ scripts/build_rft_artifacts.py
         ▼
artifacts/rft/rft_train.jsonl       ← uploaded to OpenAI Files API
artifacts/rft/rft_validation.jsonl  ← uploaded to OpenAI Files API
artifacts/rft/grader.py             ← uploaded to OpenAI Files API
         │
         │ rft_bridge.upload_and_create_job()
         ▼
OpenAI Fine-Tuning Job (managed RFT on o4-mini-2025-04-16)
         │
         │ scripts/run_rft_eval.py
         ▼
Post-training eval results in eval/model_trajectories/
```

---

## Training Example Format

Each line of `rft_train.jsonl`:
```json
{
  "messages": [
    {"role": "system",    "content": "...farm management instructions..."},
    {"role": "user",      "content": "Task ID: train_0042. Starting cash: £155,000. Scenario: standard."},
    {"role": "assistant", "content": "[Q1] capital=none\n  plot_1: crop=wheat fert=medium pest=ipm → pnl=£12,450\n  ..."}
  ],
  "metadata": {
    "task_id": "train_0042",
    "terminal_score": 0.6812,
    "normalised_score": 0.6812,
    "ending_cash": 187432.10,
    "mean_final_soil": 0.521,
    "ever_bankrupt": false,
    "quarters_completed": 40,
    "grader_variant": "scalar_final_score",
    "scenario_type": "standard"
  }
}
```

**ASSUMPTION**: OpenAI RFT JSONL uses the same `messages` format as SFT with an
additional `metadata` key that is passed verbatim to the grader at scoring time.
If the actual format differs, update `openai_rft_config.RFTTrainingExample.to_dict()`.

---

## Grader Design

The grader script (`artifacts/rft/grader.py`) is uploaded to OpenAI and called
by the RFT infrastructure to score each model rollout.

```python
def grade(sample) -> float:
    # sample.metadata["terminal_score"] is the ground truth from the simulator
    # Returns float in [0, 1]
    ...
```

Three grader variants are implemented in `grader.py`:

| Variant | Score formula | Use case |
|---------|--------------|----------|
| `scalar_final_score` | `cash_ratio × soil_factor × solvency_gate` | Primary RFT signal |
| `bankruptcy_aware` | Above × completion_ratio, 0 if still negative cash | Harder anti-bankruptcy pressure |
| `stewardship_weighted` | 60% financial + 40% soil | Detecting soil conservation learning |

The default grader for RFT is `scalar_final_score`.

---

## RFT Hyperparameters

Defined in `openai_rft_config.DEFAULT_HYPERPARAMETERS`.

**ASSUMPTION**: The following are correct for `o4-mini-2025-04-16` RFT:
```python
{
    "n_epochs": 1,
    "batch_size": "auto",
    "learning_rate_multiplier": "auto",
    "reasoning_effort": "medium",
}
```

If the API rejects these, update `openai_rft_config.DEFAULT_HYPERPARAMETERS`.
All API syntax assumptions are isolated to `openai_rft_config.py`.

---

## Job Launch

```bash
# 1. Generate task files
python scripts/build_tasks.py --scale medium

# 2. Run baseline trajectories (generates training data)
python eval/run_baselines.py --split train
python eval/run_baselines.py --split validation

# 3. Build RFT artifacts
python scripts/build_rft_artifacts.py --grader scalar_final_score

# 4. Upload + create job (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python scripts/build_rft_artifacts.py --upload

# 5. Dry run to inspect manifest without API calls
python scripts/build_rft_artifacts.py --upload --dry-run
```

---

## Post-Training Evaluation

```bash
# Requires ORS server running + OPENAI_API_KEY
python app.py &                          # start ORS server
python scripts/run_rft_eval.py \
    --split validation \
    --model o4-mini-2025-04-16           # or your fine-tuned model ID

# Summarise and compare
python eval/summarize_results.py --split validation --plot
```

---

## What Cannot Be Done Offline

- **Policy updates**: OpenAI RFT is a managed service.  The gradient step happens
  inside OpenAI's infrastructure.  This repo cannot replicate it.
- **Rollout sampling during training**: OpenAI samples rollouts internally.
  We supply the grader and seed examples; OpenAI handles the training loop.
- **Job monitoring**: Use `openai.fine_tuning.jobs.retrieve(job_id)` or the dashboard.

These limitations are by design — we are not hand-waving them; we are explicitly
scoping the repo to the artifact-generation and evaluation halves of the pipeline.
