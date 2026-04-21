# Training Plan

## Environment Design Rationale

### Why This Is a Good RL Benchmark

**Long horizon**: 40 quarters forces genuine long-horizon planning.
Greedy strategies look profitable for years 1–3 but collapse by year 7–10 as
soil health falls below 0.4 and yield modifiers compound.

**Sparse terminal signal**: The terminal_score is only available at the end.
Per-step rewards (scaled P&L) provide shaping but the terminal score is the
true objective.

**Non-trivial trade-offs**:
- High fertiliser: immediate yield boost, long-term soil damage
- Cover crops: immediate cost, long-term soil restoration
- Irrigation: large upfront cost, multi-year dry-weather insurance
- Rotation: reduces income diversity but prevents repeat-crop soil penalty

**Calibration targets** (verified with scripted baselines):
- `greedy_extractor`: high early scores, terminal_score ≈ 0.05–0.25
- `conservative_rotation`: stable scores, terminal_score ≈ 0.30–0.55
- `weather_aware_rotation`: best overall, terminal_score ≈ 0.45–0.70

---

## Task Distribution

### Splits (dry_run scale, 24/8/8)

| Split | n | Purpose |
|-------|---|---------|
| train | 24 | Policy training |
| validation | 8 | Hyperparameter tuning, early stopping |
| test | 8 | Final evaluation |

### Scenario Mix

| Scenario | Fraction | Key characteristic |
|----------|----------|-------------------|
| standard | 50% | Normal starting conditions |
| drought_stressed | 25% | High dry_bias, low initial weather |
| input_cost_shock | 15% | High fertiliser/irrigation costs |
| recovery | 10% | Low starting cash and soil |

### Scaling Up

Run `python scripts/build_tasks.py --scale medium` for 64/16/16 or `--scale full` for 128/32/32.

---

## Training Phases

### Phase 1: Baseline establishment
```
python scripts/build_tasks.py --scale dry_run
python eval/run_baselines.py --split train
python eval/run_baselines.py --split validation
python eval/summarize_results.py --split validation --plot
```

### Phase 2: RFT artifact generation
```
python scripts/build_rft_artifacts.py --grader scalar_final_score
# Inspect artifacts/rft/rft_train.jsonl and artifacts/rft/grader.py
```

### Phase 3: RFT job launch
```
export OPENAI_API_KEY=sk-...
python scripts/build_rft_artifacts.py --upload
# Note the job_id from the output
```

### Phase 4: Post-training eval
```
python app.py &   # ORS server
python scripts/run_rft_eval.py --split validation --fine-tuned-model ft:o4-mini-2025-04-16:...
python eval/summarize_results.py --split validation --plot
```

---

## Reward Signal Design

### Per-step reward
`reward = total_quarterly_pnl × 1e-4`

This keeps per-step rewards in the range [−0.5, +1.5] for typical quarters.

### Terminal reward (added to final step)
`terminal_score = max(0, cash/start_cash) × soil_factor × solvency_gate`

Added to the final commit_plan reward so the model receives a single
unified reward signal at episode end that captures the full objective.

### Why not just terminal reward?
Pure terminal rewards are hard to learn from with sparse signal over 40 steps.
Per-step P&L shaping helps the model build intuition for what actions are
locally profitable before it can plan globally.

---

## Expected Failure Modes

1. **Reward hacking via fallow abuse**: Model goes all-fallow for 40 quarters.
   Result: large negative cash from costs, zero revenue, zero terminal score.
   Fix: the bankruptcy mechanism ensures this fails.

2. **Greedy convergence**: Model learns to maximise per-step reward but ignores
   soil degradation. Result: high early rewards, low terminal score.
   This is fine — it should appear as a sub-optimal policy that RFT should move away from.

3. **Forgetting soil restoration**: Model never plants cover crops or field beans.
   Result: soil collapses after year 6, yield × 0.75 multiplier, cash collapse.
   The terminal score penalises this through the soil_factor.

---

## Evaluation Metrics

| Metric | Definition | Target (post-RFT) |
|--------|-----------|------------------|
| terminal_score | Primary composite score | > 0.50 |
| mean_final_soil | Avg soil health at episode end | > 0.55 |
| bankruptcy_rate | % episodes with ever_bankrupt=True | < 20% |
| cash_ratio | ending_cash / starting_cash | > 1.0 |
| mean_q_reward | Average per-step reward | > 0.05 |
