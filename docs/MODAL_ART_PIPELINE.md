# Modal + ART Pipeline

This repo uses OpenPipe ART for both warm-start SFT and online RL, with Modal providing the `H100:2` training runtime.

## Components

- `scripts/prepare_sft_data.py`: hosted baseline replay into JSONL chat/tool traces
- `pipeline/train_sft.py`: supervised warm start on top-quartile baseline traces
- `pipeline/train_rl.py`: online RL over hosted OpenReward farm rollouts
- `pipeline/eval_compare.py`: trained-model evaluation against `weather_aware_rotation`

## Hosted Rollouts

The rehearsal path for the hackathon uses the deployed OpenReward environment for SFT prep, RL, and evaluation. Modal jobs open hosted sessions against `devatreya/uk_arable_manager`, so runtime logs, deployment behavior, and session latency match the real event setup.

Before any expensive run, confirm the hosted environment is live with:

```bash
python scripts/test_deployed_env.py
```

For the full hosted path, use:

```bash
python scripts/prepare_sft_data.py --session-backend hosted --openreward-env-id devatreya/uk_arable_manager
modal run pipeline/train_sft.py
modal run pipeline/train_rl.py --session-backend hosted --openreward-env-id devatreya/uk_arable_manager
modal run pipeline/eval_compare.py --split validation --session-backend hosted --openreward-env-id devatreya/uk_arable_manager
```

## Required Environment Variables

Local machine:

```bash
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
export HF_TOKEN=...
export WANDB_API_KEY=...
export OPENREWARD_API_KEY=...
```

Modal runtime receives:

- `HF_TOKEN`
- `WANDB_API_KEY`
- `OPENREWARD_API_KEY`

## Modal Resources

The training code uses three Modal volumes:

- `uk-arable-art`
- `uk-arable-hf-cache`
- `uk-arable-results`

## Minimal Smoke Sequence

```bash
python scripts/prepare_sft_data.py --top-quantile 0.25 --max-tasks-per-split 4 --session-backend hosted --openreward-env-id devatreya/uk_arable_manager
modal run scripts/modal_hello.py
modal run pipeline/train_sft.py --epochs 1 --batch-size 1
modal run pipeline/train_rl.py --train-steps 1 --groups-per-step 1 --trajectories-per-group 1 --max-train-tasks 1 --max-validation-tasks 1 --eval-every 1 --session-backend hosted --openreward-env-id devatreya/uk_arable_manager
modal run pipeline/eval_compare.py --split validation --max-tasks 1 --session-backend hosted --openreward-env-id devatreya/uk_arable_manager
```

## Result Locations

Inside the Modal results volume:

- `training/sft_result.json`
- `training/rl_result.json`
- `eval/<split>_comparison.json`

## Notes

- SFT data is generated locally from hosted `weather_aware_rotation` trajectories and shipped with the Modal app snapshot.
- RL reward is the environment reward accumulated over the full trajectory.
- Validation during RL is optional and controlled by `--eval-every`.
