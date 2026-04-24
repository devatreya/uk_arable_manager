# Modal + ART Pipeline

This repo uses OpenPipe ART for both warm-start SFT and online RL, with Modal providing the `H100:2` training runtime.

## Components

- `scripts/prepare_sft_data.py`: baseline replay into JSONL chat/tool traces
- `pipeline/train_sft.py`: supervised warm start on top-quartile baseline traces
- `pipeline/train_rl.py`: online RL over in-process farm rollouts
- `pipeline/eval_compare.py`: trained-model evaluation against scripted baselines

## Why In-Process Rollouts

The RL pipeline does not need the hosted OpenReward deployment path. It opens the farm environment directly in-process through `UKArableManager`, which avoids network overhead and deployment routing issues during training.

The deployed OpenReward env is still useful for smoke-testing serving with:

```bash
python scripts/test_deployed_env.py
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
python scripts/prepare_sft_data.py --top-quantile 0.25 --max-tasks-per-split 4
modal run scripts/modal_hello.py
modal run pipeline/train_sft.py --epochs 1 --batch-size 1
modal run pipeline/train_rl.py --train-steps 1 --groups-per-step 1 --trajectories-per-group 1 --max-train-tasks 1 --max-validation-tasks 1 --eval-every 1
modal run pipeline/eval_compare.py --split validation --max-tasks 1
```

## Result Locations

Inside the Modal results volume:

- `training/sft_result.json`
- `training/rl_result.json`
- `eval/<split>_comparison.json`

## Notes

- SFT data is generated locally and shipped with the Modal app snapshot.
- RL reward is the environment reward accumulated over the full trajectory.
- Validation during RL is optional and controlled by `--eval-every`.
