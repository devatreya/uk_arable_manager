from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import MODAL_APP_NAME_PREFIX, SFT_OUTPUT_DIR, SFTJobConfig
from pipeline.modal_common import IMAGE, RUNTIME_SECRET, VOLUMES, commit_all_volumes_async, maybe_aclose, modal_result_path, require_local_env

app = modal.App(f"{MODAL_APP_NAME_PREFIX}-sft")


async def _run_sft_async(config: dict[str, Any]) -> dict[str, Any]:
    import art
    from art.local import LocalBackend
    from art.utils.sft import train_sft_from_file

    cfg = SFTJobConfig(**config)
    dataset_path = Path(cfg.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"SFT dataset not found: {dataset_path}")

    backend = LocalBackend(path=cfg.art_path)
    try:
        internal_config = art.dev.InternalModelConfig(
            engine_args={
                "gpu_memory_utilization": cfg.engine_gpu_memory_utilization,
            }
        )
        if cfg.trainer_gpu_ids is not None or cfg.inference_gpu_ids is not None:
            if cfg.trainer_gpu_ids is None or cfg.inference_gpu_ids is None:
                raise ValueError(
                    "trainer_gpu_ids and inference_gpu_ids must both be set or both be unset."
                )
            internal_config["trainer_gpu_ids"] = cfg.trainer_gpu_ids
            internal_config["inference_gpu_ids"] = cfg.inference_gpu_ids
            print(
                "Running SFT with dedicated ART GPUs:",
                f"trainer={cfg.trainer_gpu_ids}",
                f"inference={cfg.inference_gpu_ids}",
                f"gpu_memory_utilization={cfg.engine_gpu_memory_utilization}",
            )
        else:
            print(
                "Running SFT in shared ART mode inside the H100:2 container "
                "(ART does not support train_sft in dedicated mode).",
                f"gpu_memory_utilization={cfg.engine_gpu_memory_utilization}",
            )

        model = art.TrainableModel(
            name=cfg.model_name,
            project=cfg.project,
            base_model=cfg.base_model,
            _internal_config=internal_config,
        )

        print(f"Registering model {cfg.model_name} from {cfg.base_model}")
        await model.register(backend)
        print(f"Starting SFT from {dataset_path}")
        await train_sft_from_file(
            model=model,
            file_path=str(dataset_path),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            peak_lr=cfg.peak_lr,
            warmup_ratio=cfg.warmup_ratio,
            schedule_type=cfg.schedule_type,
            verbose=True,
        )
        step = await model.get_step()
        print(f"SFT finished at step {step}")
    finally:
        await maybe_aclose(backend)

    result = {
        "dataset_path": str(dataset_path),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "step": step,
        "art_path": cfg.art_path,
        "inference_name": cfg.model_name,
    }
    result_path = modal_result_path("training", "sft_result.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    await commit_all_volumes_async()
    return result


@app.function(
    image=IMAGE,
    gpu="H100:2",
    timeout=60 * 60 * 12,
    secrets=[RUNTIME_SECRET],
    volumes=VOLUMES,
)
def run_sft_remote(config: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(_run_sft_async(config))


@app.local_entrypoint()
def main(
    dataset_path: str = f"{SFT_OUTPUT_DIR}/train.jsonl",
    epochs: int = 1,
    batch_size: int = 4,
    peak_lr: float = 2e-4,
    warmup_ratio: float = 0.1,
) -> None:
    require_local_env("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN", "WANDB_API_KEY")
    config = SFTJobConfig(
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        peak_lr=peak_lr,
        warmup_ratio=warmup_ratio,
    )
    result = run_sft_remote.remote(config.to_dict())
    print(json.dumps(result, indent=2))
