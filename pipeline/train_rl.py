from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.art_rollout import load_scenarios, rollout
from pipeline.config import MODAL_APP_NAME_PREFIX, RLJobConfig
from pipeline.farm_session import close_hosted_sessions
from pipeline.modal_common import IMAGE, RUNTIME_SECRET, VOLUMES, commit_all_volumes_async, maybe_aclose, modal_result_path, require_local_env

app = modal.App(f"{MODAL_APP_NAME_PREFIX}-rl")


async def _evaluate_validation(model: Any, config: RLJobConfig) -> dict[str, Any]:
    scenarios = load_scenarios(config.validation_split, config.max_validation_tasks)
    trajectories = [
        await rollout(
            model,
            scenario,
            session_backend=config.session_backend,
            openreward_env_id=config.openreward_env_id,
            max_tool_calls=config.max_tool_calls,
            max_completion_tokens=config.max_completion_tokens,
            temperature=config.temperature,
        )
        for scenario in scenarios
    ]
    if not trajectories:
        return {}
    terminal_scores = [float(t.metrics.get("terminal_score", 0.0)) for t in trajectories]
    rewards = [float(t.reward) for t in trajectories]
    bankruptcy_rate = sum(bool(t.metrics.get("ever_bankrupt", False)) for t in trajectories) / len(trajectories)
    return {
        "num_trajectories": len(trajectories),
        "mean_terminal_score": sum(terminal_scores) / len(terminal_scores),
        "mean_total_reward": sum(rewards) / len(rewards),
        "bankruptcy_rate": bankruptcy_rate,
    }


async def _run_rl_async(config: dict[str, Any]) -> dict[str, Any]:
    import art
    from art.local import LocalBackend

    cfg = RLJobConfig(**config)
    loss_fn = cfg.loss_fn.lower()
    if loss_fn not in {"cispo", "ppo"}:
        raise ValueError(f"Unsupported loss_fn={cfg.loss_fn!r}. Expected 'cispo' or 'ppo'.")
    backend = LocalBackend(path=cfg.art_path)
    try:
        model = art.TrainableModel(
            name=cfg.model_name,
            project=cfg.project,
            base_model=cfg.base_model,
            _internal_config=art.dev.InternalModelConfig(
                trainer_gpu_ids=cfg.trainer_gpu_ids,
                inference_gpu_ids=cfg.inference_gpu_ids,
            ),
        )
        await model.register(backend)

        train_scenarios = load_scenarios(cfg.split, cfg.max_train_tasks)
        if not train_scenarios:
            raise RuntimeError(f"No training scenarios found for split={cfg.split!r}")

        rng = random.Random(cfg.seed)
        validation_history: list[dict[str, Any]] = []
        start_step = await model.get_step()

        for offset in range(cfg.train_steps):
            step = start_step + offset
            batch = rng.sample(train_scenarios, k=min(cfg.groups_per_step, len(train_scenarios)))
            groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout(
                            model,
                            scenario,
                            session_backend=cfg.session_backend,
                            openreward_env_id=cfg.openreward_env_id,
                            max_tool_calls=cfg.max_tool_calls,
                            max_completion_tokens=cfg.max_completion_tokens,
                            temperature=cfg.temperature,
                        )
                        for _ in range(cfg.trajectories_per_group)
                    )
                    for scenario in batch
                ),
                pbar_desc=f"train gather step {step}",
                max_exceptions=10,
            )
            result = await backend.train(
                model,
                groups,
                learning_rate=cfg.learning_rate,
                kl_penalty_coef=cfg.kl_penalty_coef,
                ppo=(loss_fn == "ppo"),
                verbose=True,
            )
            await model.log(groups, metrics=result.metrics, step=result.step, split="train")

            if (offset + 1) % cfg.eval_every == 0:
                validation = await _evaluate_validation(model, cfg)
                if validation:
                    validation["step"] = result.step
                    validation_history.append(validation)
                    await model.log(metrics=validation, step=result.step, split="val")

            await commit_all_volumes_async()

        latest_step = await model.get_step()
    finally:
        await close_hosted_sessions()
        await maybe_aclose(backend)

    summary = {
        "start_step": start_step,
        "final_step": latest_step,
        "train_steps_ran": cfg.train_steps,
        "validation_history": validation_history,
    }
    result_path = modal_result_path("training", "rl_result.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2))
    await commit_all_volumes_async()
    return summary


@app.function(
    image=IMAGE,
    gpu="H100:2",
    timeout=60 * 60 * 18,
    secrets=[RUNTIME_SECRET],
    volumes=VOLUMES,
)
def run_rl_remote(config: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(_run_rl_async(config))


@app.local_entrypoint()
def main(
    train_steps: int = 8,
    groups_per_step: int = 4,
    trajectories_per_group: int = 4,
    eval_every: int = 2,
    max_train_tasks: int = 16,
    max_validation_tasks: int = 4,
    session_backend: str = "hosted",
    openreward_env_id: str = "",
    model_name: str = RLJobConfig.model_name,
    project: str = RLJobConfig.project,
) -> None:
    require_local_env(
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "HF_TOKEN",
        "WANDB_API_KEY",
        "OPENREWARD_API_KEY",
    )
    config = RLJobConfig(
        model_name=model_name,
        project=project,
        train_steps=train_steps,
        groups_per_step=groups_per_step,
        trajectories_per_group=trajectories_per_group,
        eval_every=eval_every,
        max_train_tasks=max_train_tasks,
        max_validation_tasks=max_validation_tasks,
        session_backend=session_backend,
        openreward_env_id=openreward_env_id or RLJobConfig().openreward_env_id,
    )
    result = run_rl_remote.remote(config.to_dict())
    print(json.dumps(result, indent=2))
