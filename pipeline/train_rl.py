from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.art_rollout import (
    load_scenarios,
    rollout,
    summarize_rollout_batch,
    summarize_trajectory_groups,
    trajectory_groups_for_logging,
)
from pipeline.config import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MAX_TOOL_CALLS,
    MODAL_APP_NAME_PREFIX,
    RLJobConfig,
)
from pipeline.farm_session import close_hosted_sessions
from pipeline.modal_common import IMAGE, RUNTIME_SECRET, VOLUMES, commit_all_volumes_async, maybe_aclose, modal_result_path, require_local_env

app = modal.App(f"{MODAL_APP_NAME_PREFIX}-rl")


def _print_rollout_summary(label: str, summary: dict[str, Any]) -> None:
    print(f"{label}: {json.dumps(summary, sort_keys=True)}")


async def _evaluate_validation(model: Any, config: RLJobConfig) -> dict[str, Any]:
    scenarios = load_scenarios(config.validation_split, config.max_validation_tasks)
    trajectories: list[Any] = []
    for scenario in scenarios:
        try:
            trajectories.append(
                await rollout(
                    model,
                    scenario,
                    session_backend=config.session_backend,
                    openreward_env_id=config.openreward_env_id,
                    max_tool_calls=config.max_tool_calls,
                    max_completion_tokens=config.max_completion_tokens,
                    temperature=config.temperature,
                )
            )
        except Exception:
            partial_summary = summarize_rollout_batch(trajectories)
            partial_summary["failed_task_ids"] = partial_summary["failed_task_ids"] + [scenario.task_id]
            _print_rollout_summary("Validation rollout summary before failure", partial_summary)
            raise
    if not trajectories:
        return {}
    rollout_summary = summarize_rollout_batch(trajectories)
    _print_rollout_summary("Validation rollout summary", rollout_summary)
    terminal_scores = [float(t.metrics.get("terminal_score", 0.0)) for t in trajectories]
    rewards = [float(t.reward) for t in trajectories]
    bankruptcy_rate = sum(bool(t.metrics.get("ever_bankrupt", False)) for t in trajectories) / len(trajectories)
    return {
        "num_trajectories": len(trajectories),
        "mean_terminal_score": sum(terminal_scores) / len(terminal_scores),
        "mean_total_reward": sum(rewards) / len(rewards),
        "bankruptcy_rate": bankruptcy_rate,
        "rollout_summary": rollout_summary,
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
        train_rollout_history: list[dict[str, Any]] = []
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
            train_rollout_summary = summarize_trajectory_groups(groups)
            train_rollout_summary["step"] = step
            train_rollout_history.append(dict(train_rollout_summary))
            _print_rollout_summary(f"Train rollout summary step {step}", train_rollout_summary)
            result = await backend.train(
                model,
                groups,
                learning_rate=cfg.learning_rate,
                kl_penalty_coef=cfg.kl_penalty_coef,
                ppo=(loss_fn == "ppo"),
                verbose=True,
            )
            await model.log(
                trajectory_groups_for_logging(groups),
                metrics=result.metrics,
                step=result.step,
                split="train",
            )

            if (offset + 1) % cfg.eval_every == 0:
                validation = await _evaluate_validation(model, cfg)
                if validation:
                    rollout_summary = validation.pop("rollout_summary", None)
                    validation["step"] = result.step
                    if rollout_summary is not None:
                        validation_history.append({**validation, "rollout_summary": rollout_summary})
                    else:
                        validation_history.append(dict(validation))
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
        "train_rollout_history": train_rollout_history,
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
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
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
        max_tool_calls=max_tool_calls,
        max_completion_tokens=max_completion_tokens,
        session_backend=session_backend,
        openreward_env_id=openreward_env_id or RLJobConfig().openreward_env_id,
    )
    result = run_rl_remote.remote(config.to_dict())
    print(json.dumps(result, indent=2))
