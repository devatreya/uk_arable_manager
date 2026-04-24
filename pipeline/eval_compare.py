from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines import BASELINES
from grader import grade
from pipeline.art_rollout import load_scenarios, rollout
from pipeline.config import EvalJobConfig, MODAL_APP_NAME_PREFIX
from pipeline.farm_session import close_hosted_sessions
from pipeline.modal_common import IMAGE, RUNTIME_SECRET, VOLUMES, commit_all_volumes_async, maybe_aclose, modal_result_path, require_local_env
from rollout_client import HostedRolloutManager, run_local_rollout

app = modal.App(f"{MODAL_APP_NAME_PREFIX}-eval")


async def _summarize_baseline(
    split: str,
    task_specs: list[dict[str, Any]],
    baseline_name: str,
    *,
    session_backend: str,
    openreward_env_id: str,
) -> dict[str, Any]:
    policy = BASELINES[baseline_name]
    trajectories = []
    if session_backend == "hosted":
        async with HostedRolloutManager(env_id=openreward_env_id) as manager:
            for task in task_specs:
                trajectories.append(
                    await manager.run(
                        task,
                        policy,
                        baseline_name=baseline_name,
                    )
                )
    else:
        trajectories = [
            run_local_rollout(task_spec=task, policy=policy, baseline_name=baseline_name)
            for task in task_specs
        ]
    grades = [grade(traj.to_dict()) for traj in trajectories]
    return {
        "policy": baseline_name,
        "mean_terminal_cash": sum(float(traj.final_state.get("cash", 0.0)) for traj in trajectories) / len(trajectories),
        "mean_final_soil": sum(float(traj.mean_final_soil) for traj in trajectories) / len(trajectories),
        "bankruptcy_rate": sum(bool(traj.ever_bankrupt) for traj in trajectories) / len(trajectories),
        "completion_rate": sum(int(traj.quarters_completed >= 40) for traj in trajectories) / len(trajectories),
        "mean_total_episode_reward": sum(float(traj.total_reward) for traj in trajectories) / len(trajectories),
        "mean_terminal_score": sum(float(g.score) for g in grades) / len(grades),
    }


def _summarize_model(policy_name: str, trajectories: list[Any]) -> dict[str, Any]:
    return {
        "policy": policy_name,
        "mean_terminal_cash": sum(float(t.metrics.get("ending_cash", 0.0)) for t in trajectories) / len(trajectories),
        "mean_final_soil": sum(float(t.metrics.get("mean_final_soil", 0.0)) for t in trajectories) / len(trajectories),
        "bankruptcy_rate": sum(bool(t.metrics.get("ever_bankrupt", False)) for t in trajectories) / len(trajectories),
        "completion_rate": sum(int(t.metrics.get("quarters_completed", 0) >= 40) for t in trajectories) / len(trajectories),
        "mean_total_episode_reward": sum(float(t.reward) for t in trajectories) / len(trajectories),
        "mean_terminal_score": sum(float(t.metrics.get("terminal_score", 0.0)) for t in trajectories) / len(trajectories),
    }


async def _run_eval_async(config: dict[str, Any]) -> dict[str, Any]:
    import art
    from art.local import LocalBackend

    cfg = EvalJobConfig(**config)
    scenarios = load_scenarios(cfg.split, cfg.max_tasks)
    task_specs = [scenario.task_spec for scenario in scenarios]
    if not task_specs:
        raise RuntimeError(f"No tasks found for split={cfg.split!r}")

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
                "Running eval with dedicated ART GPUs:",
                f"trainer={cfg.trainer_gpu_ids}",
                f"inference={cfg.inference_gpu_ids}",
                f"gpu_memory_utilization={cfg.engine_gpu_memory_utilization}",
            )
        else:
            print(
                "Running eval in shared ART mode inside the H100:2 container.",
                f"gpu_memory_utilization={cfg.engine_gpu_memory_utilization}",
            )
        model = art.TrainableModel(
            name=cfg.model_name,
            project=cfg.project,
            base_model=cfg.base_model,
            _internal_config=internal_config,
        )
        await model.register(backend)

        model_trajectories = [
            await rollout(
                model,
                scenario,
                session_backend=cfg.session_backend,
                openreward_env_id=cfg.openreward_env_id,
                max_tool_calls=cfg.max_tool_calls,
                max_completion_tokens=cfg.max_completion_tokens,
                temperature=cfg.temperature,
            )
            for scenario in scenarios
        ]
        await model.log(model_trajectories, split=f"eval_{cfg.split}")
    finally:
        await close_hosted_sessions()
        await maybe_aclose(backend)

    baseline_summaries = [
        await _summarize_baseline(
            cfg.split,
            task_specs,
            "weather_aware_rotation",
            session_backend=cfg.session_backend,
            openreward_env_id=cfg.openreward_env_id,
        )
    ]

    summary = {
        "split": cfg.split,
        "num_tasks": len(task_specs),
        "agents": baseline_summaries + [_summarize_model(cfg.model_name, model_trajectories)],
    }

    out_path = modal_result_path("eval", f"{cfg.split}_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    await commit_all_volumes_async()
    return summary


@app.function(
    image=IMAGE,
    gpu="H100:2",
    timeout=60 * 60 * 12,
    secrets=[RUNTIME_SECRET],
    volumes=VOLUMES,
)
def run_eval_remote(config: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(_run_eval_async(config))


@app.local_entrypoint()
def main(
    split: str = "validation",
    max_tasks: int = 4,
    session_backend: str = "hosted",
    openreward_env_id: str = "",
    model_name: str = EvalJobConfig.model_name,
    project: str = EvalJobConfig.project,
) -> None:
    require_local_env(
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "HF_TOKEN",
        "WANDB_API_KEY",
        "OPENREWARD_API_KEY",
    )
    config = EvalJobConfig(
        model_name=model_name,
        project=project,
        split=split,
        max_tasks=max_tasks,
        session_backend=session_backend,
        openreward_env_id=openreward_env_id or EvalJobConfig().openreward_env_id,
    )
    result = run_eval_remote.remote(config.to_dict())
    print(json.dumps(result, indent=2))
