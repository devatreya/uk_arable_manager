"""
Build quarter-level SFT data from top-quartile scripted baseline trajectories.

This script ranks tasks with the weather-aware baseline, selects the top quantile,
then replays those tasks through the in-process farm environment while recording a
fixed read-tools -> commit_plan workflow suitable for ART SFT warm-starting.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.weather_aware_rotation import policy as weather_aware_policy
from config import TASK_FILES, TOTAL_QUARTERS
from grader import grade
from pipeline.config import (
    DEFAULT_OPENREWARD_ENV_ID,
    DEFAULT_READ_TOOL_SEQUENCE,
    DEFAULT_SESSION_BACKEND,
    DEFAULT_TOP_QUANTILE,
    SFT_OUTPUT_DIR,
    SFT_TRAIN_FILE,
    SFT_VALIDATION_FILE,
)
from pipeline.farm_session import build_farm_session, close_hosted_sessions, format_commit_plan_payload
from rollout_client import run_hosted_rollout, run_local_rollout


def _assistant_tool_call_message(call_id: str, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(payload, sort_keys=True),
                },
            }
        ],
    }


def _tool_result_message(call_id: str, content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
    }


def _load_tasks(split: str, max_tasks: int | None = None) -> list[dict[str, Any]]:
    task_file = TASK_FILES.get(split)
    if not task_file or not Path(task_file).exists():
        raise FileNotFoundError(f"Task file not found for split={split!r}: {task_file}")
    tasks = json.loads(Path(task_file).read_text())
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


async def _rank_tasks(
    split: str,
    *,
    session_backend: str,
    openreward_env_id: str,
    max_tasks: int | None = None,
) -> list[tuple[float, dict[str, Any]]]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    rollout_fn = run_hosted_rollout if session_backend == "hosted" else run_local_rollout
    for task_spec in _load_tasks(split, max_tasks=max_tasks):
        kwargs: dict[str, Any] = {
            "task_spec": task_spec,
            "policy": weather_aware_policy,
            "baseline_name": "weather_aware_rotation",
        }
        if rollout_fn is run_hosted_rollout:
            kwargs["env_id"] = openreward_env_id
            traj = await asyncio.to_thread(rollout_fn, **kwargs)
        else:
            traj = rollout_fn(**kwargs)
        score = grade(traj.to_dict()).score
        ranked.append((score, task_spec))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


async def _build_examples_for_task(
    task_spec: dict[str, Any],
    *,
    session_backend: str,
    openreward_env_id: str,
) -> list[dict[str, Any]]:
    session = build_farm_session(
        task_spec,
        session_backend=session_backend,
        openreward_env_id=openreward_env_id,
    )
    await session.open()
    tools = session.chat_tools()
    prompt = session.prompt_text()
    examples: list[dict[str, Any]] = []

    try:
        for quarter in range(TOTAL_QUARTERS):
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Quarter {quarter + 1}. Inspect the farm using the available tools and "
                        "then commit the next quarterly plan."
                    ),
                },
            ]

            for tool_offset, (tool_name, payload) in enumerate(DEFAULT_READ_TOOL_SEQUENCE, start=1):
                call_id = f"{task_spec['task_id']}_q{quarter + 1}_{tool_offset}"
                messages.append(_assistant_tool_call_message(call_id, tool_name, payload))
                result = await session.call_tool(tool_name, payload)
                messages.append(_tool_result_message(call_id, result.text))

            action = weather_aware_policy(session.state())
            commit_payload = format_commit_plan_payload(action)
            commit_call_id = f"{task_spec['task_id']}_q{quarter + 1}_commit"
            messages.append(_assistant_tool_call_message(commit_call_id, "commit_plan", commit_payload))
            result = await session.call_tool("commit_plan", commit_payload)
            messages.append(_tool_result_message(commit_call_id, result.text))
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Quarter {quarter + 1} plan committed.",
                }
            )

            examples.append({"messages": messages, "tools": tools})
            if result.finished:
                break
    finally:
        await session.close()

    return examples


async def _build_split(
    split: str,
    output_path: Path,
    top_quantile: float,
    *,
    session_backend: str,
    openreward_env_id: str,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    ranked = await _rank_tasks(
        split,
        session_backend=session_backend,
        openreward_env_id=openreward_env_id,
        max_tasks=max_tasks,
    )
    selected_count = max(1, int(len(ranked) * top_quantile))
    selected = ranked[:selected_count]

    examples: list[dict[str, Any]] = []
    for _, task_spec in selected:
        examples.extend(
            await _build_examples_for_task(
                task_spec,
                session_backend=session_backend,
                openreward_env_id=openreward_env_id,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example) + "\n")

    return {
        "split": split,
        "ranked_tasks": len(ranked),
        "selected_count": len(selected),
        "selected_tasks": [
            {"task_id": task["task_id"], "score": score}
            for score, task in selected
        ],
        "num_examples": len(examples),
        "output_path": str(output_path),
    }


async def _main_async(
    top_quantile: float,
    max_tasks_per_split: int | None,
    session_backend: str,
    openreward_env_id: str,
) -> None:
    out_dir = Path(SFT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_summary = await _build_split(
            "train",
            Path(SFT_TRAIN_FILE),
            top_quantile,
            session_backend=session_backend,
            openreward_env_id=openreward_env_id,
            max_tasks=max_tasks_per_split,
        )
        validation_summary = await _build_split(
            "validation",
            Path(SFT_VALIDATION_FILE),
            top_quantile,
            session_backend=session_backend,
            openreward_env_id=openreward_env_id,
            max_tasks=max_tasks_per_split,
        )
    finally:
        await close_hosted_sessions()

    summary = {
        "top_quantile": top_quantile,
        "session_backend": session_backend,
        "openreward_env_id": openreward_env_id,
        "train": train_summary,
        "validation": validation_summary,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nSFT data written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT warm-start data from scripted baselines")
    parser.add_argument("--top-quantile", type=float, default=DEFAULT_TOP_QUANTILE)
    parser.add_argument("--max-tasks-per-split", type=int, default=None)
    parser.add_argument("--session-backend", choices=["hosted", "inprocess"], default=DEFAULT_SESSION_BACKEND)
    parser.add_argument("--openreward-env-id", default=DEFAULT_OPENREWARD_ENV_ID)
    args = parser.parse_args()
    if not 0.0 < args.top_quantile <= 1.0:
        raise ValueError("--top-quantile must be in the range (0, 1].")
    asyncio.run(
        _main_async(
            args.top_quantile,
            args.max_tasks_per_split,
            args.session_backend,
            args.openreward_env_id,
        )
    )


if __name__ == "__main__":
    main()
