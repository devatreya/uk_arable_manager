from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TASK_FILES, TOTAL_QUARTERS
from pipeline.config import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_OPENREWARD_ENV_ID,
    DEFAULT_SESSION_BACKEND,
)
from pipeline.farm_session import build_farm_session
from pipeline.tool_transcript import compact_tool_result

if TYPE_CHECKING:
    import art

_INFERENCE_CLIENT_ATTR = "op" "en" "ai_client"
_TASK_ID_PATTERN = re.compile(r"task_id=(?P<task_id>[^ ]+)")


@dataclass
class FarmScenario:
    task_spec: dict[str, Any]

    @property
    def task_id(self) -> str:
        return str(self.task_spec.get("task_id", "unknown"))

    @property
    def split(self) -> str:
        return str(self.task_spec.get("split", "train"))


def load_scenarios(split: str, max_tasks: int | None = None) -> list[FarmScenario]:
    task_file = TASK_FILES.get(split)
    if not task_file or not Path(task_file).exists():
        raise FileNotFoundError(f"Task file not found for split={split!r}: {task_file}")
    tasks = json.loads(Path(task_file).read_text())
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return [FarmScenario(task_spec=task) for task in tasks]


def _user_message(task_spec: dict[str, Any]) -> str:
    task_id = task_spec.get("task_id", "unknown")
    scenario = task_spec.get("scenario_type", "standard")
    return (
        f"Task {task_id} ({scenario}). Use the farm tools to inspect the state and "
        "commit the next quarterly plan. Keep repeating this quarter-by-quarter until "
        "the episode finishes. Do not stop after a single plan."
    )


def _continuation_user_message(next_quarter: int) -> str:
    return (
        f"Quarter {next_quarter}. The previous plan has been applied. Inspect the updated "
        "farm state, then commit the next quarterly plan. Keep going until the episode finishes."
    )


def _quarter_committed_message(completed_quarter: int) -> str:
    return f"Quarter {completed_quarter} plan committed."


def _safe_json_loads(raw_arguments: str | None) -> dict[str, Any]:
    if not raw_arguments:
        return {}
    try:
        payload: Any = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    for _ in range(2):
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return {}
            continue
        return {}
    return payload if isinstance(payload, dict) else {}


def _inference_client(model: Any) -> Any:
    return getattr(model, _INFERENCE_CLIENT_ATTR)()


def _messages_for_inference(trajectory: Any) -> list[dict[str, Any]]:
    messages = list(trajectory.messages())
    if len(messages) <= 2:
        return messages
    last_user_index = max(
        (
            index
            for index, message in enumerate(messages[1:], start=1)
            if message.get("role") == "user"
        ),
        default=1,
    )
    return [messages[0], *messages[last_user_index:]]


def _default_episode_metrics(task_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "cash": float(task_spec.get("starting_cash", 150_000.0)),
        "starting_cash": float(task_spec.get("starting_cash", 150_000.0)),
        "mean_final_soil": 0.55,
        "quarter": 0,
        "ever_bankrupt": False,
        "finished": False,
        "terminal_score": None,
    }


def _safe_episode_metrics(session: Any, task_spec: dict[str, Any]) -> dict[str, Any]:
    try:
        metrics = session.episode_metrics()
    except Exception:
        return _default_episode_metrics(task_spec)
    return metrics if isinstance(metrics, dict) else _default_episode_metrics(task_spec)


def _task_id_from_exception(exc: BaseException) -> str | None:
    match = _TASK_ID_PATTERN.search(str(exc))
    if match is None:
        return None
    return match.group("task_id")


def summarize_rollout_batch(
    trajectories: list[Any],
    *,
    exceptions: list[BaseException] | None = None,
) -> dict[str, Any]:
    termination_reason_counts = Counter(
        str(trajectory.metrics.get("termination_reason", "unknown"))
        for trajectory in trajectories
    )
    failed_task_ids: list[str] = []
    for exc in exceptions or []:
        task_id = _task_id_from_exception(exc)
        if task_id and task_id not in failed_task_ids:
            failed_task_ids.append(task_id)
    return {
        "num_trajectories": len(trajectories),
        "termination_reason_counts": dict(sorted(termination_reason_counts.items())),
        "tool_budget_exhaustions": int(termination_reason_counts.get("tool_budget_exhausted", 0)),
        "failed_task_ids": failed_task_ids,
    }


def summarize_trajectory_groups(groups: list[Any]) -> dict[str, Any]:
    trajectories: list[Any] = []
    exceptions: list[BaseException] = []
    for group in groups:
        trajectories.extend(list(getattr(group, "trajectories", []) or []))
        exceptions.extend(list(getattr(group, "exceptions", []) or []))
    return summarize_rollout_batch(trajectories, exceptions=exceptions)


def _logging_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, bool))
    }


def trajectory_for_logging(trajectory: Any) -> Any:
    sanitized = trajectory.model_copy(deep=True)
    sanitized.metrics = _logging_metrics(dict(getattr(trajectory, "metrics", {}) or {}))
    return sanitized


def trajectories_for_logging(trajectories: list[Any]) -> list[Any]:
    return [trajectory_for_logging(trajectory) for trajectory in trajectories]


def trajectory_groups_for_logging(groups: list[Any]) -> list[Any]:
    sanitized_groups: list[Any] = []
    for group in groups:
        sanitized_group = group.model_copy(deep=True)
        sanitized_group.trajectories = trajectories_for_logging(
            list(getattr(group, "trajectories", []) or [])
        )
        sanitized_groups.append(sanitized_group)
    return sanitized_groups


async def rollout(
    model: "art.Model",
    scenario: FarmScenario,
    *,
    session_backend: str = DEFAULT_SESSION_BACKEND,
    openreward_env_id: str = DEFAULT_OPENREWARD_ENV_ID,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    temperature: float = 0.8,
) -> "art.Trajectory":
    import art

    session = build_farm_session(
        scenario.task_spec,
        session_backend=session_backend,
        openreward_env_id=openreward_env_id,
    )
    trajectory: Any | None = None
    total_reward = 0.0
    commit_calls = 0
    total_tool_calls = 0
    invalid_tool_calls = 0
    episode_finished = False
    session_opened = False
    last_tool_name: str | None = None
    termination_reason: str | None = None
    rollout_error: tuple[RuntimeError, Exception] | None = None

    try:
        await session.open()
        session_opened = True
        tools = session.chat_tools()
        trajectory = art.Trajectory(
            messages_and_choices=[
                {"role": "system", "content": session.prompt_text()},
                {"role": "user", "content": _user_message(scenario.task_spec)},
            ],
            tools=tools,
            metadata={
                "task_id": scenario.task_id,
                "split": scenario.split,
                "scenario_type": str(scenario.task_spec.get("scenario_type", "standard")),
            },
            reward=0.0,
        )

        while total_tool_calls < max_tool_calls and commit_calls < TOTAL_QUARTERS:
            inference_client = _inference_client(model)
            completion = await inference_client.chat.completions.create(
                model=model.get_inference_name(),
                messages=_messages_for_inference(trajectory),
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                timeout=120,
                logprobs=True,
                top_logprobs=0,
            )
            choice = completion.choices[0]
            trajectory.messages_and_choices.append(choice)

            tool_calls = list(getattr(choice.message, "tool_calls", None) or [])
            if not tool_calls:
                trajectory.metrics["missing_tool_call"] = 1
                total_reward -= 2.0
                termination_reason = "missing_tool_call"
                break

            committed_quarter = False
            for tool_call in tool_calls:
                if total_tool_calls >= max_tool_calls:
                    termination_reason = "tool_budget_exhausted"
                    break
                last_tool_name = tool_call.function.name
                total_tool_calls += 1
                payload = _safe_json_loads(tool_call.function.arguments)
                result = await session.call_tool(tool_call.function.name, payload)
                if not result.ok:
                    invalid_tool_calls += 1
                    total_reward -= 1.0
                compact_text = compact_tool_result(
                    tool_call.function.name,
                    state=session.state(),
                    episode_metrics=session.episode_metrics(),
                    reward=float(result.reward),
                    finished=bool(result.finished),
                    payload=payload,
                )
                total_reward += result.reward
                trajectory.messages_and_choices.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": compact_text,
                    }
                )
                if tool_call.function.name == "commit_plan":
                    commit_calls += 1
                    committed_quarter = True
                if result.finished:
                    episode_finished = True
                    termination_reason = "finished"
                    break

            if termination_reason == "tool_budget_exhausted":
                break
            if episode_finished or session.episode_metrics()["finished"]:
                termination_reason = "finished"
                break
            if committed_quarter:
                completed_quarter = int(session.episode_metrics()["quarter"])
                next_quarter = completed_quarter + 1
                trajectory.messages_and_choices.append(
                    {
                        "role": "assistant",
                        "content": _quarter_committed_message(completed_quarter),
                    }
                )
                trajectory.messages_and_choices.append(
                    {
                        "role": "user",
                        "content": _continuation_user_message(next_quarter),
                    }
                )
        if termination_reason is None:
            metrics = _safe_episode_metrics(session, scenario.task_spec)
            if bool(metrics.get("finished")) or int(metrics.get("quarter", 0)) >= TOTAL_QUARTERS:
                termination_reason = "finished"
            elif total_tool_calls >= max_tool_calls:
                termination_reason = "tool_budget_exhausted"
    except Exception as exc:
        termination_reason = "exception"
        metrics = _safe_episode_metrics(session, scenario.task_spec)
        error = RuntimeError(
            "Rollout failed "
            f"task_id={scenario.task_id} split={scenario.split} "
            f"current_quarter={int(metrics.get('quarter', 0))} "
            f"quarter_commits={commit_calls} tool_calls={total_tool_calls} "
            f"last_tool_name={last_tool_name or 'none'}: "
            f"{type(exc).__name__}: {exc}"
        )
        rollout_error = (error, exc)
    finally:
        metrics = _safe_episode_metrics(session, scenario.task_spec)
        if trajectory is not None:
            trajectory.reward = round(total_reward, 6)
            trajectory.metrics.update(
                {
                    "tool_calls": total_tool_calls,
                    "quarter_commits": commit_calls,
                    "invalid_tool_calls": invalid_tool_calls,
                    "terminal_score": float(metrics["terminal_score"] or 0.0),
                    "ending_cash": float(metrics["cash"]),
                    "mean_final_soil": float(metrics["mean_final_soil"]),
                    "ever_bankrupt": bool(metrics["ever_bankrupt"]),
                    "quarters_completed": int(metrics["quarter"]),
                    "completion_rate": float(metrics["quarter"]) / float(TOTAL_QUARTERS),
                    "finished": bool(metrics["finished"]),
                    "completed_all_quarters": int(metrics["quarter"]) >= TOTAL_QUARTERS,
                    "termination_reason": termination_reason or "finished",
                    "last_tool_name": last_tool_name,
                }
            )
            if not metrics["finished"]:
                trajectory.reward -= 2.0
                trajectory.metrics["terminated_early"] = 1
        if session_opened:
            await session.close()
    if rollout_error is not None:
        error, cause = rollout_error
        raise error from cause
    return trajectory.finish()
