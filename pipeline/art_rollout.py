from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TASK_FILES, TOTAL_QUARTERS
from pipeline.config import DEFAULT_OPENREWARD_ENV_ID, DEFAULT_SESSION_BACKEND
from pipeline.farm_session import build_farm_session

if TYPE_CHECKING:
    import art

_INFERENCE_CLIENT_ATTR = "op" "en" "ai_client"


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
        "commit the next quarterly plan. Keep going until the episode finishes."
    )


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


async def rollout(
    model: "art.Model",
    scenario: FarmScenario,
    *,
    session_backend: str = DEFAULT_SESSION_BACKEND,
    openreward_env_id: str = DEFAULT_OPENREWARD_ENV_ID,
    max_tool_calls: int = 160,
    max_completion_tokens: int = 512,
    temperature: float = 0.8,
) -> "art.Trajectory":
    import art

    session = build_farm_session(
        scenario.task_spec,
        session_backend=session_backend,
        openreward_env_id=openreward_env_id,
    )
    await session.open()
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

    total_reward = 0.0
    commit_calls = 0
    total_tool_calls = 0
    invalid_tool_calls = 0
    episode_finished = False

    try:
        while total_tool_calls < max_tool_calls and commit_calls < TOTAL_QUARTERS:
            inference_client = _inference_client(model)
            completion = await inference_client.chat.completions.create(
                model=model.get_inference_name(),
                messages=trajectory.messages(),
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
                break

            for tool_call in tool_calls:
                total_tool_calls += 1
                payload = _safe_json_loads(tool_call.function.arguments)
                result = await session.call_tool(tool_call.function.name, payload)
                if not result.ok:
                    invalid_tool_calls += 1
                    total_reward -= 1.0
                total_reward += result.reward
                trajectory.messages_and_choices.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.text,
                    }
                )
                if tool_call.function.name == "commit_plan":
                    commit_calls += 1
                if result.finished:
                    episode_finished = True
                    break

            if episode_finished or session.episode_metrics()["finished"]:
                break
    finally:
        metrics = session.episode_metrics()
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
            }
        )
        if not metrics["finished"]:
            trajectory.reward -= 2.0
            trajectory.metrics["terminated_early"] = 1
        await session.close()

    return trajectory.finish()
