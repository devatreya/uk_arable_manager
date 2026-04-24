from __future__ import annotations

import asyncio
import os
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BANKRUPTCY_HARD_THRESHOLD, NUM_PLOTS, TOTAL_QUARTERS
from env import UKArableManager

_HOSTED_CLIENT: Any | None = None
_HOSTED_ENVIRONMENTS: dict[str, Any] = {}
_HOSTED_TASKS: dict[tuple[str, str], list[Any]] = {}
_HOSTED_TOOL_SPECS: dict[str, list[Any]] = {}
_HOSTED_LOCK: asyncio.Lock | None = None


def _blocks_to_text(blocks: list[Any]) -> str:
    lines: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if text:
            lines.append(text)
    return "\n".join(lines)


def _commit_plan_payload(action: dict[str, Any]) -> dict[str, Any]:
    plots = action.get("plots", [])
    payload: dict[str, Any] = {"capital_action": action.get("capital_action", "none")}
    for index in range(NUM_PLOTS):
        plan = plots[index] if index < len(plots) else {}
        payload[f"plot_{index + 1}"] = {
            "crop": plan.get("crop", "fallow"),
            "fertiliser": plan.get("fertiliser", "medium"),
            "pest_control": plan.get("pest_control", "none"),
        }
    return payload


def _configure_ssl_cert_file() -> None:
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def _tool_specs_to_chat_tools(specs: list[Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for spec in specs:
        input_schema = getattr(spec, "input_schema", None)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": input_schema or {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
        )
    return tools


def _episode_metrics_from_state(state: dict[str, Any], terminal_score: float | None = None) -> dict[str, Any]:
    plots = state.get("plots", [])
    mean_final_soil = (
        sum(
            0.45 * float(plot.get("_organic_matter", 0.55))
            + 0.20 * float(plot.get("_structure", 0.55))
            + 0.15 * float(plot.get("_ph", 0.55))
            + 0.20 * float(plot.get("_nutrient_balance", 0.55))
            for plot in plots
        ) / len(plots)
        if plots
        else 0.55
    )
    cash = float(state.get("cash", 0.0))
    quarter = int(state.get("quarter", 0))
    finished = quarter >= TOTAL_QUARTERS or cash < BANKRUPTCY_HARD_THRESHOLD
    return {
        "cash": cash,
        "starting_cash": float(state.get("starting_cash", 150_000.0)),
        "mean_final_soil": float(mean_final_soil),
        "quarter": quarter,
        "ever_bankrupt": bool(state.get("ever_bankrupt", False)),
        "finished": finished,
        "terminal_score": terminal_score,
    }


async def _get_hosted_client() -> Any:
    global _HOSTED_CLIENT, _HOSTED_LOCK
    if _HOSTED_LOCK is None:
        _HOSTED_LOCK = asyncio.Lock()
    async with _HOSTED_LOCK:
        if _HOSTED_CLIENT is None:
            api_key = os.environ.get("OPENREWARD_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENREWARD_API_KEY must be set for hosted OpenReward sessions.")
            _configure_ssl_cert_file()
            from openreward import AsyncOpenReward

            _HOSTED_CLIENT = AsyncOpenReward(api_key=api_key)
        return _HOSTED_CLIENT


async def _get_hosted_environment(env_id: str) -> Any:
    global _HOSTED_LOCK
    environment = _HOSTED_ENVIRONMENTS.get(env_id)
    if environment is not None:
        return environment
    client = await _get_hosted_client()
    if _HOSTED_LOCK is None:
        _HOSTED_LOCK = asyncio.Lock()
    async with _HOSTED_LOCK:
        environment = _HOSTED_ENVIRONMENTS.get(env_id)
        if environment is not None:
            return environment
        environment = client.environments.get(name=env_id)
        _HOSTED_ENVIRONMENTS[env_id] = environment
        return environment


async def _list_hosted_tasks(env_id: str, split: str) -> list[Any]:
    cache_key = (env_id, split)
    tasks = _HOSTED_TASKS.get(cache_key)
    if tasks is not None:
        return tasks
    environment = await _get_hosted_environment(env_id)
    tasks = await environment.list_tasks(split=split)
    global _HOSTED_LOCK
    if _HOSTED_LOCK is None:
        _HOSTED_LOCK = asyncio.Lock()
    async with _HOSTED_LOCK:
        cached = _HOSTED_TASKS.get(cache_key)
        if cached is not None:
            return cached
        _HOSTED_TASKS[cache_key] = tasks
        return tasks


async def _resolve_hosted_task(env_id: str, task_spec: dict[str, Any]) -> Any:
    split = str(task_spec.get("split", "train"))
    task_id = str(task_spec.get("task_id", ""))
    tasks = await _list_hosted_tasks(env_id, split)
    for task in tasks:
        candidate = getattr(task, "task_spec", {})
        if str(candidate.get("task_id", "")) == task_id:
            return task
    raise RuntimeError(
        f"Could not resolve hosted task task_id={task_id!r} split={split!r} in env={env_id!r}."
    )


async def _list_hosted_tool_specs(env_id: str) -> list[Any]:
    specs = _HOSTED_TOOL_SPECS.get(env_id)
    if specs is not None:
        return specs
    environment = await _get_hosted_environment(env_id)
    specs = list(await environment.list_tools())
    global _HOSTED_LOCK
    if _HOSTED_LOCK is None:
        _HOSTED_LOCK = asyncio.Lock()
    async with _HOSTED_LOCK:
        cached = _HOSTED_TOOL_SPECS.get(env_id)
        if cached is not None:
            return cached
        _HOSTED_TOOL_SPECS[env_id] = specs
        return specs


async def close_hosted_sessions() -> None:
    global _HOSTED_CLIENT, _HOSTED_LOCK
    if _HOSTED_CLIENT is None:
        return
    close = getattr(_HOSTED_CLIENT, "close", None)
    if close is not None:
        maybe = close()
        if inspect.isawaitable(maybe):
            await maybe
    _HOSTED_CLIENT = None
    _HOSTED_ENVIRONMENTS.clear()
    _HOSTED_TASKS.clear()
    _HOSTED_TOOL_SPECS.clear()
    _HOSTED_LOCK = None


@dataclass
class ToolCallResult:
    ok: bool
    text: str
    reward: float
    finished: bool
    metadata: dict[str, Any] | None = None


class InProcessFarmSession:
    def __init__(self, task_spec: dict[str, Any]) -> None:
        self.env = UKArableManager(task_spec=task_spec)

    async def open(self) -> None:
        result = self.env.setup()
        if inspect.isawaitable(result):
            await result

    async def close(self) -> None:
        result = self.env.teardown()
        if inspect.isawaitable(result):
            await result

    def prompt_text(self) -> str:
        return _blocks_to_text(list(self.env.get_prompt()))

    def chat_tools(self) -> list[dict[str, Any]]:
        specs = list(self.env.list_tools().tools)
        if hasattr(self.env, "list_task_tools"):
            specs.extend(list(self.env.list_task_tools().tools))
        return _tool_specs_to_chat_tools(specs)

    def state(self) -> dict[str, Any]:
        sim = self.env._ensure_sim()
        state = sim.state.to_dict()
        state["current_prices"] = sim.get_current_prices()
        return state

    async def call_tool(self, name: str, payload: dict[str, Any]) -> ToolCallResult:
        try:
            raw = await self.env._call_tool(name, payload)
            result = raw.root
        except Exception as exc:
            return ToolCallResult(
                ok=False,
                text=f"Tool exception: {type(exc).__name__}: {exc}",
                reward=-1.0,
                finished=False,
                metadata=None,
            )

        if not result.ok:
            return ToolCallResult(
                ok=False,
                text=f"Tool error: {result.error}",
                reward=-1.0,
                finished=False,
                metadata=None,
            )

        output = result.output
        return ToolCallResult(
            ok=True,
            text=_blocks_to_text(list(output.blocks)),
            reward=float(output.reward or 0.0),
            finished=bool(output.finished),
            metadata=dict(output.metadata or {}),
        )

    def episode_metrics(self) -> dict[str, Any]:
        sim = self.env._ensure_sim()
        state = sim.state
        mean_final_soil = sum(plot.soil_health for plot in state.plots) / len(state.plots)
        finished = state.quarter >= TOTAL_QUARTERS or state.cash < BANKRUPTCY_HARD_THRESHOLD
        terminal_score = sim.terminal_score() if finished else None
        return {
            "cash": float(state.cash),
            "starting_cash": float(state.starting_cash),
            "mean_final_soil": float(mean_final_soil),
            "quarter": int(state.quarter),
            "ever_bankrupt": bool(state.ever_bankrupt),
            "finished": bool(finished),
            "terminal_score": terminal_score,
        }


def format_commit_plan_payload(action: dict[str, Any]) -> dict[str, Any]:
    return _commit_plan_payload(action)


class HostedFarmSession:
    def __init__(self, task_spec: dict[str, Any], *, env_id: str) -> None:
        self.task_spec = task_spec
        self.env_id = env_id
        self._environment: Any | None = None
        self._session_cm: Any | None = None
        self._session: Any | None = None
        self._prompt_text = ""
        self._tools: list[dict[str, Any]] = []
        self._cached_state: dict[str, Any] | None = None
        self._episode_metrics: dict[str, Any] | None = None

    async def open(self) -> None:
        self._environment = await _get_hosted_environment(self.env_id)
        task = await _resolve_hosted_task(self.env_id, self.task_spec)
        self._session_cm = self._environment.session(task=task)
        self._session = await self._session_cm.__aenter__()
        prompt_blocks = await self._session.get_prompt()
        self._prompt_text = _blocks_to_text(list(prompt_blocks))
        specs = await _list_hosted_tool_specs(self.env_id)
        self._tools = _tool_specs_to_chat_tools(specs)

    async def close(self) -> None:
        if self._session_cm is None:
            return
        await self._session_cm.__aexit__(None, None, None)
        self._session_cm = None
        self._session = None

    def prompt_text(self) -> str:
        return self._prompt_text

    def chat_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    def state(self) -> dict[str, Any]:
        if self._cached_state is None:
            return {}
        return dict(self._cached_state)

    def _update_from_metadata(self, metadata: dict[str, Any] | None) -> None:
        if not metadata:
            return
        state = metadata.get("state")
        if isinstance(state, dict):
            self._cached_state = dict(state)
        episode_metrics = metadata.get("episode_metrics")
        if isinstance(episode_metrics, dict):
            self._episode_metrics = dict(episode_metrics)

    async def call_tool(self, name: str, payload: dict[str, Any]) -> ToolCallResult:
        if self._session is None:
            raise RuntimeError("HostedFarmSession.open() must be awaited before calling tools.")
        try:
            output = await self._session.call_tool(name, payload)
        except Exception as exc:
            return ToolCallResult(
                ok=False,
                text=f"Tool exception: {type(exc).__name__}: {exc}",
                reward=-1.0,
                finished=False,
                metadata=None,
            )

        metadata = dict(output.metadata or {})
        self._update_from_metadata(metadata)
        if self._episode_metrics is None and self._cached_state is not None:
            self._episode_metrics = _episode_metrics_from_state(self._cached_state)

        return ToolCallResult(
            ok=True,
            text=_blocks_to_text(list(output.blocks)),
            reward=float(output.reward or 0.0),
            finished=bool(output.finished),
            metadata=metadata,
        )

    def episode_metrics(self) -> dict[str, Any]:
        if self._episode_metrics is not None:
            return dict(self._episode_metrics)
        if self._cached_state is not None:
            return _episode_metrics_from_state(self._cached_state)
        return {
            "cash": 0.0,
            "starting_cash": float(self.task_spec.get("starting_cash", 150_000.0)),
            "mean_final_soil": 0.55,
            "quarter": 0,
            "ever_bankrupt": False,
            "finished": False,
            "terminal_score": None,
        }


def build_farm_session(
    task_spec: dict[str, Any],
    *,
    session_backend: str,
    openreward_env_id: str,
) -> HostedFarmSession | InProcessFarmSession:
    if session_backend == "hosted":
        return HostedFarmSession(task_spec, env_id=openreward_env_id)
    if session_backend == "inprocess":
        return InProcessFarmSession(task_spec)
    raise ValueError(
        f"Unsupported session_backend={session_backend!r}. Expected 'hosted' or 'inprocess'."
    )
