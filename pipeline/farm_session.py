from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BANKRUPTCY_HARD_THRESHOLD, NUM_PLOTS, TOTAL_QUARTERS
from env import UKArableManager


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
        tools: list[dict[str, Any]] = []
        for spec in specs:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.input_schema or {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            )
        return tools

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
