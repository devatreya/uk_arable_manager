import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat.chat_completion import Choice

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOTAL_QUARTERS
from pipeline import art_rollout
from pipeline.art_rollout import (
    FarmScenario,
    _messages_for_inference,
    rollout,
    trajectory_for_logging,
)
from pipeline.config import DEFAULT_READ_TOOL_SEQUENCE


class FakeToolResult:
    def __init__(self, *, ok: bool = True, reward: float = 0.0, finished: bool = False) -> None:
        self.ok = ok
        self.reward = reward
        self.finished = finished


class FakeFarmSession:
    def __init__(self, *, raise_on_open: bool = False, raise_on_tool: str | None = None) -> None:
        self.raise_on_open = raise_on_open
        self.raise_on_tool = raise_on_tool
        self.quarter = 0
        self.closed = False

    async def open(self) -> None:
        if self.raise_on_open:
            raise RuntimeError("open failed")

    async def close(self) -> None:
        self.closed = True

    def prompt_text(self) -> str:
        return "Manage the farm."

    def chat_tools(self) -> list[dict[str, object]]:
        return []

    def state(self) -> dict[str, object]:
        return {
            "quarter": self.quarter,
            "cash": 150_000.0 + float(self.quarter),
            "starting_cash": 150_000.0,
            "irrigation_owned": False,
            "plots": [
                {
                    "current_crop": "fallow",
                    "previous_crop": "cover_crop",
                    "_organic_matter": 0.55,
                    "_structure": 0.55,
                    "_ph": 0.55,
                    "_nutrient_balance": 0.55,
                }
                for _ in range(4)
            ],
            "weather_history": [
                {
                    "quarter": max(self.quarter - 1, 0),
                    "regime": "normal",
                    "rainfall_index": 1.0,
                    "temperature_index": 1.0,
                }
            ],
            "current_prices": {
                "wheat": 140.0,
                "barley": 120.0,
                "oilseed_rape": 320.0,
                "field_beans": 120.0,
                "cover_crop": 0.0,
                "fallow": 0.0,
                "fertiliser_low": 50.0,
                "fertiliser_medium": 100.0,
                "fertiliser_high": 150.0,
                "irrigation_cost": 35_000.0,
            },
        }

    async def call_tool(self, name: str, payload: dict[str, object]) -> FakeToolResult:
        del payload
        if name == self.raise_on_tool:
            raise RuntimeError("tool blew up")
        if name == "commit_plan":
            self.quarter += 1
            return FakeToolResult(reward=1.0, finished=self.quarter >= TOTAL_QUARTERS)
        return FakeToolResult(reward=0.0, finished=False)

    def episode_metrics(self) -> dict[str, object]:
        finished = self.quarter >= TOTAL_QUARTERS
        return {
            "cash": 150_000.0 + float(self.quarter),
            "starting_cash": 150_000.0,
            "mean_final_soil": 0.55,
            "quarter": self.quarter,
            "ever_bankrupt": False,
            "finished": finished,
            "terminal_score": 1.25 if finished else None,
        }


def _make_choice(tool_calls: list[tuple[str, dict[str, object]]]) -> Choice:
    return Choice.model_validate(
        {
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(payload, sort_keys=True),
                        },
                    }
                    for index, (name, payload) in enumerate(tool_calls)
                ]
                or None,
            },
        }
    )


class FakeCompletions:
    def __init__(self, planned_tool_calls: list[list[tuple[str, dict[str, object]]]]) -> None:
        self.planned_tool_calls = planned_tool_calls
        self.call_index = 0

    async def create(self, **kwargs) -> SimpleNamespace:
        del kwargs
        tool_calls = self.planned_tool_calls[self.call_index]
        self.call_index += 1
        return SimpleNamespace(choices=[_make_choice(tool_calls)])


class FakeModel:
    def __init__(self, planned_tool_calls: list[list[tuple[str, dict[str, object]]]]) -> None:
        self._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(planned_tool_calls)))

    def openai_client(self) -> SimpleNamespace:
        return self._client

    def get_inference_name(self) -> str:
        return "fake-model"


def _commit_payload() -> dict[str, object]:
    return {
        "capital_action": "none",
        "plot_1": {"crop": "wheat", "fertiliser": "medium", "pest_control": "ipm"},
        "plot_2": {"crop": "barley", "fertiliser": "medium", "pest_control": "ipm"},
        "plot_3": {"crop": "field_beans", "fertiliser": "low", "pest_control": "none"},
        "plot_4": {"crop": "cover_crop", "fertiliser": "low", "pest_control": "none"},
    }


def _scenario() -> FarmScenario:
    return FarmScenario(
        task_spec={
            "task_id": "train_9999",
            "split": "train",
            "scenario_type": "standard",
            "starting_cash": 150_000.0,
        }
    )


def test_messages_for_inference_keeps_system_and_current_quarter_only() -> None:
    trajectory = SimpleNamespace(
        messages=lambda: [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Quarter 1"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "Quarter 1 state", "tool_call_id": "t1"},
            {"role": "assistant", "content": "Quarter 1 plan committed."},
            {"role": "user", "content": "Quarter 2"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "Quarter 2 state", "tool_call_id": "t2"},
        ]
    )

    assert _messages_for_inference(trajectory) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Quarter 2"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": "Quarter 2 state", "tool_call_id": "t2"},
    ]


def test_trajectory_for_logging_strips_string_metrics() -> None:
    class FakeTrajectory:
        def __init__(self) -> None:
            self.metrics = {
                "termination_reason": "finished",
                "last_tool_name": "commit_plan",
                "tool_calls": 200,
                "completed_all_quarters": True,
            }

        def model_copy(self, deep: bool = True) -> "FakeTrajectory":
            del deep
            copied = FakeTrajectory()
            copied.metrics = dict(self.metrics)
            return copied

    sanitized = trajectory_for_logging(FakeTrajectory())

    assert sanitized.metrics == {
        "tool_calls": 200,
        "completed_all_quarters": True,
    }


@pytest.mark.asyncio
async def test_rollout_budget_allows_full_episode_and_caps_early(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeFarmSession()
    monkeypatch.setattr(art_rollout, "build_farm_session", lambda *args, **kwargs: session)

    per_quarter_tool_calls = list(DEFAULT_READ_TOOL_SEQUENCE) + [("commit_plan", _commit_payload())]
    model = FakeModel([per_quarter_tool_calls for _ in range(TOTAL_QUARTERS)])

    full_trajectory = await rollout(model, _scenario(), max_tool_calls=240)
    assert full_trajectory.metrics["termination_reason"] == "finished"
    assert full_trajectory.metrics["completed_all_quarters"] is True
    assert full_trajectory.metrics["quarters_completed"] == TOTAL_QUARTERS
    assert full_trajectory.metrics["tool_calls"] == TOTAL_QUARTERS * len(per_quarter_tool_calls)
    assert full_trajectory.metrics["last_tool_name"] == "commit_plan"
    assert session.closed is True

    limited_session = FakeFarmSession()
    monkeypatch.setattr(art_rollout, "build_farm_session", lambda *args, **kwargs: limited_session)
    limited_model = FakeModel([per_quarter_tool_calls for _ in range(TOTAL_QUARTERS)])

    limited_trajectory = await rollout(limited_model, _scenario(), max_tool_calls=160)
    assert limited_trajectory.metrics["termination_reason"] == "tool_budget_exhausted"
    assert limited_trajectory.metrics["completed_all_quarters"] is False
    assert limited_trajectory.metrics["quarters_completed"] == 32
    assert limited_trajectory.metrics["tool_calls"] == 160
    assert limited_trajectory.metrics["last_tool_name"] == "commit_plan"
    assert limited_trajectory.metrics["terminated_early"] == 1
    assert limited_session.closed is True


@pytest.mark.asyncio
async def test_rollout_marks_missing_tool_call_and_applies_early_termination_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeFarmSession()
    monkeypatch.setattr(art_rollout, "build_farm_session", lambda *args, **kwargs: session)
    model = FakeModel([[]])

    trajectory = await rollout(model, _scenario(), max_tool_calls=240)

    assert trajectory.metrics["termination_reason"] == "missing_tool_call"
    assert trajectory.metrics["missing_tool_call"] == 1
    assert trajectory.metrics["terminated_early"] == 1
    assert trajectory.reward == -4.0
    assert trajectory.metrics["last_tool_name"] is None


@pytest.mark.asyncio
async def test_rollout_exception_includes_task_and_quarter_context(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeFarmSession(raise_on_tool="read_farm_state")
    monkeypatch.setattr(art_rollout, "build_farm_session", lambda *args, **kwargs: session)
    model = FakeModel([[("read_farm_state", {})]])

    with pytest.raises(RuntimeError, match=r"task_id=train_9999 .*current_quarter=0 .*tool_calls=1"):
        await rollout(model, _scenario(), max_tool_calls=240)

    assert session.closed is True
