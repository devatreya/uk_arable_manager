import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.farm_session import HostedFarmSession


class _FakeHostedSession:
    def __init__(self, output: SimpleNamespace) -> None:
        self._output = output

    async def call_tool(self, name: str, payload: dict[str, object]) -> SimpleNamespace:
        del name, payload
        return self._output


@pytest.mark.asyncio
async def test_hosted_session_prefers_metadata_over_text_parsing() -> None:
    hosted = HostedFarmSession(
        {
            "task_id": "validation_0000",
            "split": "validation",
            "starting_cash": 150_000.0,
        },
        env_id="dummy",
    )
    hosted._session = _FakeHostedSession(
        SimpleNamespace(
            blocks=[
                SimpleNamespace(text="── Quarter 12/40 complete ─────────────────"),
                SimpleNamespace(text="Cash balance: £200,000"),
            ],
            reward=3.0,
            finished=True,
            metadata={
                "state": {
                    "quarter": 40,
                    "cash": 1_200_000.0,
                    "starting_cash": 150_000.0,
                    "irrigation_owned": False,
                    "ever_bankrupt": False,
                    "weather_regime": "normal",
                    "weather_history": [],
                    "plots": [],
                    "current_prices": {},
                },
                "episode_metrics": {
                    "cash": 1_200_000.0,
                    "starting_cash": 150_000.0,
                    "mean_final_soil": 0.51,
                    "quarter": 40,
                    "ever_bankrupt": False,
                    "finished": True,
                    "terminal_score": 1.2345,
                },
            },
        )
    )

    result = await hosted.call_tool("commit_plan", {})

    assert result.ok is True
    assert hosted.state()["quarter"] == 40
    assert hosted.state()["cash"] == 1_200_000.0
    assert hosted.episode_metrics()["quarter"] == 40
    assert hosted.episode_metrics()["finished"] is True
    assert hosted.episode_metrics()["terminal_score"] == 1.2345
