"""
Rollout client: runs policies against a live OpenReward server or directly
against the simulator for local/offline use.

Two modes:
  LocalRollout   — calls FarmSimulator directly (no HTTP)
  ServerRollout  — calls a running localhost server via openreward.EnvironmentsAPI
  HostedRollout  — calls a deployed OpenReward environment over HTTPS
"""
from __future__ import annotations

import asyncio
import json
import os
import ssl
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import NUM_PLOTS, TOTAL_QUARTERS
from hosted_state import initial_state_from_task, update_state_from_tool_output
from pipeline.config import DEFAULT_READ_TOOL_SEQUENCE
from sim import FarmAction, FarmSimulator, PlotAction
from trajectory_logger import TrajectoryLogger, TrajectoryLog


# ── Policy protocol ───────────────────────────────────────────────────────────
# A policy is any callable with signature:
#   policy(state: Dict[str, Any]) -> Dict[str, Any]
# where the returned dict matches the commit_plan action shape:
#   {
#     "capital_action": "none" | "buy_irrigation",
#     "plots": [
#       {"crop": str, "fertiliser": str, "pest_control": str},
#       ...  (4 entries)
#     ]
#   }
Policy = Callable[[Dict[str, Any]], Dict[str, Any]]

DEFAULT_OPENREWARD_ENV_ID = os.environ.get("OPENREWARD_ENV_ID", "devatreya/uk_arable_manager")
_ORIGINAL_CREATE_DEFAULT_CONTEXT = ssl.create_default_context


def _configure_ssl_cert_file() -> None:
    try:
        import certifi
    except ImportError:
        return
    cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    if getattr(ssl, "_uk_arable_certifi_patched", False):
        return

    def _create_default_context(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        if "cafile" not in kwargs:
            kwargs["cafile"] = cafile
        return _ORIGINAL_CREATE_DEFAULT_CONTEXT(*args, **kwargs)

    ssl.create_default_context = _create_default_context
    ssl._uk_arable_certifi_patched = True


_configure_ssl_cert_file()


# ── Local rollout (no HTTP) ───────────────────────────────────────────────────

class LocalRollout:
    """
    Run a policy directly against FarmSimulator.
    Fast, fully deterministic, no server required.
    """

    def __init__(self, task_spec: Dict[str, Any]) -> None:
        self.task_spec = task_spec
        self.sim = FarmSimulator(task_spec)

    def run(
        self,
        policy: Policy,
        baseline_name: Optional[str] = None,
        save_snapshots: bool = False,
    ) -> TrajectoryLog:
        logger = TrajectoryLogger(self.task_spec, baseline_name=baseline_name)
        sim = self.sim

        while sim.state.quarter < TOTAL_QUARTERS:
            obs = sim.state.to_dict()
            action_dict = policy(obs)
            action = _dict_to_action(action_dict)
            result = sim.step(action)

            snapshot = sim.snapshot() if save_snapshots else None
            logger.record_step(
                quarter=sim.state.quarter - 1,
                action=action_dict,
                reward=result.reward,
                pnl=result.pnl,
                observation=f"pnl={result.pnl:.0f} weather={result.weather.regime}",
                weather=result.weather.to_dict(),
                plot_pnl=result.plot_pnl,
                bankrupt=result.bankrupt,
                pest_pressure=result.pest_pressure,
                state_snapshot=snapshot,
            )

            if result.finished:
                break

        return logger.finalise(
            final_state=sim.state.to_dict(),
            terminal_score=sim.terminal_score(),
        )


def run_local_rollout(
    task_spec: Dict[str, Any],
    policy: Policy,
    baseline_name: Optional[str] = None,
    save_to: Optional[Path] = None,
    save_snapshots: bool = False,
) -> TrajectoryLog:
    """Convenience wrapper."""
    runner = LocalRollout(task_spec)
    traj = runner.run(policy, baseline_name=baseline_name, save_snapshots=save_snapshots)
    if save_to is not None:
        traj.save(save_to)
    return traj


def _extract_state_metadata(response: Any) -> Dict[str, Any]:
    metadata = getattr(response, "metadata", None) or {}
    state = metadata.get("state", {})
    return dict(state) if isinstance(state, dict) else {}


def _extract_episode_metrics(response: Any) -> Dict[str, Any]:
    metadata = getattr(response, "metadata", None) or {}
    metrics = metadata.get("episode_metrics", {})
    return dict(metrics) if isinstance(metrics, dict) else {}


def _extract_step_metrics(response: Any) -> Dict[str, Any]:
    metadata = getattr(response, "metadata", None) or {}
    step = metadata.get("step", {})
    return dict(step) if isinstance(step, dict) else {}


# ── Server rollout (HTTP via ORS client) ──────────────────────────────────────

class ServerRollout:
    """
    Run a policy against a running OpenReward server (started via server.py).
    Uses openreward.EnvironmentsAPI directly so it works against localhost
    without subdomain routing.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        env_name: str = "UKArableManager",
    ) -> None:
        self.base_url = base_url
        self.env_name = env_name

    def run(
        self,
        task_spec: Dict[str, Any],
        policy: Policy,
        baseline_name: Optional[str] = None,
    ) -> TrajectoryLog:
        from openreward import EnvironmentsAPI

        logger = TrajectoryLogger(task_spec, baseline_name=baseline_name)

        with EnvironmentsAPI(base_url=self.base_url, api_key="local") as api:
            env = api.get(self.env_name, base_url=self.base_url)
            tasks = env.list_tasks(task_spec.get("split", "train"))
            # Find task matching task_id, or use first task as fallback
            task_id = task_spec.get("task_id")
            task = next(
                (t for t in tasks if t.task_spec.get("task_id") == task_id),
                tasks[0] if tasks else None,
            )
            if task is None:
                raise RuntimeError(f"No tasks found for split {task_spec.get('split', 'train')!r}")

            with env.session(task) as session:
                obs_state: Dict[str, Any] = {}
                quarter = 0

                while quarter < TOTAL_QUARTERS:
                    fs_resp = session.call_tool("read_farm_state", {})
                    obs_state["farm_state_text"] = _extract_text(fs_resp)

                    action_dict = policy(obs_state)
                    commit_input = _action_to_commit_plan_input(action_dict)
                    result = session.call_tool("commit_plan", commit_input)

                    finished = result.finished
                    reward   = result.reward or 0.0
                    obs_text = _extract_text(result)

                    logger.record_step(
                        quarter=quarter,
                        action=action_dict,
                        reward=float(reward),
                        pnl=0.0,
                        observation=obs_text,
                        weather={},
                        plot_pnl=[0.0] * NUM_PLOTS,
                        bankrupt=False,
                        pest_pressure=[False] * NUM_PLOTS,
                    )

                    quarter += 1
                    if finished:
                        break

        return logger.finalise(final_state={}, terminal_score=None)


class HostedRollout:
    """
    Run a policy against a deployed OpenReward environment using the current SDK.
    """

    def __init__(self, env_id: str = DEFAULT_OPENREWARD_ENV_ID) -> None:
        self.env_id = env_id

    def run(
        self,
        task_spec: Dict[str, Any],
        policy: Policy,
        baseline_name: Optional[str] = None,
    ) -> TrajectoryLog:
        return asyncio.run(
            run_hosted_rollout_async(
                task_spec,
                policy,
                env_id=self.env_id,
                baseline_name=baseline_name,
            )
        )

def _task_index(task_spec: Dict[str, Any]) -> int:
    task_id = str(task_spec.get("task_id", ""))
    try:
        return int(task_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Task id {task_id!r} does not end with a numeric index.") from exc


class HostedRolloutManager:
    def __init__(self, env_id: str = DEFAULT_OPENREWARD_ENV_ID) -> None:
        self.env_id = env_id
        self._client: Any | None = None
        self._environment: Any | None = None

    async def __aenter__(self) -> "HostedRolloutManager":
        _configure_ssl_cert_file()
        from openreward import AsyncOpenReward

        api_key = os.environ.get("OPENREWARD_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENREWARD_API_KEY must be set for hosted OpenReward rollouts.")
        self._client = AsyncOpenReward(api_key=api_key)
        self._environment = self._client.environments.get(name=self.env_id)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        client = self._client
        self._environment = None
        self._client = None
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            maybe = close()
            if asyncio.iscoroutine(maybe):
                await maybe

    async def run(
        self,
        task_spec: Dict[str, Any],
        policy: Policy,
        *,
        baseline_name: Optional[str] = None,
        save_to: Optional[Path] = None,
    ) -> TrajectoryLog:
        if self._environment is None:
            raise RuntimeError("HostedRolloutManager must be used inside 'async with'.")

        logger = TrajectoryLogger(task_spec, baseline_name=baseline_name)
        split = str(task_spec.get("split", "train"))
        index = _task_index(task_spec)
        state = initial_state_from_task(task_spec)
        final_state: Dict[str, Any] = dict(state)
        terminal_score: float | None = None
        ever_bankrupt = False

        async with self._environment.session(split=split, index=index) as session:
            quarter = 0
            while quarter < TOTAL_QUARTERS:
                for tool_name, tool_payload in DEFAULT_READ_TOOL_SEQUENCE:
                    tool_output = await session.call_tool(tool_name, tool_payload)
                    metadata_state = _extract_state_metadata(tool_output)
                    if metadata_state:
                        state = metadata_state
                    else:
                        parsed = update_state_from_tool_output(
                            tool_name,
                            _extract_text(tool_output),
                            state,
                            payload=tool_payload,
                        )
                        state = parsed["state"]

                action_dict = policy(state)
                commit_input = _action_to_commit_plan_input(action_dict)
                result = await session.call_tool("commit_plan", commit_input)

                state_after = _extract_state_metadata(result)
                episode_metrics = _extract_episode_metrics(result)
                step_metrics = _extract_step_metrics(result)
                if state_after:
                    state = state_after
                else:
                    parsed = update_state_from_tool_output(
                        "commit_plan",
                        _extract_text(result),
                        state,
                        payload=commit_input,
                    )
                    state = parsed["state"]
                    if not step_metrics:
                        step_metrics = dict(parsed.get("step", {}))
                    if not episode_metrics:
                        episode_metrics = dict(parsed.get("episode_metrics", {}))
                final_state = dict(state)
                terminal_score = episode_metrics.get("terminal_score", terminal_score)
                ever_bankrupt = bool(episode_metrics.get("ever_bankrupt", ever_bankrupt))

                logger.record_step(
                    quarter=quarter,
                    action=action_dict,
                    reward=float(getattr(result, "reward", 0.0) or 0.0),
                    pnl=float(step_metrics.get("pnl", 0.0)),
                    observation=_extract_text(result),
                    weather=dict(step_metrics.get("weather", {})),
                    plot_pnl=[float(v) for v in step_metrics.get("plot_pnl", [0.0] * NUM_PLOTS)],
                    bankrupt=ever_bankrupt,
                    pest_pressure=[bool(v) for v in step_metrics.get("pest_pressure", [False] * NUM_PLOTS)],
                    state_snapshot=state_after or dict(state),
                )

                quarter += 1
                if getattr(result, "finished", False):
                    break

        trajectory = logger.finalise(final_state=final_state, terminal_score=terminal_score)
        if save_to is not None:
            trajectory.save(save_to)
        return trajectory


async def run_hosted_rollout_async(
    task_spec: Dict[str, Any],
    policy: Policy,
    *,
    env_id: str = DEFAULT_OPENREWARD_ENV_ID,
    baseline_name: Optional[str] = None,
    save_to: Optional[Path] = None,
    manager: HostedRolloutManager | None = None,
) -> TrajectoryLog:
    if manager is not None:
        return await manager.run(task_spec, policy, baseline_name=baseline_name, save_to=save_to)

    async with HostedRolloutManager(env_id=env_id) as owned_manager:
        return await owned_manager.run(task_spec, policy, baseline_name=baseline_name, save_to=save_to)


def run_hosted_rollout(
    task_spec: Dict[str, Any],
    policy: Policy,
    *,
    env_id: str = DEFAULT_OPENREWARD_ENV_ID,
    baseline_name: Optional[str] = None,
    save_to: Optional[Path] = None,
) -> TrajectoryLog:
    runner = HostedRollout(env_id=env_id)
    traj = runner.run(task_spec, policy, baseline_name=baseline_name)
    if save_to is not None:
        traj.save(save_to)
    return traj


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dict_to_action(d: Dict[str, Any]) -> FarmAction:
    capital = d.get("capital_action", "none")
    plot_dicts = d.get("plots", [])
    plots: List[PlotAction] = []
    for i in range(NUM_PLOTS):
        pd = plot_dicts[i] if i < len(plot_dicts) else {}
        plots.append(PlotAction(
            plot_id=i,
            crop=pd.get("crop", "fallow"),
            fertiliser=pd.get("fertiliser", "medium"),
            pest_control=pd.get("pest_control", "none"),
        ))
    return FarmAction(capital_action=capital, plots=plots)


def _action_to_commit_plan_input(d: Dict[str, Any]) -> Dict[str, Any]:
    plots = d.get("plots", [{}] * NUM_PLOTS)
    result: Dict[str, Any] = {
        "capital_action": d.get("capital_action", "none"),
    }
    for i in range(NUM_PLOTS):
        pd = plots[i] if i < len(plots) else {}
        result[f"plot_{i+1}"] = {
            "crop": pd.get("crop", "fallow"),
            "fertiliser": pd.get("fertiliser", "medium"),
            "pest_control": pd.get("pest_control", "none"),
        }
    return result


def _extract_text(response: Any) -> str:
    if hasattr(response, "blocks"):
        return "\n".join(b.text for b in response.blocks if hasattr(b, "text"))
    return str(response)
