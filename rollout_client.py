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
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import NUM_PLOTS, TOTAL_QUARTERS
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


def _configure_ssl_cert_file() -> None:
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


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
        from openreward import OpenReward

        logger = TrajectoryLogger(task_spec, baseline_name=baseline_name)
        split = str(task_spec.get("split", "train"))
        task_id = str(task_spec.get("task_id", ""))

        _configure_ssl_cert_file()
        client = OpenReward(api_key=os.environ["OPENREWARD_API_KEY"])
        try:
            env = client.environments.get(name=self.env_id)
            tasks = env.list_tasks(split)
            task = next(
                (candidate for candidate in tasks if str(candidate.task_spec.get("task_id", "")) == task_id),
                None,
            )
            if task is None:
                raise RuntimeError(
                    f"Task task_id={task_id!r} not found in split={split!r} for env={self.env_id!r}"
                )

            with env.session(task=task) as session:
                quarter = 0
                final_state: Dict[str, Any] = {}
                terminal_score: float | None = None
                ever_bankrupt = False

                while quarter < TOTAL_QUARTERS:
                    farm_state = session.call_tool("read_farm_state", {})
                    obs_state = _extract_state_metadata(farm_state)

                    action_dict = policy(obs_state)
                    commit_input = _action_to_commit_plan_input(action_dict)
                    result = session.call_tool("commit_plan", commit_input)

                    state_after = _extract_state_metadata(result)
                    episode_metrics = _extract_episode_metrics(result)
                    step_metrics = _extract_step_metrics(result)
                    final_state = state_after or final_state
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
                        state_snapshot=state_after or None,
                    )

                    quarter += 1
                    if getattr(result, "finished", False):
                        break

        finally:
            close = getattr(client, "close", None)
            if callable(close):
                maybe = close()
                if inspect.isawaitable(maybe):
                    asyncio.run(maybe)

        return logger.finalise(final_state=final_state, terminal_score=terminal_score)


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
