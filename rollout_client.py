"""
Rollout client: runs policies against a live OpenReward server or directly
against the simulator for local/offline use.

Two modes:
  LocalRollout   — calls FarmSimulator directly (no HTTP)
  ServerRollout  — calls a running server via openreward.EnvironmentsAPI
"""
from __future__ import annotations

import json
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
