"""
Trajectory logger for uk_arable_manager episodes.

Captures full episode history: task spec, per-step actions/rewards/observations,
final state, and graded outcomes. Serialisable to JSON for offline evaluation
and dataset generation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StepLog:
    quarter: int
    action: Dict[str, Any]
    reward: float
    pnl: float
    observation: str          # compact text returned by commit_plan
    weather: Dict[str, Any]
    plot_pnl: List[float]
    bankrupt: bool
    pest_pressure: List[bool]
    state_snapshot: Optional[Dict[str, Any]] = None


@dataclass
class TrajectoryLog:
    task_spec: Dict[str, Any]
    steps: List[StepLog] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    terminal_score: Optional[float] = None
    ever_bankrupt: bool = False
    quarters_completed: int = 0
    total_reward: float = 0.0
    total_pnl: float = 0.0
    baseline_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def mean_final_soil(self) -> float:
        plots = self.final_state.get("plots", [])
        if not plots:
            return 0.55
        def h(p: Dict[str, Any]) -> float:
            return (0.45 * p.get("_organic_matter", 0.55)
                    + 0.20 * p.get("_structure", 0.55)
                    + 0.15 * p.get("_ph", 0.55)
                    + 0.20 * p.get("_nutrient_balance", 0.55))
        return sum(h(p) for p in plots) / len(plots)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_spec": self.task_spec,
            "steps": [
                {
                    "quarter": s.quarter,
                    "action": s.action,
                    "reward": s.reward,
                    "pnl": s.pnl,
                    "observation": s.observation,
                    "weather": s.weather,
                    "plot_pnl": s.plot_pnl,
                    "bankrupt": s.bankrupt,
                    "pest_pressure": s.pest_pressure,
                    "state_snapshot": s.state_snapshot,
                }
                for s in self.steps
            ],
            "final_state": self.final_state,
            "terminal_score": self.terminal_score,
            "ever_bankrupt": self.ever_bankrupt,
            "quarters_completed": self.quarters_completed,
            "total_reward": self.total_reward,
            "total_pnl": self.total_pnl,
            "mean_final_soil": self.mean_final_soil,
            "baseline_name": self.baseline_name,
            "created_at": self.created_at,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrajectoryLog":
        with open(path) as f:
            d = json.load(f)
        traj = cls(task_spec=d["task_spec"])
        traj.final_state = d.get("final_state", {})
        traj.terminal_score = d.get("terminal_score")
        traj.ever_bankrupt = d.get("ever_bankrupt", False)
        traj.quarters_completed = d.get("quarters_completed", 0)
        traj.total_reward = d.get("total_reward", 0.0)
        traj.total_pnl = d.get("total_pnl", 0.0)
        traj.baseline_name = d.get("baseline_name")
        traj.created_at = d.get("created_at", 0.0)
        for s in d.get("steps", []):
            traj.steps.append(StepLog(
                quarter=s["quarter"],
                action=s["action"],
                reward=s["reward"],
                pnl=s["pnl"],
                observation=s.get("observation", ""),
                weather=s["weather"],
                plot_pnl=s["plot_pnl"],
                bankrupt=s["bankrupt"],
                pest_pressure=s["pest_pressure"],
                state_snapshot=s.get("state_snapshot"),
            ))
        return traj


class TrajectoryLogger:
    """Wraps a TrajectoryLog and provides a clean recording API."""

    def __init__(self, task_spec: Dict[str, Any], baseline_name: Optional[str] = None) -> None:
        self.log = TrajectoryLog(task_spec=task_spec, baseline_name=baseline_name)

    def record_step(
        self,
        quarter: int,
        action: Dict[str, Any],
        reward: float,
        pnl: float,
        observation: str,
        weather: Dict[str, Any],
        plot_pnl: List[float],
        bankrupt: bool,
        pest_pressure: List[bool],
        state_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log.total_reward += reward
        self.log.total_pnl += pnl
        if bankrupt:
            self.log.ever_bankrupt = True
        self.log.steps.append(StepLog(
            quarter=quarter,
            action=action,
            reward=reward,
            pnl=pnl,
            observation=observation,
            weather=weather,
            plot_pnl=plot_pnl,
            bankrupt=bankrupt,
            pest_pressure=pest_pressure,
            state_snapshot=state_snapshot,
        ))

    def finalise(
        self,
        final_state: Dict[str, Any],
        terminal_score: Optional[float],
    ) -> TrajectoryLog:
        self.log.final_state = final_state
        self.log.terminal_score = terminal_score
        self.log.quarters_completed = len(self.log.steps)
        return self.log
