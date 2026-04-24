"""
Grader logic for uk_arable_manager.

Three grader variants:
  ScalarFinalScoreGrader     — the raw terminal_score
  BankruptcyAwareGrader      — penalises bankruptcy more aggressively
  StewardshipWeightedGrader  — weights soil preservation above cash

Each grader accepts a trajectory dict (produced by TrajectoryLogger)
and returns a GradeResult with a scalar score in [0, 1] and metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from config import (
    SOLVENCY_GATE_PENALTY,
    TERMINAL_SOIL_MAX,
    TERMINAL_SOIL_MIN,
)


@dataclass
class GradeResult:
    score: float               # primary scalar [0, ∞)
    normalised: float          # clipped to [0, 1]
    ending_cash: float
    starting_cash: float
    cash_ratio: float
    mean_final_soil: float
    ever_bankrupt: bool
    quarters_completed: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "normalised": self.normalised,
            "ending_cash": self.ending_cash,
            "starting_cash": self.starting_cash,
            "cash_ratio": self.cash_ratio,
            "mean_final_soil": self.mean_final_soil,
            "ever_bankrupt": self.ever_bankrupt,
            "quarters_completed": self.quarters_completed,
            **self.metadata,
        }


def _extract_terminal_state(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Pull key quantities from a logged trajectory."""
    final_state = trajectory.get("final_state", {})
    if not final_state:
        # Fall back to last snapshot if available
        snapshots = trajectory.get("snapshots", [])
        final_state = snapshots[-1] if snapshots else {}

    starting_cash = float(trajectory.get("task_spec", {}).get("starting_cash", 150_000.0))
    ending_cash = float(final_state.get("cash", 0.0))
    ever_bankrupt = bool(final_state.get("ever_bankrupt", trajectory.get("ever_bankrupt", False)))
    quarters = int(final_state.get("quarter", trajectory.get("quarters_completed", 0)))

    plots = final_state.get("plots", [])
    if plots:
        def _plot_health(p: Dict[str, Any]) -> float:
            om = p.get("_organic_matter", 0.55)
            st = p.get("_structure", 0.55)
            ph = p.get("_ph", 0.55)
            nu = p.get("_nutrient_balance", 0.55)
            return 0.45 * om + 0.20 * st + 0.15 * ph + 0.20 * nu
        mean_soil = float(np.mean([_plot_health(p) for p in plots]))
    else:
        mean_soil = float(trajectory.get("mean_final_soil", 0.55))

    return {
        "starting_cash": starting_cash,
        "ending_cash": ending_cash,
        "ever_bankrupt": ever_bankrupt,
        "quarters_completed": quarters,
        "mean_final_soil": mean_soil,
    }


def _soil_factor(mean_soil: float) -> float:
    clipped = float(np.clip(mean_soil, TERMINAL_SOIL_MIN, TERMINAL_SOIL_MAX))
    return (clipped - TERMINAL_SOIL_MIN) / (TERMINAL_SOIL_MAX - TERMINAL_SOIL_MIN)


# ── Grader 1: Scalar final score (matches simulator terminal_score) ───────────

class ScalarFinalScoreGrader:
    """
    Replicates the simulator's terminal_score:
      max(0, cash/start_cash) × soil_factor × solvency_gate
    This is the canonical terminal evaluation signal.
    """

    name = "scalar_final_score"

    def grade(self, trajectory: Dict[str, Any]) -> GradeResult:
        t = _extract_terminal_state(trajectory)
        cash_ratio = max(0.0, t["ending_cash"] / max(1.0, t["starting_cash"]))
        sf = _soil_factor(t["mean_final_soil"])
        solvency = 1.0 if not t["ever_bankrupt"] else SOLVENCY_GATE_PENALTY
        score = cash_ratio * sf * solvency

        return GradeResult(
            score=score,
            normalised=float(np.clip(score, 0.0, 1.0)),
            ending_cash=t["ending_cash"],
            starting_cash=t["starting_cash"],
            cash_ratio=cash_ratio,
            mean_final_soil=t["mean_final_soil"],
            ever_bankrupt=t["ever_bankrupt"],
            quarters_completed=t["quarters_completed"],
            metadata={"soil_factor": sf, "solvency_gate": solvency},
        )


# ── Grader 2: Bankruptcy-aware ────────────────────────────────────────────────

class BankruptcyAwareGrader:
    """
    Like ScalarFinalScoreGrader but with a harder bankruptcy penalty:
    - scores drop to 0.0 if ever_bankrupt AND final cash is still negative.
    - partial penalty (×0.2) if recovered from bankruptcy.
    - incomplete episodes (< 40 quarters) penalised by completion ratio.
    """

    name = "bankruptcy_aware"

    def grade(self, trajectory: Dict[str, Any]) -> GradeResult:
        from config import TOTAL_QUARTERS
        t = _extract_terminal_state(trajectory)
        cash_ratio = max(0.0, t["ending_cash"] / max(1.0, t["starting_cash"]))
        sf = _soil_factor(t["mean_final_soil"])

        if t["ever_bankrupt"] and t["ending_cash"] < 0.0:
            solvency = 0.0
        elif t["ever_bankrupt"]:
            solvency = SOLVENCY_GATE_PENALTY
        else:
            solvency = 1.0

        completion = t["quarters_completed"] / TOTAL_QUARTERS
        score = cash_ratio * sf * solvency * completion

        return GradeResult(
            score=score,
            normalised=float(np.clip(score, 0.0, 1.0)),
            ending_cash=t["ending_cash"],
            starting_cash=t["starting_cash"],
            cash_ratio=cash_ratio,
            mean_final_soil=t["mean_final_soil"],
            ever_bankrupt=t["ever_bankrupt"],
            quarters_completed=t["quarters_completed"],
            metadata={
                "soil_factor": sf,
                "solvency_gate": solvency,
                "completion": completion,
            },
        )


# ── Grader 3: Stewardship-weighted ───────────────────────────────────────────

class StewardshipWeightedGrader:
    """
    Puts 40% weight on soil stewardship and 60% on financial performance.
    Useful for detecting whether a model learned soil conservation.
    """

    name = "stewardship_weighted"
    SOIL_WEIGHT = 0.40
    CASH_WEIGHT = 0.60

    def grade(self, trajectory: Dict[str, Any]) -> GradeResult:
        t = _extract_terminal_state(trajectory)
        cash_ratio = max(0.0, t["ending_cash"] / max(1.0, t["starting_cash"]))
        sf = _soil_factor(t["mean_final_soil"])
        solvency = 1.0 if not t["ever_bankrupt"] else SOLVENCY_GATE_PENALTY

        financial_score = min(1.0, cash_ratio) * solvency
        soil_score = sf

        score = self.CASH_WEIGHT * financial_score + self.SOIL_WEIGHT * soil_score

        return GradeResult(
            score=score,
            normalised=float(np.clip(score, 0.0, 1.0)),
            ending_cash=t["ending_cash"],
            starting_cash=t["starting_cash"],
            cash_ratio=cash_ratio,
            mean_final_soil=t["mean_final_soil"],
            ever_bankrupt=t["ever_bankrupt"],
            quarters_completed=t["quarters_completed"],
            metadata={
                "financial_score": financial_score,
                "soil_factor": sf,
                "soil_score": soil_score,
                "solvency_gate": solvency,
            },
        )


# ── Registry ──────────────────────────────────────────────────────────────────

GRADERS = {
    ScalarFinalScoreGrader.name:     ScalarFinalScoreGrader,
    BankruptcyAwareGrader.name:      BankruptcyAwareGrader,
    StewardshipWeightedGrader.name:  StewardshipWeightedGrader,
}

DEFAULT_GRADER = ScalarFinalScoreGrader.name


def grade(trajectory: Dict[str, Any], grader_name: str = DEFAULT_GRADER) -> GradeResult:
    cls = GRADERS.get(grader_name)
    if cls is None:
        raise ValueError(f"Unknown grader '{grader_name}'. Available: {list(GRADERS)}")
    return cls().grade(trajectory)
