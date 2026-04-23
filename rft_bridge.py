"""
RFT bridge: converts ORS episode trajectories into OpenAI RFT-ready artifacts.

This module handles the complete pipeline:
  1. Load trajectories from disk
  2. Grade each episode with the grader module
  3. Produce training/validation JSONL for OpenAI Files API
  4. Produce a grader script for the RFT job
  5. Build a job manifest ready for launch
  6. Upload artifacts and create job (when OPENAI_API_KEY is set)

Honest note on what happens outside this repo:
  - Dataset upload:     this repo handles it via the OpenAI Files API.
  - Policy update:      happens inside OpenAI's managed RFT infrastructure.
  - Rollout collection: ORS episodes are the source of trajectories; this
                        bridge consumes trajectories already logged to disk.
  - Post-training eval: scripts/run_rft_eval.py runs the graded eval harness.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from grader import GRADERS, DEFAULT_GRADER, grade
from openai_rft_config import (
    GRADER_SCRIPT_TEMPLATE,
    RFT_MODEL,
    RFTJobManifest,
    RFTTrainingExample,
)
from trajectory_logger import TrajectoryLog


# ── Trajectory-to-training-example conversion ────────────────────────────────

def trajectory_to_training_example(
    traj: TrajectoryLog,
    grader_name: str = DEFAULT_GRADER,
    model_transcript: Optional[str] = None,
) -> RFTTrainingExample:
    """
    Convert a trajectory log into one RFT training example.

    The assistant message is the full rollout transcript (all tool calls
    and responses across the 40 quarters).  The grader receives the
    terminal_score in metadata so it doesn't need to parse the text.

    If model_transcript is None we reconstruct a synthetic transcript from
    the logged actions and observations.  For real RFT training you should
    capture the actual model output during rollout.
    """
    spec = traj.task_spec
    grade_result = grade(traj.to_dict(), grader_name)

    system_msg = (
        "You are managing a 400-acre arable farm in Cambridgeshire over 10 years. "
        "Use the available tools each quarter to observe the farm and commit a plan."
    )

    user_msg_lines = [
        "Farm management task.",
        f"Task ID: {spec.get('task_id', 'unknown')}",
        f"Starting cash: £{spec.get('starting_cash', 150000):,.0f}",
        f"Scenario: {spec.get('scenario_type', 'standard')}",
        "Manage the farm for 40 quarters to maximise the terminal score.",
    ]
    user_msg = "\n".join(user_msg_lines)

    if model_transcript is not None:
        assistant_content = model_transcript
    else:
        # Reconstruct transcript from logged steps
        parts = []
        for step in traj.steps:
            action = step.action
            cap = action.get("capital_action", "none")
            parts.append(f"[Q{step.quarter+1}] capital={cap}")
            for i, pa in enumerate(action.get("plots", [])):
                parts.append(
                    f"  plot_{i+1}: crop={pa.get('crop')} fert={pa.get('fertiliser')} "
                    f"pest={pa.get('pest_control')} → pnl=£{step.plot_pnl[i]:,.0f}"
                )
            parts.append(f"  reward={step.reward:.4f}  cash=£{step.pnl:,.0f}")
        parts.append(f"TERMINAL SCORE: {traj.terminal_score:.4f}")
        assistant_content = "\n".join(parts)

    metadata: Dict[str, Any] = {
        "task_id": spec.get("task_id", "unknown"),
        "terminal_score": grade_result.score,
        "normalised_score": grade_result.normalised,
        "ending_cash": grade_result.ending_cash,
        "starting_cash": grade_result.starting_cash,
        "mean_final_soil": grade_result.mean_final_soil,
        "ever_bankrupt": grade_result.ever_bankrupt,
        "quarters_completed": grade_result.quarters_completed,
        "grader_variant": grader_name,
        "scenario_type": spec.get("scenario_type", "standard"),
        "split": spec.get("split", "train"),
    }

    return RFTTrainingExample(
        messages=[
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_content},
        ],
        metadata=metadata,
    )


# ── Batch artifact builder ────────────────────────────────────────────────────

def build_rft_artifacts(
    trajectory_dir: Path,
    output_dir: Path,
    grader_name: str = DEFAULT_GRADER,
    splits: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Build JSONL artifacts for all requested splits.
    Returns dict mapping split name → output JSONL path.
    """
    if splits is None:
        splits = ["train", "validation", "test"]

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Path] = {}

    for split in splits:
        split_dir = trajectory_dir / split
        if not split_dir.exists():
            print(f"  [rft_bridge] No trajectory dir for split={split}, skipping.")
            continue

        # Support both flat layout (split/*.json) and per-baseline layout (split/baseline/*.json)
        traj_files = sorted(split_dir.glob("*.json"))
        if not traj_files:
            traj_files = sorted(split_dir.glob("*/*.json"))
        if not traj_files:
            print(f"  [rft_bridge] No trajectories found in {split_dir}")
            continue

        examples: List[Dict[str, Any]] = []
        for fp in traj_files:
            try:
                traj = TrajectoryLog.load(fp)
                ex = trajectory_to_training_example(traj, grader_name)
                examples.append(ex.to_dict())
            except Exception as e:
                print(f"  [rft_bridge] Warning: could not process {fp.name}: {e}")

        out_path = output_dir / f"rft_{split}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        artifacts[split] = out_path
        print(f"  [rft_bridge] Wrote {len(examples)} examples → {out_path}")

    # Write grader script
    grader_path = output_dir / "grader.py"
    grader_path.write_text(GRADER_SCRIPT_TEMPLATE)
    artifacts["grader"] = grader_path
    print(f"  [rft_bridge] Grader script → {grader_path}")

    return artifacts


# ── OpenAI upload + job creation ──────────────────────────────────────────────

def upload_and_create_job(
    artifacts: Dict[str, Path],
    job_suffix: str = "uk-arable",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Upload artifacts to OpenAI Files API and create an RFT fine-tuning job.

    Set dry_run=True to build the manifest without making API calls.
    Requires OPENAI_API_KEY environment variable.

    Returns a dict with file IDs and job ID (or the dry-run manifest).
    """
    import openai

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not dry_run:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.  Export it or pass dry_run=True."
        )

    manifest = RFTJobManifest(model=RFT_MODEL, suffix=job_suffix)
    result: Dict[str, Any] = {"dry_run": dry_run, "model": RFT_MODEL}

    def _upload(path: Path, purpose: str) -> str:
        if dry_run:
            return f"[dry-run-file-id:{path.name}]"
        client = openai.OpenAI(api_key=api_key)
        with open(path, "rb") as fh:
            response = client.files.create(file=fh, purpose=purpose)
        return response.id

    # Upload training file
    if "train" in artifacts:
        fid = _upload(artifacts["train"], "fine-tune")
        manifest.training_file_id = fid
        result["training_file_id"] = fid
        print(f"  [rft_bridge] Uploaded training file → {fid}")

    # Upload validation file
    if "validation" in artifacts:
        fid = _upload(artifacts["validation"], "fine-tune")
        manifest.validation_file_id = fid
        result["validation_file_id"] = fid
        print(f"  [rft_bridge] Uploaded validation file → {fid}")

    # Load grader source (embedded inline in job payload, not uploaded)
    grader_source = ""
    if "grader" in artifacts:
        grader_source = artifacts["grader"].read_text()
        print(f"  [rft_bridge] Grader script embedded inline ({len(grader_source)} chars)")

    # Build correct payload for OpenAI SDK v2.x: method is a dict with nested reinforcement config
    payload: Dict[str, Any] = {
        "model": manifest.model,
        "training_file": manifest.training_file_id,
        "suffix": manifest.suffix,
        "method": {
            "type": "reinforcement",
            "reinforcement": {
                "grader": {
                    "type": "python",
                    "name": "uk_arable_scalar_final_score",
                    "source": grader_source,
                },
            },
        },
    }
    if manifest.validation_file_id:
        payload["validation_file"] = manifest.validation_file_id

    result["manifest"] = payload

    if not dry_run and manifest.training_file_id:
        client = openai.OpenAI(api_key=api_key)
        job = client.fine_tuning.jobs.create(**payload)
        result["job_id"] = job.id
        result["job_status"] = job.status
        print(f"  [rft_bridge] RFT job created → {job.id}  status={job.status}")
    elif dry_run:
        # Redact long source in dry-run print
        pretty = dict(payload)
        pretty["method"] = {**payload["method"], "reinforcement": {
            **payload["method"]["reinforcement"],
            "grader": {**payload["method"]["reinforcement"]["grader"], "source": f"<{len(grader_source)} chars>"},
        }}
        print(f"  [rft_bridge] DRY RUN — manifest:\n{json.dumps(pretty, indent=2)}")

    return result


# ── Eval artifact builder ─────────────────────────────────────────────────────

def build_eval_summary(
    trajectory_dir: Path,
    grader_name: str = DEFAULT_GRADER,
    splits: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Grade all trajectories and return a list of result dicts."""
    if splits is None:
        splits = ["validation", "test"]

    results = []
    for split in splits:
        split_dir = trajectory_dir / split
        if not split_dir.exists():
            continue
        fps = sorted(split_dir.glob("*.json")) or sorted(split_dir.glob("*/*.json"))
        for fp in fps:
            try:
                traj = TrajectoryLog.load(fp)
                g = grade(traj.to_dict(), grader_name)
                results.append({
                    "task_id": traj.task_spec.get("task_id", fp.stem),
                    "split": split,
                    "baseline": traj.baseline_name,
                    **g.to_dict(),
                })
            except Exception as e:
                print(f"  [rft_bridge] Warning: {fp.name}: {e}")

    return results
