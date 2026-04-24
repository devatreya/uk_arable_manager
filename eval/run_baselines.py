"""
Run all scripted baselines against all tasks in a given split.

Usage:
  python eval/run_baselines.py                    # validation split
  python eval/run_baselines.py --split train      # train split
  python eval/run_baselines.py --split test --baselines greedy_extractor conservative_rotation

Outputs:
  eval/trajectories/<split>/<baseline>/<task_id>.json
  eval/results/<split>_<baseline>_summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TASK_FILES
from pipeline.config import DEFAULT_OPENREWARD_ENV_ID, DEFAULT_SESSION_BACKEND
from rollout_client import HostedRolloutManager, run_local_rollout
from baselines import BASELINES
from grader import grade, DEFAULT_GRADER


async def _run_hosted_baseline_on_split(
    *,
    baseline_name: str,
    tasks: list[dict],
    out_dir: Path,
    openreward_env_id: str,
    grader_name: str,
    hosted_concurrency: int,
) -> list:
    policy = BASELINES[baseline_name]
    results = []
    semaphore = asyncio.Semaphore(max(1, hosted_concurrency))

    async def _run_task(i: int, task_spec: dict) -> tuple[int, dict, Any]:
        task_id = task_spec.get("task_id", f"task_{i:04d}")
        out_path = out_dir / f"{task_id}.json"
        async with semaphore:
            traj = await manager.run(
                task_spec,
                policy,
                baseline_name=baseline_name,
                save_to=out_path,
            )
        return i, task_spec, traj

    async with HostedRolloutManager(env_id=openreward_env_id) as manager:
        pending = [_run_task(i, task_spec) for i, task_spec in enumerate(tasks)]
        for completed, future in enumerate(asyncio.as_completed(pending), start=1):
            i, task_spec, traj = await future
            task_id = task_spec.get("task_id", f"task_{i:04d}")
            g = grade(traj.to_dict(), grader_name)
            result = {
                "task_id": task_id,
                "baseline": baseline_name,
                "split": str(task_spec.get("split", "unknown")),
                **g.to_dict(),
            }
            results.append(result)
            if completed % 5 == 0 or completed == len(tasks):
                scores = [r["score"] for r in results]
                print(f"    {baseline_name}/{task_spec.get('split', 'unknown')} [{completed}/{len(tasks)}]  "
                      f"mean_score={sum(scores)/len(scores):.3f}")
    results.sort(key=lambda row: row["task_id"])
    return results


def run_baseline_on_split(
    baseline_name: str,
    split: str,
    out_base: Path,
    grader_name: str = DEFAULT_GRADER,
    max_tasks: int = None,
    save_snapshots: bool = False,
    session_backend: str = DEFAULT_SESSION_BACKEND,
    openreward_env_id: str = DEFAULT_OPENREWARD_ENV_ID,
    hosted_concurrency: int = 4,
) -> list:
    task_file = TASK_FILES.get(split)
    if not task_file or not task_file.exists():
        print(f"  Task file not found: {task_file}")
        print(f"  Run: python scripts/build_tasks.py first")
        return []

    with open(task_file) as f:
        tasks = json.load(f)

    if max_tasks:
        tasks = tasks[:max_tasks]

    policy = BASELINES[baseline_name]
    out_dir = out_base / split / baseline_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if session_backend == "hosted":
        return asyncio.run(
            _run_hosted_baseline_on_split(
                baseline_name=baseline_name,
                tasks=tasks,
                out_dir=out_dir,
                openreward_env_id=openreward_env_id,
                grader_name=grader_name,
                hosted_concurrency=hosted_concurrency,
            )
        )

    for i, task_spec in enumerate(tasks):
        task_id = task_spec.get("task_id", f"task_{i:04d}")
        out_path = out_dir / f"{task_id}.json"

        traj = run_local_rollout(
            task_spec=task_spec,
            policy=policy,
            baseline_name=baseline_name,
            save_to=out_path,
            save_snapshots=save_snapshots,
        )

        g = grade(traj.to_dict(), grader_name)
        result = {
            "task_id": task_id,
            "baseline": baseline_name,
            "split": split,
            **g.to_dict(),
        }
        results.append(result)

        if (i + 1) % 5 == 0 or i == len(tasks) - 1:
            scores = [r["score"] for r in results]
            print(f"    {baseline_name}/{split} [{i+1}/{len(tasks)}]  "
                  f"mean_score={sum(scores)/len(scores):.3f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scripted baselines")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--baselines", nargs="+", default=["weather_aware_rotation"])
    parser.add_argument("--grader", default=DEFAULT_GRADER)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--traj-dir", type=Path, default=Path("eval/trajectories"))
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--session-backend", choices=["hosted", "inprocess"], default=DEFAULT_SESSION_BACKEND)
    parser.add_argument("--openreward-env-id", default=DEFAULT_OPENREWARD_ENV_ID)
    parser.add_argument("--hosted-concurrency", type=int, default=4)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running Baselines on {args.split} split ===")
    print(f"Baselines: {args.baselines}")

    all_results = []
    for bname in args.baselines:
        if bname not in BASELINES:
            print(f"  Unknown baseline '{bname}', skipping.")
            continue
        print(f"\n  Baseline: {bname}")
        results = run_baseline_on_split(
            baseline_name=bname,
            split=args.split,
            out_base=args.traj_dir,
            grader_name=args.grader,
            max_tasks=args.max_tasks,
            save_snapshots=args.save_snapshots,
            session_backend=args.session_backend,
            openreward_env_id=args.openreward_env_id,
            hosted_concurrency=args.hosted_concurrency,
        )
        all_results.extend(results)

        if results:
            out_path = args.results_dir / f"{args.split}_{bname}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Results → {out_path}")

    # Combined summary
    if all_results:
        combined_path = args.results_dir / f"{args.split}_all_baselines.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results → {combined_path}")


if __name__ == "__main__":
    main()
