"""
Build OpenAI RFT training/validation/evaluation JSONL artifacts from
logged episode trajectories.

Usage:
  # After running eval/run_baselines.py to generate trajectories:
  python scripts/build_rft_artifacts.py

  # With a specific grader and output dir:
  python scripts/build_rft_artifacts.py --grader bankruptcy_aware --out artifacts/rft

  # Dry-run upload (shows manifest, no API calls):
  python scripts/build_rft_artifacts.py --upload --dry-run

  # Real upload + job creation:
  python scripts/build_rft_artifacts.py --upload
  # (requires OPENAI_API_KEY env var)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grader import GRADERS, DEFAULT_GRADER
from rft_bridge import build_eval_summary, build_rft_artifacts, upload_and_create_job

DEFAULT_TRAJ_DIR = Path("eval/trajectories")
DEFAULT_OUT_DIR = Path("artifacts/rft")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenAI RFT artifacts")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--grader", default=DEFAULT_GRADER, choices=list(GRADERS))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--upload", action="store_true", help="Upload to OpenAI + create job")
    parser.add_argument("--dry-run", action="store_true", help="With --upload: show manifest only")
    args = parser.parse_args()

    print(f"\n=== Building RFT Artifacts ===")
    print(f"Trajectory dir : {args.traj_dir}")
    print(f"Output dir     : {args.out}")
    print(f"Grader         : {args.grader}")
    print(f"Splits         : {args.splits}")

    if not args.traj_dir.exists():
        print(f"\nTrajectory dir {args.traj_dir} does not exist.")
        print("Run eval/run_baselines.py first to generate trajectories.")
        sys.exit(1)

    artifacts = build_rft_artifacts(
        trajectory_dir=args.traj_dir,
        output_dir=args.out,
        grader_name=args.grader,
        splits=args.splits,
    )

    # Also write a graded eval summary
    summary = build_eval_summary(
        trajectory_dir=args.traj_dir,
        grader_name=args.grader,
        splits=["validation", "test"],
    )
    if summary:
        summary_path = args.out / "eval_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nEval summary → {summary_path}")
        # Quick stats
        scores = [r["score"] for r in summary]
        if scores:
            print(f"  n={len(scores)}  mean={sum(scores)/len(scores):.3f}  "
                  f"min={min(scores):.3f}  max={max(scores):.3f}")

    if args.upload:
        print("\n=== Uploading to OpenAI ===")
        result = upload_and_create_job(
            artifacts=artifacts,
            dry_run=args.dry_run,
        )
        out_path = args.out / "upload_result.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Upload result → {out_path}")
    else:
        print("\nSkipping upload (pass --upload to push to OpenAI)")


if __name__ == "__main__":
    main()
