"""
Summarise and compare baseline results.

Usage:
  python eval/summarize_results.py
  python eval/summarize_results.py --results-dir eval/results --plot

Compares:
  - scripted baselines (greedy_extractor, conservative_rotation, weather_aware_rotation)
  - pre-training model (if eval/model_trajectories/ exists)
  - post-training model (same directory, labelled separately)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_dir: Path, split: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load all JSON result files for a given split."""
    combined: Dict[str, List[Dict[str, Any]]] = {}
    for fp in sorted(results_dir.glob(f"{split}_*.json")):
        name = fp.stem.replace(f"{split}_", "")
        with open(fp) as f:
            data = json.load(f)
        combined[name] = data if isinstance(data, list) else [data]
    return combined


def summarise_group(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    scores = [r.get("score", r.get("terminal_score", 0.0)) for r in results]
    soils = [r.get("mean_final_soil", 0.0) for r in results]
    cash_ratios = [r.get("cash_ratio", 0.0) for r in results]
    bankruptcies = [bool(r.get("ever_bankrupt", False)) for r in results]
    quarters = [r.get("quarters_completed", 0) for r in results]

    n = len(scores)
    return {
        "n": n,
        "score_mean": round(sum(scores) / n, 4),
        "score_min": round(min(scores), 4),
        "score_max": round(max(scores), 4),
        "soil_mean": round(sum(soils) / n, 4),
        "cash_ratio_mean": round(sum(cash_ratios) / n, 4),
        "bankruptcy_rate": round(sum(bankruptcies) / n, 4),
        "mean_quarters": round(sum(quarters) / n, 1),
    }


def print_comparison_table(all_results: Dict[str, List[Dict[str, Any]]]) -> None:
    header = f"{'Baseline':<30} {'n':>4} {'score_mean':>10} {'soil_mean':>10} {'cash_ratio':>10} {'bankrupt%':>10} {'q_complete':>10}"
    print("\n" + header)
    print("-" * len(header))
    for name, results in sorted(all_results.items()):
        s = summarise_group(results)
        if not s:
            continue
        print(
            f"{name:<30} {s['n']:>4} {s['score_mean']:>10.3f} "
            f"{s['soil_mean']:>10.3f} {s['cash_ratio_mean']:>10.3f} "
            f"{s['bankruptcy_rate']*100:>9.1f}% {s['mean_quarters']:>10.1f}"
        )


def maybe_plot(all_results: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    names = list(all_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    metrics = [
        ("score", "Terminal Score", axes[0]),
        ("mean_final_soil", "Mean Final Soil Health", axes[1]),
        ("cash_ratio", "Cash Ratio (end/start)", axes[2]),
    ]

    for metric, title, ax in metrics:
        values = []
        labels = []
        for name, results in all_results.items():
            vals = [r.get(metric, 0.0) for r in results if metric in r]
            if vals:
                values.append(vals)
                labels.append(name.replace("_", "\n"))

        if values:
            bp = ax.boxplot(values, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
            ax.set_title(title, fontsize=10)
            ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("uk_arable_manager — Baseline Comparison", fontsize=12)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise baseline results")
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plots-dir", type=Path, default=Path("eval/plots"))
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Results dir {args.results_dir} not found.")
        print("Run eval/run_baselines.py first.")
        sys.exit(1)

    all_results = load_results(args.results_dir, args.split)
    if not all_results:
        print(f"No result files found for split={args.split} in {args.results_dir}")
        sys.exit(1)

    print(f"\n=== uk_arable_manager Baseline Summary (split={args.split}) ===")
    print_comparison_table(all_results)

    # Print per-scenario breakdown
    for name, results in sorted(all_results.items()):
        scenarios = {}
        for r in results:
            sc = r.get("scenario_type", "unknown")
            scenarios.setdefault(sc, []).append(r.get("score", 0.0))
        if any(len(v) > 1 for v in scenarios.values()):
            print(f"\n  {name} by scenario:")
            for sc, scores in sorted(scenarios.items()):
                print(f"    {sc:<20} n={len(scores):>3}  mean={sum(scores)/len(scores):.3f}")

    if args.plot:
        maybe_plot(all_results, args.plots_dir)


if __name__ == "__main__":
    main()
