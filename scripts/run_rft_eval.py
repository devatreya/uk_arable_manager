"""
Run post-training evaluation of o4-mini against the ORS environment.

This script:
  1. Loads validation/test tasks
  2. Runs o4-mini against a live ORS server, allowing it to call farm tools
  3. Logs trajectories
  4. Grades with all three graders
  5. Compares against scripted baseline results

Prerequisites:
  - ORS server running: python app.py
  - OPENAI_API_KEY set
  - Baseline results exist in eval/results/

Usage:
  python scripts/run_rft_eval.py --split validation --model o4-mini-2025-04-16
  python scripts/run_rft_eval.py --split test --fine-tuned-model ft:o4-mini-2025-04-16:uk-arable:xxx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TASK_FILES, TOTAL_QUARTERS
from grader import GRADERS, DEFAULT_GRADER, grade
from openai_rft_config import RFT_MODEL
from trajectory_logger import TrajectoryLogger


DEFAULT_ORS_URL = "http://localhost:8080"


def run_model_on_task(
    task_spec: Dict[str, Any],
    model: str,
    ors_url: str,
    api_key: str,
    max_quarters: int = TOTAL_QUARTERS,
) -> Dict[str, Any]:
    """
    Run an OpenAI model against the ORS environment for one task.
    The model calls farm tools via the OpenAI tool-use API against the ORS server.

    IMPORTANT: This function connects two systems:
      - ORS server (farm environment, tool execution)
      - OpenAI API (model inference)

    The model receives the farm prompt and must call tools iteratively.
    Each commit_plan call with finished=True ends the episode.
    """
    import openai
    import httpx

    client = openai.OpenAI(api_key=api_key)
    logger = TrajectoryLogger(task_spec, baseline_name=f"model:{model}")

    # Create ORS session
    ors_client = httpx.Client(base_url=ors_url, timeout=30.0)
    session_id = f"eval-{task_spec['task_id']}-{int(time.time())}"

    try:
        # Create session
        sess_resp = ors_client.post(
            f"/environments/UKArableManager/sessions",
            json={"task_spec": task_spec},
            headers={"X-Session-ID": session_id},
        )
        sess_resp.raise_for_status()

        # Get prompt
        prompt_resp = ors_client.get(
            f"/environments/UKArableManager/sessions/prompt",
            headers={"X-Session-ID": session_id},
        )
        prompt_resp.raise_for_status()
        prompt_data = prompt_resp.json()
        system_prompt = prompt_data.get("blocks", [{}])[0].get("text", "Manage the farm.")

        # Get tool specs
        tools_resp = ors_client.get("/environments/UKArableManager/tools")
        tools_resp.raise_for_status()
        ors_tools = tools_resp.json().get("tools", [])

        # Convert ORS tools to OpenAI tool format
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            for t in ors_tools
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Begin managing the farm. Task ID: {task_spec['task_id']}"},
        ]

        quarter = 0
        finished = False
        total_reward = 0.0
        final_observation = ""

        while quarter < max_quarters and not finished:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_unset=True))

            if not msg.tool_calls:
                # Model responded without tool calls — treat as end
                break

            tool_results = []
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                # Call tool on ORS server
                ors_resp = ors_client.post(
                    f"/environments/UKArableManager/sessions/tools",
                    json={"name": tool_name, "input": tool_input},
                    headers={"X-Session-ID": session_id},
                )
                ors_resp.raise_for_status()
                ors_result = ors_resp.json()

                tool_output_text = ""
                step_reward = 0.0
                step_finished = False

                if ors_result.get("ok"):
                    out = ors_result["output"]
                    tool_output_text = " ".join(b["text"] for b in out.get("blocks", []) if "text" in b)
                    step_reward = out.get("reward") or 0.0
                    step_finished = out.get("finished", False)
                else:
                    tool_output_text = f"Error: {ors_result.get('error', 'unknown')}"

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output_text,
                })

                if tool_name == "commit_plan":
                    total_reward += step_reward
                    logger.record_step(
                        quarter=quarter,
                        action={"tool": tool_name, "input": tool_input},
                        reward=float(step_reward),
                        pnl=0.0,
                        observation=tool_output_text,
                        weather={},
                        plot_pnl=[0.0] * 4,
                        bankrupt=False,
                        pest_pressure=[False] * 4,
                    )
                    quarter += 1
                    final_observation = tool_output_text
                    if step_finished:
                        finished = True

            messages.extend(tool_results)

    finally:
        try:
            ors_client.delete(
                f"/environments/UKArableManager/sessions",
                headers={"X-Session-ID": session_id},
            )
        except Exception:
            pass
        ors_client.close()

    traj = logger.finalise(final_state={}, terminal_score=None)
    return traj.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run o4-mini eval against ORS environment")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--model", default=RFT_MODEL)
    parser.add_argument("--fine-tuned-model", default=None, help="Override with fine-tuned model ID")
    parser.add_argument("--ors-url", default=DEFAULT_ORS_URL)
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--out-dir", type=Path, default=Path("eval/model_trajectories"))
    parser.add_argument("--grader", default=DEFAULT_GRADER, choices=list(GRADERS))
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    model = args.fine_tuned_model or args.model
    print(f"\n=== RFT Evaluation ===")
    print(f"Model  : {model}")
    print(f"Split  : {args.split}")
    print(f"ORS URL: {args.ors_url}")

    task_file = TASK_FILES.get(args.split)
    if not task_file or not task_file.exists():
        print(f"Task file not found: {task_file}  — run scripts/build_tasks.py first")
        sys.exit(1)

    with open(task_file) as f:
        tasks = json.load(f)

    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    out_dir = args.out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, task in enumerate(tasks):
        task_id = task.get("task_id", f"task_{i}")
        print(f"  [{i+1}/{len(tasks)}] {task_id} ... ", end="", flush=True)
        try:
            traj = run_model_on_task(task, model, args.ors_url, api_key)
            out_path = out_dir / f"{task_id}.json"
            with open(out_path, "w") as f:
                json.dump(traj, f, indent=2)
            g = grade(traj, args.grader)
            results.append(g.to_dict())
            print(f"score={g.score:.3f}  bankrupt={g.ever_bankrupt}")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"task_id": task_id, "error": str(e)})

    # Summary
    valid = [r for r in results if "score" in r]
    if valid:
        scores = [r["score"] for r in valid]
        print(f"\nResults: n={len(valid)}  mean={sum(scores)/len(scores):.3f}  "
              f"min={min(scores):.3f}  max={max(scores):.3f}")
        bankrupt_rate = sum(1 for r in valid if r.get("ever_bankrupt")) / len(valid)
        print(f"Bankruptcy rate: {bankrupt_rate:.1%}")

    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
