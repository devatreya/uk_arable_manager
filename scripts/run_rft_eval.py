"""
Run post-training evaluation of o4-mini against a live OpenReward server.

This script:
  1. Loads validation/test tasks
  2. Opens an OpenReward session via the openreward SDK
  3. Gets OpenAI-formatted tool specs directly from the session
  4. Lets the model call farm tools iteratively until the episode finishes
  5. Logs trajectories and grades them

Prerequisites:
  - Server running: python server.py  (or python app.py)
  - OPENAI_API_KEY set
  - Task files built: python scripts/build_tasks.py

Usage:
  python scripts/run_rft_eval.py --split validation --model o4-mini-2025-04-16
  python scripts/run_rft_eval.py --split test --fine-tuned-model ft:...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TASK_FILES, TOTAL_QUARTERS
from grader import GRADERS, DEFAULT_GRADER, grade
from openai_rft_config import RFT_MODEL
from trajectory_logger import TrajectoryLogger


DEFAULT_SERVER_URL = "http://localhost:8080"
ENV_NAME = "UKArableManager"


def _extract_text(tool_output: Any) -> str:
    blocks = getattr(tool_output, "blocks", None) or []
    return "\n".join(getattr(b, "text", "") for b in blocks if hasattr(b, "text"))


def run_model_on_task(
    task_spec: Dict[str, Any],
    model: str,
    server_url: str,
    api_key: str,
    max_tool_calls: int = 250,
) -> Dict[str, Any]:
    """
    Run an OpenAI model against the OpenReward server for one task.

    The model sees the farm prompt and calls tools iteratively via OpenAI's
    tool-use API.  Each commit_plan advances the episode by one quarter.
    """
    import openai
    from openreward import EnvironmentsAPI

    client = openai.OpenAI(api_key=api_key)
    logger = TrajectoryLogger(task_spec, baseline_name=f"model:{model}")

    quarter = 0
    total_reward = 0.0
    finished = False
    final_observation = ""

    with EnvironmentsAPI(base_url=server_url, api_key="local") as api:
        env = api.get(ENV_NAME, base_url=server_url)

        # Locate the matching Task (fall back to first task if id not found)
        tasks = env.list_tasks(task_spec.get("split", "validation"))
        task_id = task_spec.get("task_id")
        task = next(
            (t for t in tasks if t.task_spec.get("task_id") == task_id),
            tasks[0] if tasks else None,
        )
        if task is None:
            raise RuntimeError(f"No tasks for split {task_spec.get('split')!r}")

        with env.session(task) as session:
            # SDK emits Responses-API-style specs; wrap for Chat Completions
            raw_tools = session.list_tools(format="openai")
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
                for t in raw_tools
            ]

            prompt_blocks = session.get_prompt()
            system_prompt = "\n".join(
                getattr(b, "text", "") for b in prompt_blocks if hasattr(b, "text")
            ) or "Manage the farm."

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Begin managing the farm. Task ID: {task_spec.get('task_id','?')}. "
                               f"Call tools each quarter and commit_plan to advance time.",
                },
            ]

            calls_made = 0
            while not finished and quarter < TOTAL_QUARTERS and calls_made < max_tool_calls:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
                msg = response.choices[0].message
                messages.append(msg.model_dump(exclude_unset=True))

                if not msg.tool_calls:
                    break  # model gave text without calling a tool — end

                tool_results = []
                for tc in msg.tool_calls:
                    calls_made += 1
                    tool_name = tc.function.name
                    try:
                        tool_input = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        tool_input = {}

                    try:
                        out = session.call_tool(tool_name, tool_input)
                        text = _extract_text(out)
                        step_reward = float(out.reward or 0.0)
                        step_finished = bool(out.finished)
                    except Exception as e:
                        text = f"Tool error: {e}"
                        step_reward = 0.0
                        step_finished = False

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": text,
                    })

                    if tool_name == "commit_plan":
                        total_reward += step_reward
                        logger.record_step(
                            quarter=quarter,
                            action={"tool": tool_name, "input": tool_input},
                            reward=step_reward,
                            pnl=0.0,
                            observation=text,
                            weather={},
                            plot_pnl=[0.0, 0.0, 0.0, 0.0],
                            bankrupt=False,
                            pest_pressure=[False] * 4,
                        )
                        quarter += 1
                        final_observation = text
                        if step_finished:
                            finished = True

                messages.extend(tool_results)

    traj = logger.finalise(final_state={}, terminal_score=None)
    d = traj.to_dict()
    d["model_total_reward"] = total_reward
    d["final_observation"] = final_observation
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Run o4-mini eval vs OpenReward env")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--model", default=RFT_MODEL)
    parser.add_argument("--fine-tuned-model", default=None, help="Override with fine-tuned model ID")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
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
    print(f"Server : {args.server_url}")

    task_file = TASK_FILES.get(args.split)
    if not task_file or not task_file.exists():
        print(f"Task file not found: {task_file}  — run scripts/build_tasks.py first")
        sys.exit(1)

    with open(task_file) as f:
        tasks = json.load(f)

    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    out_dir = args.out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks):
        task_id = task.get("task_id", f"task_{i}")
        print(f"  [{i+1}/{len(tasks)}] {task_id} ... ", end="", flush=True)
        try:
            traj = run_model_on_task(task, model, args.server_url, api_key)
            out_path = out_dir / f"{task_id}.json"
            with open(out_path, "w") as f:
                json.dump(traj, f, indent=2)
            g = grade(traj, args.grader)
            results.append(g.to_dict())
            print(f"score={g.score:.3f}  bankrupt={g.ever_bankrupt}")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"task_id": task_id, "error": str(e)})

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
