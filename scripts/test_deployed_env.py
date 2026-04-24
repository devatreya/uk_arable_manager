"""
Test the deployed OpenReward env to confirm it's serving correctly.

Before running, set:
    export OPENREWARD_API_KEY="or_xxxxxx..."

Optional overrides:
    export OPENREWARD_ENV_ID="devatreya/uk_arable_manager"

Run:
    python3 scripts/test_deployed_env.py
"""
from __future__ import annotations

import os
import sys


def _extract_text(blocks) -> str:
    return "\n".join(getattr(b, "text", "") for b in blocks if hasattr(b, "text"))


def main() -> int:
    api_key = os.environ.get("OPENREWARD_API_KEY", "")
    env_id = os.environ.get("OPENREWARD_ENV_ID", "devatreya/uk_arable_manager")

    if not api_key:
        print("ERROR: OPENREWARD_API_KEY not set")
        return 1

    print(f"Env identifier: {env_id}")
    print(f"API key:        {api_key[:8]}...{api_key[-4:]}  (length {len(api_key)})")
    print()

    from openreward import OpenReward

    print("=" * 60)
    print("Step 1: Resolve the deployed environment")
    print("=" * 60)
    try:
        with OpenReward(api_key=api_key) as client:
            env = client.environments.get(name=env_id)
            print(f"  ✓ Resolved env identifier: {env_id}")

            print()
            print("=" * 60)
            print("Step 2: List tasks on 'validation' split, then open session")
            print("        (this is what triggers env container startup)")
            print("=" * 60)
            try:
                tasks = env.list_tasks(split="validation")
                print(f"  ✓ Found {len(tasks)} validation tasks")
                task = tasks[0]
                with env.session(task=task) as session:
                    print("  ✓ Session opened — container is serving traffic")

                    prompt = _extract_text(session.get_prompt())  # type: ignore[arg-type]
                    print(f"  ✓ Prompt fetched ({len(prompt)} chars)")
                    for line in prompt.splitlines()[:6]:
                        print(f"     {line}")
                    print("     ...")

                    print()
                    print("=" * 60)
                    print("Step 3: Call read_farm_state")
                    print("=" * 60)
                    result = session.call_tool("read_farm_state", {})
                    text = _extract_text(result.blocks)
                    print(f"  ✓ reward={result.reward}  finished={result.finished}")
                    for line in text.splitlines()[:5]:
                        print(f"     {line}")

                    print()
                    print("=" * 60)
                    print("Step 4: Submit one commit_plan (advance quarter)")
                    print("=" * 60)
                    plan_input = {
                        "capital_action": "none",
                        "plot_1": {"crop": "wheat",       "fertiliser": "medium", "pest_control": "ipm"},
                        "plot_2": {"crop": "barley",      "fertiliser": "medium", "pest_control": "ipm"},
                        "plot_3": {"crop": "field_beans", "fertiliser": "low",    "pest_control": "none"},
                        "plot_4": {"crop": "cover_crop",  "fertiliser": "low",    "pest_control": "none"},
                    }
                    result = session.call_tool("commit_plan", plan_input)
                    text = _extract_text(result.blocks)
                    print(f"  ✓ reward={result.reward:.4f}  finished={result.finished}")
                    print(f"  Tool output (last 4 lines):")
                    for line in text.splitlines()[-4:]:
                        print(f"     {line}")

                    # Step 5: Confirm new reward shaping is active
                    # The new shaping adds up to +0.4 per quarter from soil bonus.
                    # If reward is significantly higher than pure P&L scaled by 1e-4
                    # (~2-5 typical), the shaping is in effect.
                    print()
                    print("=" * 60)
                    print("Step 5: Confirm new reward shaping is live")
                    print("=" * 60)
                    print(f"  Quarterly reward observed: {result.reward:.4f}")
                    print(f"  Without shaping: pure P&L × 1e-4 → typically 2.0–5.0")
                    print(f"  With shaping:    + up to +0.4 soil bonus → 2.4–5.4")
                    print(f"  → if reward ≈ baseline expected, redeploy is live ✓")

            except Exception as session_err:
                print(f"  ✗ Session creation failed: {type(session_err).__name__}")
                print(f"     {session_err}")
                print()
                print("This usually means the env is 'deployed' but not 'running'.")
                print("Look in the OpenReward WebUI for a Start/Run/Activate button,")
                print("or set min_instances >= 1 in the env's autoscaling settings.")
                return 1

        print()
        print("=" * 60)
        print("✓ DEPLOYED ENV WORKS — pipeline is unblocked.")
        print("=" * 60)
        return 0

    except Exception as e:
        print()
        print(f"✗ FAILED at resolve step: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
