"""
Test the deployed OpenReward env to confirm it's serving correctly.

Before running, set:
    export OPENREWARD_API_KEY="or_xxxxxx..."

Optional overrides (defaults assume openreward.ai standard routing):
    export OPENREWARD_NAMESPACE="devatreya"
    export OPENREWARD_ENV_NAME="uk_arable_manager"
    export OPENREWARD_BASE_URL="https://openreward.ai"

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
    namespace = os.environ.get("OPENREWARD_NAMESPACE", "devatreya")
    env_name = os.environ.get("OPENREWARD_ENV_NAME", "uk_arable_manager")
    base_url = os.environ.get("OPENREWARD_BASE_URL", "https://openreward.ai")

    if not api_key:
        print("ERROR: OPENREWARD_API_KEY not set")
        print("  export OPENREWARD_API_KEY='or_xxxxxxxxxxxx'")
        return 1

    print(f"Base URL:   {base_url}")
    print(f"Namespace:  {namespace}")
    print(f"Env name:   {env_name}")
    print(f"API key:    {api_key[:8]}...{api_key[-4:]}  (length {len(api_key)})")
    print()

    # The high-level OpenReward client auto-handles subdomain/path routing
    # for matrix-hosted environments at openreward.ai.
    from openreward import OpenReward

    print("=" * 60)
    print("Step 1: Connect, list splits and tasks")
    print("=" * 60)
    try:
        with OpenReward(api_key=api_key) as client:
            # Try a few common patterns for the env identifier — different
            # platform versions register the env under different names.
            candidates = [
                f"{namespace}/{env_name}",
                env_name,
                "UKArableManager",
                f"{namespace}/UKArableManager",
            ]
            env = None
            last_error = None
            for cand in candidates:
                try:
                    env = client.environments.get(cand)
                    print(f"  ✓ Found env via identifier: {cand!r}")
                    break
                except Exception as e:
                    last_error = e
                    print(f"  - tried {cand!r}: {type(e).__name__}")
            if env is None:
                raise RuntimeError(
                    f"Could not locate env. Last error: {last_error}\n"
                    f"Try setting OPENREWARD_NAMESPACE / OPENREWARD_ENV_NAME explicitly."
                )

            splits = env.list_splits()
            print(f"  ✓ Splits: {[s.name for s in splits]}")

            tasks = env.list_tasks("validation")
            print(f"  ✓ validation has {len(tasks)} tasks")
            if tasks:
                first = tasks[0].task_spec
                print(f"     first task_id: {first.get('task_id')}")
                print(f"     scenario:      {first.get('scenario_type')}")
                print(f"     real_data:     {first.get('real_data_mode')}")
                print(f"     starting_cash: £{first.get('starting_cash', 0):,.0f}")
            if len(tasks) < 5:
                print("  ⚠ Expected 16 validation tasks; got few — data files may not be mounted")

            # Step 2: Open a session, fetch the prompt
            print()
            print("=" * 60)
            print("Step 2: Start a session and fetch the prompt")
            print("=" * 60)
            with env.session(tasks[0]) as session:
                prompt_blocks = session.get_prompt()
                prompt_text = _extract_text(prompt_blocks)
                print(f"  ✓ Prompt length: {len(prompt_text)} chars")
                print(f"  First 6 lines of prompt:")
                for line in prompt_text.splitlines()[:6]:
                    print(f"     {line}")
                print(f"     ...")

                # Step 3: Call a read tool
                print()
                print("=" * 60)
                print("Step 3: Call read_farm_state")
                print("=" * 60)
                result = session.call_tool("read_farm_state", {})
                text = _extract_text(result.blocks)
                print(f"  ✓ reward={result.reward}  finished={result.finished}")
                print(f"  Tool output (first 5 lines):")
                for line in text.splitlines()[:5]:
                    print(f"     {line}")

                # Step 4: Submit one commit_plan to advance the quarter
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
                print(f"  Tool output (last 3 lines):")
                for line in text.splitlines()[-3:]:
                    print(f"     {line}")

        print()
        print("=" * 60)
        print("✓ DEPLOYED ENV WORKS — ready to point Modal training at it.")
        print("=" * 60)
        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        print("=" * 60)
        print()
        print("Common causes:")
        print("  - Env not deployed yet (check Deployments tab in WebUI)")
        print("  - Code changes not pushed (git push to trigger redeploy)")
        print("  - Wrong env identifier — check the URL or WebUI for the exact name")
        print("  - API key wrong / expired")
        print("  - Data files not at /orwd_data/ (re-upload via WebUI)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
