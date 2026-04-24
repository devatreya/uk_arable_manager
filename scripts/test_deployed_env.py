"""
Test the deployed OpenReward env to confirm it's serving correctly.

Uses the AsyncOpenReward pattern from the official OpenReward docs.

Requires:
    Python 3.11+
    openreward 0.1.34+

Before running, set:
    export OPENREWARD_API_KEY="or_xxxxxx..."

Optional overrides:
    export OPENREWARD_ENV_ID="devatreya/uk_arable_manager"

Run:
    python3 scripts/test_deployed_env.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from importlib.metadata import PackageNotFoundError, version


MIN_OPENREWARD_VERSION = (0, 1, 34)


def _version_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split(".") if part.isdigit())


def _configure_ssl_cert_file() -> str | None:
    if os.environ.get("SSL_CERT_FILE"):
        return os.environ["SSL_CERT_FILE"]

    try:
        import certifi
    except ImportError:
        return None

    os.environ["SSL_CERT_FILE"] = certifi.where()
    return os.environ["SSL_CERT_FILE"]


def _extract_text(blocks) -> str:
    return "\n".join(getattr(b, "text", "") for b in blocks if hasattr(b, "text"))


async def _resolve_task(environment, *, split: str, index: int):
    """Support both the docs API and older SDKs that only expose list_tasks()."""
    get_task = getattr(environment, "get_task", None)
    if callable(get_task):
        return await get_task(split=split, index=index)

    list_tasks = getattr(environment, "list_tasks", None)
    if not callable(list_tasks):
        raise AttributeError(
            "Environment exposes neither get_task(split, index) nor list_tasks(split)"
        )

    tasks = await list_tasks(split=split)
    if not tasks:
        raise RuntimeError(f"No tasks found for split {split!r}")

    try:
        return tasks[index]
    except IndexError as exc:
        raise IndexError(
            f"Split {split!r} has {len(tasks)} task(s), index {index} is out of range"
        ) from exc


async def run() -> int:
    api_key = os.environ.get("OPENREWARD_API_KEY", "")
    env_id = os.environ.get("OPENREWARD_ENV_ID", "devatreya/uk_arable_manager")

    if not api_key:
        print("ERROR: OPENREWARD_API_KEY not set")
        return 1

    if sys.version_info < (3, 11):
        print("ERROR: Python 3.11+ is required for the current OpenReward SDK.")
        print(f"Current Python: {sys.version.split()[0]}")
        print("Python 3.10 installs openreward 0.1.33, which uses deprecated session endpoints.")
        return 1

    try:
        sdk_version = version("openreward")
    except PackageNotFoundError:
        print("ERROR: openreward is not installed in this Python environment.")
        return 1

    if _version_tuple(sdk_version) < MIN_OPENREWARD_VERSION:
        print("ERROR: openreward 0.1.34+ is required for deployed env sessions.")
        print(f"Installed openreward: {sdk_version}")
        return 1

    ssl_cert_file = _configure_ssl_cert_file()

    print(f"Env identifier: {env_id}")
    print(f"API key:        {api_key[:8]}...{api_key[-4:]}  (length {len(api_key)})")
    print(f"Python:         {sys.version.split()[0]}")
    print(f"openreward:     {sdk_version}")
    if ssl_cert_file:
        print(f"SSL_CERT_FILE:  {ssl_cert_file}")
    print()

    from openreward import AsyncOpenReward

    print("=" * 60)
    print("Step 1: Resolve the deployed environment")
    print("=" * 60)

    client = AsyncOpenReward(api_key=api_key)

    try:
        environment = client.environments.get(name=env_id)
        print(f"  ✓ Resolved env identifier: {env_id}")

        print()
        print("=" * 60)
        print("Step 2: Fetch a validation task + open session")
        print("        (this is what triggers env container startup)")
        print("=" * 60)
        try:
            task = await _resolve_task(environment, split="validation", index=0)
            print("  ✓ Resolved validation task index=0")

            async with environment.session(task=task) as session:
                print("  ✓ Session opened — container is serving traffic")

                prompt_blocks = await session.get_prompt()
                prompt = _extract_text(prompt_blocks)
                print(f"  ✓ Prompt fetched ({len(prompt)} chars)")
                for line in prompt.splitlines()[:6]:
                    print(f"     {line}")
                print("     ...")

                print()
                print("=" * 60)
                print("Step 3: Call read_farm_state")
                print("=" * 60)
                result = await session.call_tool("read_farm_state", {})
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
                result = await session.call_tool("commit_plan", plan_input)
                text = _extract_text(result.blocks)
                print(f"  ✓ reward={result.reward:.4f}  finished={result.finished}")
                print(f"  Tool output (last 4 lines):")
                for line in text.splitlines()[-4:]:
                    print(f"     {line}")

                print()
                print("=" * 60)
                print("Step 5: Confirm new reward shaping is live")
                print("=" * 60)
                print(f"  Quarterly reward observed: {result.reward:.4f}")
                print(f"  Without shaping: pure P&L × 1e-4 → typically 2.0–5.0")
                print(f"  With shaping:    + up to +0.4 soil bonus → 2.4–5.4")
                print(f"  → if reward ≈ baseline expected, redeploy is live ✓")

        except Exception as session_err:
            message = str(session_err)
            print(f"  ✗ Session creation failed: {type(session_err).__name__}")
            print(f"     {message}")
            print()

            if "CERTIFICATE_VERIFY_FAILED" in message:
                print("This Python install is missing CA roots for OpenReward's sessions endpoint.")
                print("Re-run with SSL_CERT_FILE set to certifi's bundle, or run macOS' Install Certificates command.")
            elif "matrix.openreward.ai/create_session" in message:
                print("This usually means you're on an older OpenReward SDK/client path.")
                print("Use Python 3.11+ with openreward 0.1.34+.")
            elif "sessions.openreward.ai" in message and "/task" in message:
                print("The sessions service is reachable, but the deployed env server is returning 404 for task routes.")
                print("This usually means the deployed container is running an outdated OpenReward server SDK.")
                print("Redeploy after updating requirements.txt to openreward 0.1.105 on Python 3.11+.")
            else:
                print("This usually means the env is deployed but the hosted session path is still unhealthy.")
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
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            maybe = close()
            if asyncio.iscoroutine(maybe):
                await maybe


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
