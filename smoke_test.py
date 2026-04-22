"""
Local smoke test for UKArableManager using the openreward.environments SDK directly.

No server process needed — instantiates the environment in-process, which exercises
the full tool contract (the same code path the hosted server invokes per session).

Run:
  python smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from env import (
    CommitPlanInput,
    PlotPlanInput,
    ReadSoilInput,
    ReadWeatherInput,
    UKArableManager,
)
from openreward.environments import ToolOutput


def _default_plan(quarter: int) -> CommitPlanInput:
    """Simple rotating plan: wheat/barley/osr/field_beans cycling through quarters."""
    rotation = [
        ("wheat",        "oilseed_rape", "field_beans", "barley"),
        ("barley",       "field_beans",  "wheat",       "oilseed_rape"),
        ("oilseed_rape", "wheat",        "barley",       "field_beans"),
        ("field_beans",  "barley",       "oilseed_rape", "wheat"),
    ]
    crops = rotation[quarter % 4]
    plot = lambda crop: PlotPlanInput(crop=crop, fertiliser="medium", pest_control="ipm")
    return CommitPlanInput(
        capital_action="none",
        plot_1=plot(crops[0]),
        plot_2=plot(crops[1]),
        plot_3=plot(crops[2]),
        plot_4=plot(crops[3]),
    )


def _text(result: ToolOutput) -> str:
    return "\n".join(b.text for b in result.blocks if hasattr(b, "text"))


def run_smoke_test() -> None:
    print("=" * 60)
    print("uk_arable_manager — OpenReward smoke test")
    print("=" * 60)

    # ── 1. List splits ─────────────────────────────────────────────
    splits = UKArableManager.list_splits()
    print(f"\n[1] list_splits → {[s.name for s in splits]}")
    assert {s.name for s in splits} == {"train", "validation", "test"}

    # ── 2. List tasks ──────────────────────────────────────────────
    tasks = UKArableManager.list_tasks("validation")
    print(f"[2] list_tasks('validation') → {len(tasks)} tasks")
    assert len(tasks) > 0
    task_spec = dict(tasks[0])
    print(f"    task_id={task_spec['task_id']}  scenario={task_spec['scenario_type']}  "
          f"cash=£{task_spec['starting_cash']:,.0f}")

    # ── 3. Create session (instantiate env) ────────────────────────
    env = UKArableManager(task_spec=task_spec)
    env.setup()
    print(f"\n[3] Session created  →  FarmSimulator seeded with {task_spec['seed']}")

    # ── 4. Read prompt ─────────────────────────────────────────────
    prompt_blocks = env.get_prompt()
    prompt_text = _text(ToolOutput(blocks=prompt_blocks, reward=None, finished=False))
    print(f"\n[4] get_prompt  ({len(prompt_text)} chars)")
    # Show first 8 lines
    for line in prompt_text.splitlines()[:8]:
        print(f"    {line}")
    print("    ...")

    # ── 5. Read-only state tools ───────────────────────────────────
    print("\n[5] State tools")

    r = env.read_farm_state()
    assert isinstance(r, ToolOutput) and r.reward == 0.0 and not r.finished
    print(f"    read_farm_state      → reward={r.reward}  finished={r.finished}")
    print(f"    {_text(r).splitlines()[0]}")

    r = env.read_soil_report(ReadSoilInput(plots=[0, 1, 2, 3]))
    assert isinstance(r, ToolOutput) and r.reward == 0.0 and not r.finished
    print(f"    read_soil_report     → reward={r.reward}  finished={r.finished}")

    r = env.read_weather_history(ReadWeatherInput(lookback_quarters=12))
    assert isinstance(r, ToolOutput) and r.reward == 0.0 and not r.finished
    print(f"    read_weather_history → reward={r.reward}  finished={r.finished}  "
          f"({len(_text(r).splitlines())} lines)")

    r = env.read_price_board()
    assert isinstance(r, ToolOutput) and r.reward == 0.0 and not r.finished
    print(f"    read_price_board     → reward={r.reward}  finished={r.finished}")

    # ── 6. Full 40-quarter rollout ─────────────────────────────────
    print("\n[6] Full 40-quarter rollout")
    total_reward = 0.0
    quarters_run = 0

    for q in range(42):  # max 42 to catch bugs
        plan = _default_plan(q)
        result = env.commit_plan(plan)
        assert isinstance(result, ToolOutput)
        total_reward += result.reward or 0.0
        quarters_run += 1

        if q < 3 or result.finished:
            regime = env.sim.state.weather_history[-1].regime if env.sim else "?"
            print(f"    Q{q+1:02d}: reward={result.reward:+.4f}  regime={regime:<7}  "
                  f"cash=£{env.sim.state.cash:,.0f}  finished={result.finished}")

        if result.finished:
            terminal = result.reward  # last reward includes terminal score
            break
    else:
        print("    ERROR: episode did not finish in 42 steps")
        sys.exit(1)

    assert quarters_run == 40, f"Expected 40 quarters, ran {quarters_run}"
    print(f"\n    Quarters run: {quarters_run}/40  ✓")
    print(f"    Total reward: {total_reward:.4f}")
    print(f"    Terminal score (last reward): {terminal:.4f}")

    env.teardown()

    print("\n" + "=" * 60)
    print("Smoke test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
