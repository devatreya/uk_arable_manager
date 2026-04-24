from __future__ import annotations

import inspect
import os
from pathlib import Path

import modal

from pipeline.config import MODAL_ART_ROOT, MODAL_HF_CACHE, MODAL_PROJECT_ROOT, MODAL_RESULTS_ROOT, PROJECT_ROOT

PROJECT_IGNORE = [
    ".git",
    ".git/**",
    ".claude",
    ".claude/**",
    ".pytest_cache",
    ".pytest_cache/**",
    "__pycache__",
    "**/__pycache__",
    "*.pyc",
    ".art",
    ".art/**",
    "tutorial_first_env_gsm8k",
    "tutorial_first_env_gsm8k/**",
    "tutorial_gsm8k",
    "tutorial_gsm8k/**",
    "eval/model_trajectories",
    "eval/model_trajectories/**",
]

ART_VOLUME = modal.Volume.from_name("uk-arable-art", create_if_missing=True)
HF_CACHE_VOLUME = modal.Volume.from_name("uk-arable-hf-cache", create_if_missing=True)
RESULTS_VOLUME = modal.Volume.from_name("uk-arable-results", create_if_missing=True)

RUNTIME_SECRET_ENV_NAMES = ("HF_TOKEN", "WANDB_API_KEY", "OPENREWARD_API_KEY")
_runtime_secret_values = {
    name: value
    for name, value in (
        (name, os.environ.get(name))
        for name in RUNTIME_SECRET_ENV_NAMES
    )
    if value
}
RUNTIME_SECRET = modal.Secret.from_dict(
    {
        **(_runtime_secret_values or {"UK_ARABLE_RUNTIME_SECRET": "configured-at-launch"})
    }
)

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "procps")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements.modal.txt"))
    .env(
        {
            "PYTHONPATH": MODAL_PROJECT_ROOT,
            "HF_HOME": MODAL_HF_CACHE,
            "WANDB_DIR": f"{MODAL_RESULTS_ROOT}/wandb",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .workdir(MODAL_PROJECT_ROOT)
    .add_local_dir(
        str(PROJECT_ROOT),
        remote_path=MODAL_PROJECT_ROOT,
        ignore=PROJECT_IGNORE,
        copy=True,
    )
)

VOLUMES = {
    MODAL_ART_ROOT: ART_VOLUME,
    MODAL_HF_CACHE: HF_CACHE_VOLUME,
    MODAL_RESULTS_ROOT: RESULTS_VOLUME,
}


def require_local_env(*names: str) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def commit_all_volumes() -> None:
    ART_VOLUME.commit()
    HF_CACHE_VOLUME.commit()
    RESULTS_VOLUME.commit()


async def commit_all_volumes_async() -> None:
    await ART_VOLUME.commit.aio()
    await HF_CACHE_VOLUME.commit.aio()
    await RESULTS_VOLUME.commit.aio()


def modal_result_path(*parts: str) -> Path:
    return Path(MODAL_RESULTS_ROOT, *parts)


async def maybe_aclose(resource: object) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result
