from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TRAINABLE_MODEL_NAME = "uk-arable-qwen25-7b"
TRAINABLE_PROJECT = "uk-arable-manager"

SFT_OUTPUT_DIR = "artifacts/sft"
SFT_TRAIN_FILE = f"{SFT_OUTPUT_DIR}/train.jsonl"
SFT_VALIDATION_FILE = f"{SFT_OUTPUT_DIR}/validation.jsonl"
TRAINING_OUTPUT_DIR = "artifacts/training"
EVAL_OUTPUT_DIR = "artifacts/eval"

MODAL_APP_NAME_PREFIX = "uk-arable"
MODAL_PROJECT_ROOT = "/root/project"
MODAL_ART_ROOT = "/vol/art"
MODAL_RESULTS_ROOT = "/vol/results"
MODAL_HF_CACHE = "/vol/hf-cache"
DEFAULT_OPENREWARD_ENV_ID = os.environ.get("OPENREWARD_ENV_ID", "devatreya/uk_arable_manager")
DEFAULT_SESSION_BACKEND = os.environ.get("UK_ARABLE_SESSION_BACKEND", "hosted")

DEFAULT_TOP_QUANTILE = 0.25
DEFAULT_READ_TOOL_SEQUENCE = (
    ("read_farm_state", {}),
    ("read_soil_report", {"plots": [0, 1, 2, 3]}),
    ("read_weather_history", {"lookback_quarters": 4}),
    ("read_price_board", {}),
)


@dataclass
class SFTJobConfig:
    model_name: str = TRAINABLE_MODEL_NAME
    project: str = TRAINABLE_PROJECT
    base_model: str = BASE_MODEL
    dataset_path: str = SFT_TRAIN_FILE
    epochs: int = 1
    batch_size: int = 4
    peak_lr: float = 2e-4
    warmup_ratio: float = 0.1
    schedule_type: str = "cosine"
    art_path: str = MODAL_ART_ROOT
    engine_gpu_memory_utilization: float = 0.8
    trainer_gpu_ids: list[int] | None = None
    inference_gpu_ids: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RLJobConfig:
    model_name: str = TRAINABLE_MODEL_NAME
    project: str = TRAINABLE_PROJECT
    base_model: str = BASE_MODEL
    split: str = "train"
    validation_split: str = "validation"
    train_steps: int = 8
    groups_per_step: int = 4
    trajectories_per_group: int = 4
    learning_rate: float = 1e-5
    loss_fn: str = "cispo"
    kl_penalty_coef: float = 0.0
    eval_every: int = 2
    max_tool_calls: int = 160
    max_train_tasks: int | None = None
    max_validation_tasks: int = 4
    temperature: float = 0.8
    max_completion_tokens: int = 512
    art_path: str = MODAL_ART_ROOT
    session_backend: str = DEFAULT_SESSION_BACKEND
    openreward_env_id: str = DEFAULT_OPENREWARD_ENV_ID
    trainer_gpu_ids: list[int] = field(default_factory=lambda: [0])
    inference_gpu_ids: list[int] = field(default_factory=lambda: [1])
    seed: int = 23

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalJobConfig:
    model_name: str = TRAINABLE_MODEL_NAME
    project: str = TRAINABLE_PROJECT
    base_model: str = BASE_MODEL
    split: str = "validation"
    max_tasks: int | None = None
    max_tool_calls: int = 160
    max_completion_tokens: int = 512
    temperature: float = 0.0
    art_path: str = MODAL_ART_ROOT
    session_backend: str = DEFAULT_SESSION_BACKEND
    openreward_env_id: str = DEFAULT_OPENREWARD_ENV_ID
    trainer_gpu_ids: list[int] = field(default_factory=lambda: [0])
    inference_gpu_ids: list[int] = field(default_factory=lambda: [1])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
