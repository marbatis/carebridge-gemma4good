from __future__ import annotations

import json
import os
import platform
import subprocess
from functools import lru_cache
from glob import glob
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _try_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"[error] {exc}"


def _force_cpu() -> bool:
    return _bool_env("CAREBRIDGE_FORCE_CPU")


def find_gemma_model_ref() -> str:
    explicit = os.getenv("CAREBRIDGE_GEMMA_MODEL")
    if explicit:
        return explicit

    kaggle_model_configs = sorted(glob("/kaggle/input/models/**/config.json", recursive=True))
    if kaggle_model_configs:
        return str(Path(kaggle_model_configs[0]).parent)

    general_gemma_configs = sorted(glob("/kaggle/input/**/config.json", recursive=True))
    for config_path in general_gemma_configs:
        if "gemma" in config_path.lower():
            return str(Path(config_path).parent)

    cached_hf_configs = sorted(
        glob(str(Path.home() / ".cache" / "huggingface" / "hub" / "models--*gemma*" / "**" / "config.json"), recursive=True)
    )
    if cached_hf_configs:
        return str(Path(cached_hf_configs[0]).parent)

    return "google/gemma-3-4b-it"


def transformers_health() -> dict[str, object]:
    try:
        import huggingface_hub
        import torch
        import transformers

        return {
            "available": True,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "torch": torch.__version__,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


@lru_cache(maxsize=1)
def detect_runtime() -> dict[str, object]:
    in_kaggle = os.path.isdir("/kaggle")
    is_rerun = _bool_env("KAGGLE_IS_COMPETITION_RERUN")
    smoke_mode = _bool_env("CAREBRIDGE_SMOKE_MODE", default=not is_rerun)
    backend = os.getenv("CAREBRIDGE_BACKEND", "smoke" if smoke_mode else "gemma")

    return {
        "in_kaggle": in_kaggle,
        "is_competition_rerun": is_rerun,
        "smoke_mode": smoke_mode,
        "backend": backend,
        "force_cpu": _force_cpu(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": _try_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "gemma_model_ref": find_gemma_model_ref(),
        "gemma_model_dirs": sorted(glob("/kaggle/input/models/**/config.json", recursive=True))[:10],
        "transformers_health": transformers_health(),
    }


def runtime_report_json() -> str:
    return json.dumps(detect_runtime(), indent=2)
