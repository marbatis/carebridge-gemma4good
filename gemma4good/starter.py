"""
Gemma 4 Good Hackathon starter notebook source.

This is a development notebook scaffold, not a final submission notebook.
It carries over the reliable operational lessons from AIMO:

- discover Kaggle paths dynamically
- separate smoke runs from heavy runs
- print runtime facts early
- keep the project checklist close to the code
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from glob import glob
from pathlib import Path


PROJECT_NAME = "CareBridge"
PROJECT_SUBTITLE = "Offline multilingual health navigation with Gemma 4"
PRIMARY_TRACK = "Health & Sciences"
SECONDARY_TECH_TRACK = "Ollama or Unsloth"


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"[error] {exc}"


def bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def find_competition_inputs() -> list[str]:
    return sorted(glob("/kaggle/input/competitions/*"))


def find_model_dirs(name_hint: str = "gemma-4") -> list[str]:
    hits = []
    for config_path in sorted(glob("/kaggle/input/models/**/config.json", recursive=True)):
        lower = config_path.lower()
        if name_hint in lower:
            hits.append(str(Path(config_path).parent))
    return hits


def detect_gpu() -> str:
    return sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])


def detect_runtime() -> dict:
    in_kaggle = os.path.isdir("/kaggle")
    is_rerun = bool_env("KAGGLE_IS_COMPETITION_RERUN")
    force_smoke = bool_env("G4G_SMOKE_MODE", default=not is_rerun)

    return {
        "project_name": PROJECT_NAME,
        "project_subtitle": PROJECT_SUBTITLE,
        "primary_track": PRIMARY_TRACK,
        "secondary_tech_track": SECONDARY_TECH_TRACK,
        "in_kaggle": in_kaggle,
        "is_competition_rerun": is_rerun,
        "smoke_mode": force_smoke,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": detect_gpu(),
        "competition_inputs": find_competition_inputs(),
        "gemma_model_dirs": find_model_dirs(),
    }


def print_runtime_report() -> None:
    report = detect_runtime()
    print(json.dumps(report, indent=2))


def print_build_checklist() -> None:
    checklist = [
        "Choose one narrow health workflow to demo.",
        "Pick one Gemma 4 model path that actually resolves in this runtime.",
        "Decide whether this notebook is smoke mode or heavy mode.",
        "Create one retrieval source with trusted documents.",
        "Define one structured output format for guidance and escalation.",
        "Record screenshots and short clips as you go.",
        "Draft the writeup before polishing the UI.",
    ]
    print("\nBuild checklist:")
    for item in checklist:
        print(f"- {item}")


def smoke_demo(question: str) -> dict:
    return {
        "mode": "smoke",
        "question": question,
        "status": "placeholder",
        "next_step": "replace with Gemma 4 inference or app call",
    }


def main() -> None:
    print_runtime_report()
    print_build_checklist()

    if detect_runtime()["smoke_mode"]:
        sample = smoke_demo("I have a fever and sore throat. What should I do next?")
        print("\nSmoke demo:")
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
