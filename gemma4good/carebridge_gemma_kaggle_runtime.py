"""
CareBridge Gemma Kaggle runtime notebook source.

Goal:
- run the existing CareBridge package inside Kaggle
- attach a real Gemma 4 model
- keep the same decision structure and JSON output shape
- fail loudly if the notebook only hits fallback mode
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import zipfile
from glob import glob
from pathlib import Path


MODEL_INSTANCE_REF = "google/gemma-4/transformers/gemma-4-e2b-it"
HF_TRANSFORMERS_GIT = "git+https://github.com/huggingface/transformers.git"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def supports_gemma4() -> bool:
    probe = [
        sys.executable,
        "-c",
        (
            "from transformers import Gemma4ForConditionalGeneration; "
            "print(1)"
        ),
    ]
    try:
        output = subprocess.check_output(probe, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        print("Gemma4 support probe failed:", exc.output)
        return False
    return output.endswith("1")


def ensure_runtime_packages() -> None:
    transformers_install = [HF_TRANSFORMERS_GIT]
    optional_installs: list[str] = ["huggingface-hub>=1.5,<2"]
    if package_version("accelerate") is None:
        optional_installs.append("accelerate>=1,<2")
    if package_version("sentencepiece") is None:
        optional_installs.append("sentencepiece>=0.2,<1")
    if package_version("kagglehub") is None:
        optional_installs.append("kagglehub>=0.3,<1")

    print("Installing runtime packages:", transformers_install + optional_installs)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            *transformers_install,
        ]
    )
    if optional_installs:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--upgrade",
                *optional_installs,
            ]
        )
    importlib.invalidate_caches()
    if not supports_gemma4():
        raise RuntimeError("Transformers upgrade completed, but Gemma 4 support is still unavailable.")


def detect_mounted_model() -> str | None:
    for config_path in sorted(glob("/kaggle/input/models/**/config.json", recursive=True)):
        if "gemma" in config_path.lower():
            return str(Path(config_path).parent)
    for config_path in sorted(glob("/kaggle/input/**/config.json", recursive=True)):
        if "gemma" in config_path.lower():
            return str(Path(config_path).parent)
    return None


def resolve_model_path() -> str:
    mounted = detect_mounted_model()
    if mounted:
        print(f"Using mounted Kaggle model: {mounted}")
        return mounted

    import kagglehub

    downloaded = kagglehub.model_download(MODEL_INSTANCE_REF)
    print(f"Downloaded model via kagglehub: {downloaded}")
    return downloaded


def ensure_repo_on_path() -> Path:
    candidates = [
        Path.cwd().resolve(),
        Path("/kaggle/input/carebridge-app-source"),
        Path("/kaggle/input"),
    ]

    for root in candidates:
        package_root = root / "gemma4good"
        if package_root.exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root

    for package_root in Path("/kaggle/input").glob("*/gemma4good"):
        root = package_root.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return root

    zip_candidates = list(Path("/kaggle/input").glob("*/gemma4good.zip"))
    if zip_candidates:
        extract_root = Path("/kaggle/working/carebridge_app_source")
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_candidates[0]) as zf:
            zf.extractall(extract_root)
        root = extract_root
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return root

    raise FileNotFoundError(
        "Expected the CareBridge source to be present either in the notebook bundle or in a Kaggle dataset input."
    )


def runtime_snapshot() -> dict[str, object]:
    return {
        "cwd": str(Path.cwd().resolve()),
        "python": sys.version,
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "sentencepiece": package_version("sentencepiece"),
        "kagglehub": package_version("kagglehub"),
        "mounted_model": detect_mounted_model(),
    }


def build_requests(AssessmentRequest: type) -> list[tuple[str, object]]:
    requests = [
        (
            "postpartum_emergency",
            AssessmentRequest(
                question="My wife gave birth 10 days ago and now has a severe headache, chest pain, and swelling.",
                language="English",
                caregiver_role="partner",
                care_scenario="maternal-postpartum",
                patient_age_band="adult",
                pregnancy_postpartum_status="postpartum-6-weeks",
                transport_access="limited",
                connectivity="spotty",
                location_context="Rural county with limited clinic access",
                user_profile="10 days postpartum",
            ),
        ),
        (
            "child_dehydration",
            AssessmentRequest(
                question="My 3 year old has a high fever, barely drank today, and has had no wet diapers since this morning.",
                language="English",
                caregiver_role="parent",
                care_scenario="pediatric-fever",
                patient_age_band="child",
                pregnancy_postpartum_status="not-applicable",
                transport_access="limited",
                connectivity="stable",
                location_context="Small town with urgent care 45 minutes away",
                user_profile="3 year old child",
            ),
        ),
    ]
    if os.getenv("CAREBRIDGE_RUN_ALL_DEMOS", "").lower() in {"1", "true", "yes", "on"}:
        return requests
    return requests[:1]


def main() -> None:
    ensure_runtime_packages()
    repo_root = ensure_repo_on_path()

    os.environ["CAREBRIDGE_BACKEND"] = "gemma"
    os.environ["CAREBRIDGE_SMOKE_MODE"] = "0"
    os.environ["CAREBRIDGE_FORCE_CPU"] = "1"
    os.environ["CAREBRIDGE_GEMMA_MAX_NEW_TOKENS"] = "128"
    os.environ["CAREBRIDGE_GEMMA_TEMPERATURE"] = "0"
    os.environ["CAREBRIDGE_GEMMA_MODEL"] = resolve_model_path()

    print("Runtime snapshot:")
    print(json.dumps(runtime_snapshot(), indent=2))

    from gemma4good.carebridge_app.backend import build_service
    from gemma4good.carebridge_app.models import AssessmentRequest
    from gemma4good.carebridge_app.runtime import detect_runtime

    print("CareBridge runtime:")
    print(json.dumps(detect_runtime(), indent=2))

    base_dir = repo_root / "gemma4good" / "carebridge_app"
    service = build_service(base_dir)

    outputs: dict[str, dict[str, object]] = {}
    started = time.time()
    for name, request in build_requests(AssessmentRequest):
        response = service.assess(request)
        payload = response.model_dump()
        outputs[name] = payload
        print(f"\n=== {name} ===")
        print(json.dumps(payload, indent=2))
        if payload["mode"] != "gemma":
            raise RuntimeError(
                f"Expected a real Gemma response for {name}, but got mode={payload['mode']}"
            )

    elapsed = time.time() - started
    outputs["_meta"] = {
        "model_ref": os.environ["CAREBRIDGE_GEMMA_MODEL"],
        "elapsed_seconds": round(elapsed, 2),
        "backend": "gemma",
    }

    output_path = Path("/kaggle/working/carebridge_demo_outputs.json")
    output_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"\nWrote outputs to {output_path}")


if __name__ == "__main__":
    main()
