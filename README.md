# CareBridge

CareBridge is a local-first maternal-and-family resilience assistant built for the `Gemma 4 Good Hackathon`.

It is designed for high-stress caregiver moments where access to care is delayed, internet is weak, or the next step is unclear. The current prototype focuses on:

- postpartum warning signs and emergency escalation
- pediatric fever and dehydration support
- heat, smoke, and respiratory risk

The architecture intentionally separates deterministic safety structure from model-generated narration:

- deterministic triage and escalation backbone
- grounded retrieval from local health notes
- optional `Gemma` narrative layer for clearer caregiver-facing communication

## Repository Layout

- `gemma4good/carebridge_app/`
  Main FastAPI application.
- `gemma4good/carebridge_gemma_kaggle_runtime.py`
  Kaggle-oriented runtime entrypoint.
- `tests/`
  Local tests for the app and Gemma adapter behavior.
- `requirements-gemma.txt`
  Local dependency set for the Gemma-enabled path.

## Run Locally

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn gemma4good.carebridge_app.app:app --reload
```

Open:

- `http://127.0.0.1:8000/`

## Enable Gemma

Example environment:

```bash
export CAREBRIDGE_BACKEND=gemma
export CAREBRIDGE_GEMMA_MODEL=google/gemma-3-4b-it
export CAREBRIDGE_GEMMA_MAX_NEW_TOKENS=320
export CAREBRIDGE_GEMMA_TEMPERATURE=0.2
```

If those variables are not set, the app falls back to a deterministic smoke backend so the workflow can still be demonstrated safely.

## Test

```bash
pytest -q tests/test_carebridge_app.py tests/test_gemma_backend.py
```

## Why This Project Exists

CareBridge is not positioned as a diagnosis engine. It is a family decision-support and handoff tool for moments when systems are overloaded, far away, or hard to reach.

The prototype shows one narrow but credible workflow end to end: helping a caregiver recognize postpartum danger signs earlier, understand why escalation is urgent, and hand off the right information cleanly.
