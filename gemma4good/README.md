# Gemma 4 Good Starter

## Competition Facts

- Competition: `The Gemma 4 Good Hackathon`
- Host: Google DeepMind
- Start date: April 2, 2026
- Final submission deadline: May 18, 2026 at 11:59 PM UTC
- Submission type: judged hackathon, not a scored test-set competition
- Provided competition data: none

Official competition page:
- https://www.kaggle.com/competitions/gemma-4-good-hackathon

The notebook payload confirms there is no dataset in the competition package.

## What Matters Here

This competition is not about squeezing out leaderboard points.

The judges want:
- impact and vision
- a strong, clear video story
- real technical execution with Gemma 4

Required submission pieces:
- Kaggle Writeup
- public video
- public code repository
- live demo
- media gallery

Scoring:
- Impact & Vision: 40
- Video Pitch & Storytelling: 30
- Technical Depth & Execution: 30

## What To Carry Over From AIMO

The useful lessons from AIMO are operational, not mathematical:

- Start from a runtime that actually works before adding ambitious logic.
- Use dynamic path discovery instead of hardcoded Kaggle mount paths.
- Keep a smoke mode for ordinary notebook validation runs.
- Separate heavy prep work from the demo/runtime path.
- Build the artifact checklist early so the final submission is not a scramble.

## Recommended Direction

Recommended concept:
- `Offline Multilingual Health Navigator`

Why this is a strong first bet:
- clear impact story
- easy to demonstrate visually
- naturally fits Gemma 4 strengths: local inference, multilingual support, function calling, multimodal inputs, grounded retrieval
- can target both `Health & Sciences` and a technology prize depending on implementation

Core demo:
- user describes symptoms or uploads a photo/document
- the app asks follow-up questions
- it grounds answers in trusted local documents
- it outputs plain-language guidance, risk flags, and escalation suggestions
- it works in low-connectivity or privacy-sensitive settings

## Suggested Track Positioning

Primary track:
- `Health & Sciences`

Secondary technical angle, depending on implementation:
- `Ollama` if we ship a strong local desktop demo
- `llama.cpp` if we optimize for constrained hardware
- `Unsloth` if we fine-tune Gemma 4 for a specific health task

## Execution Plan

1. Build one narrow, real demo.
2. Use Gemma 4 in a way that is visible and defensible.
3. Record evidence while building: screenshots, short clips, benchmark notes.
4. Draft the writeup before the app feels "done".
5. Treat the 3-minute video as a product launch, not an afterthought.

## Immediate Build Plan

1. Use the starter notebook to validate Kaggle runtime paths and available Gemma assets.
2. Pick one narrow use case:
   - maternal health guidance
   - first-aid escalation assistant
   - medication information explainer
   - multilingual care navigation
3. Build a thin demo first:
   - one ingestion mode
   - one retrieval source
   - one clear output format
4. Add differentiation only after the thin demo works:
   - multimodal input
   - structured function calls
   - fine-tuning
   - edge/local packaging

## Files In This Starter

- `README.md`: competition strategy and recommended direction
- `writeup_outline.md`: writeup structure aligned with judging
- `starter.py`: Kaggle notebook source scaffold
- `carebridge_app/`: actual FastAPI prototype for the recommended concept

## Local Prototype

The first working demo app lives in:
- `gemma4good/carebridge_app`

Run it locally from the workspace root with:

```bash
uvicorn gemma4good.carebridge_app.app:app --reload
```

What it already does:
- grounded retrieval over local health guidance notes
- structured triage output with maternal, pediatric, and climate-respiratory paths
- explains why the case matters in the U.S. right now and why it matters long term
- produces a handoff card a caregiver can use with a clinician, nurse line, or emergency team
- smoke backend for safe iteration
- optional real Gemma backend that keeps the same UX and risk structure

This is the right shape for the hackathon:
- demoable quickly
- easy to explain in a writeup and video
- narrow enough to finish

## Enable Gemma

The app keeps the deterministic caregiver workflow and uses Gemma only for the narrative layer:
- summary
- U.S.-relevance framing
- long-term framing
- caregiver-friendly next steps
- follow-up questions

Recommended environment variables:

```bash
export CAREBRIDGE_BACKEND=gemma
export CAREBRIDGE_GEMMA_MODEL=google/gemma-3-4b-it
export CAREBRIDGE_GEMMA_MAX_NEW_TOKENS=320
export CAREBRIDGE_GEMMA_TEMPERATURE=0.2
```

If you are running locally instead of Kaggle, install a compatible stack first:

```bash
pip install -r requirements-gemma.txt
```

On Kaggle, the runtime should already provide `torch`, and you mainly need the correct Gemma model attachment plus a compatible `transformers` stack.
