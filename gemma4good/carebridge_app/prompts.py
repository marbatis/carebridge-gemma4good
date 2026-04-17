from __future__ import annotations

import json

from .models import AssessmentRequest, AssessmentResponse, SourceExcerpt


SYSTEM_PROMPT = """You are CareBridge, a grounded health-navigation assistant.

Your job is to:
- summarize the user concern in plain language
- surface red flags clearly
- avoid overclaiming or pretending to diagnose
- suggest next-step actions based on grounded sources
- keep the response useful for low-connectivity and privacy-sensitive settings

You must not fabricate medical facts or imply certainty when you only have partial information.
You must preserve the risk level and care focus supplied by the application.
Return only the requested structured content with no markdown fences and no extra commentary.
"""


def build_gemma_messages(
    *,
    system_prompt: str,
    request: AssessmentRequest,
    baseline: AssessmentResponse,
    grounded_sources: list[SourceExcerpt],
) -> list[dict[str, object]]:
    case_payload = {
        "language": request.language,
        "caregiver_role": request.caregiver_role,
        "care_scenario": request.care_scenario,
        "patient_age_band": request.patient_age_band,
        "pregnancy_postpartum_status": request.pregnancy_postpartum_status,
        "transport_access": request.transport_access,
        "connectivity": request.connectivity,
        "zip_code": request.zip_code,
        "location_context": request.location_context,
        "user_profile": request.user_profile,
        "question": request.question,
    }
    baseline_payload = {
        "urgency": baseline.urgency,
        "care_focus": baseline.care_focus,
        "summary": baseline.summary,
        "care_barriers": baseline.care_barriers,
        "warning_signs": baseline.warning_signs,
        "next_steps": baseline.next_steps,
        "follow_up_questions": baseline.follow_up_questions,
    }
    source_payload = [source.model_dump() for source in grounded_sources]

    user_prompt = (
        "Generate the narrative layer for this CareBridge assessment.\n\n"
        "Rules:\n"
        "- Keep the supplied urgency and care focus exactly as-is.\n"
        "- Stay grounded in the case details and source notes only.\n"
        "- Do not diagnose.\n"
        "- Write for a caregiver in plain language.\n"
        "- Acknowledge U.S. care access realities in `why_now_in_america`.\n"
        "- Explain why this kind of tool matters for future generations in `why_this_matters_long_term`.\n"
        "- If the requested language is not English, translate the JSON values into that language while keeping the JSON keys in English.\n"
        "- Prefer this exact labeled format instead of prose:\n"
        "SUMMARY: one paragraph\n"
        "WHY_NOW_IN_AMERICA: one paragraph\n"
        "WHY_THIS_MATTERS_LONG_TERM: one paragraph\n"
        "RATIONALE:\n- bullet\n- bullet\n- bullet\n"
        "NEXT_STEPS:\n- bullet\n- bullet\n- bullet\n"
        "FOLLOW_UP_QUESTIONS:\n- bullet\n- bullet\n- bullet\n"
        "- If space is tight, complete SUMMARY and the two WHY sections first, then add as many list bullets as you can.\n"
        "- Keep every section concise.\n\n"
        f"Case:\n{json.dumps(case_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Deterministic baseline:\n{json.dumps(baseline_payload, indent=2, ensure_ascii=False)}\n\n"
        f"Grounded sources:\n{json.dumps(source_payload, indent=2, ensure_ascii=False)}"
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                }
            ],
        },
    ]
