import json
from pathlib import Path

from gemma4good.carebridge_app.backend import GemmaBackend
from gemma4good.carebridge_app.gemma_adapter import (
    collapse_system_into_user,
    normalize_gemma4_response,
    parse_gemma_draft,
)
from gemma4good.carebridge_app.knowledge_base import KnowledgeBase
from gemma4good.carebridge_app.models import AssessmentRequest


BASE_DIR = Path(__file__).resolve().parents[1] / "gemma4good" / "carebridge_app"


class FakeGenerator:
    def __init__(self, output: str):
        self.output = output
        self.calls: list[list[dict[str, object]]] = []

    def generate(
        self,
        messages: list[dict[str, object]],
        *,
        max_new_tokens: int = 700,
        temperature: float = 0.2,
    ) -> str:
        self.calls.append(messages)
        return self.output


class SequenceGenerator:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls: list[list[dict[str, object]]] = []

    def generate(
        self,
        messages: list[dict[str, object]],
        *,
        max_new_tokens: int = 700,
        temperature: float = 0.2,
    ) -> str:
        self.calls.append(messages)
        return self.outputs.pop(0)


def _postpartum_request() -> AssessmentRequest:
    return AssessmentRequest(
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
    )


def test_gemma_backend_merges_model_narrative_and_keeps_risk_structure() -> None:
    knowledge_base = KnowledgeBase(BASE_DIR / "knowledge")
    request = _postpartum_request()
    retrieved = knowledge_base.retrieve(
        " ".join([request.question, request.location_context, request.user_profile]),
        top_k=4,
    )
    generator = FakeGenerator(
        json.dumps(
            {
                "summary": "This is a postpartum emergency pattern and the family should not wait for a routine callback.",
                "why_now_in_america": "Postpartum care gaps and rural access delays make missed escalation especially dangerous in the U.S. right now.",
                "why_this_matters_long_term": "Helping families recognize postpartum red flags earlier protects mothers, babies, and caregivers across generations.",
                "rationale": [
                    "The case includes chest pain, severe headache, and swelling shortly after birth.",
                    "Transport and connectivity barriers could slow escalation.",
                    "Grounded notes reinforce that postpartum context changes the threshold for urgent action.",
                ],
                "next_steps": [
                    "Call emergency services or go to emergency care now.",
                    "Say clearly that the patient is 10 days postpartum.",
                    "Keep another adult nearby and avoid home monitoring only.",
                ],
                "follow_up_questions": [
                    "How much bleeding is happening right now?",
                    "Has there been any fever, blurred vision, or shortness of breath?",
                    "Who can help with transport or emergency childcare right now?",
                ],
            }
        )
    )

    response = GemmaBackend(generator=generator).assess(request, retrieved)

    assert response.mode == "gemma"
    assert response.urgency == "emergency"
    assert response.care_focus == "Maternal and postpartum safety"
    assert response.summary.startswith("This is a postpartum emergency pattern")
    assert response.handoff_card.risk_level == "emergency"
    assert response.handoff_card.immediate_actions[0] == "Call emergency services or go to emergency care now."
    assert response.grounded_sources
    assert generator.calls


def test_gemma_backend_falls_back_to_deterministic_output_on_bad_json() -> None:
    knowledge_base = KnowledgeBase(BASE_DIR / "knowledge")
    request = _postpartum_request()
    retrieved = knowledge_base.retrieve(request.question, top_k=4)

    response = GemmaBackend(generator=FakeGenerator("not valid json")).assess(request, retrieved)

    assert response.mode == "gemma-fallback"
    assert response.urgency == "emergency"
    assert response.summary
    assert response.grounded_sources


def test_gemma_backend_repairs_non_json_output_before_falling_back() -> None:
    knowledge_base = KnowledgeBase(BASE_DIR / "knowledge")
    request = _postpartum_request()
    retrieved = knowledge_base.retrieve(request.question, top_k=4)
    generator = SequenceGenerator(
        [
            "Here is the answer in prose instead of JSON.",
            json.dumps(
                {
                    "summary": "This is still a postpartum emergency pattern and the household should escalate now.",
                    "why_now_in_america": "Postpartum care gaps in the U.S. make early escalation and clear caregiver guidance especially important.",
                    "why_this_matters_long_term": "Better family triage habits protect mothers, infants, and caregivers across generations.",
                    "rationale": [
                        "Chest pain, severe headache, and swelling are red flags after birth.",
                        "Transport and connectivity barriers make delays more dangerous.",
                        "The grounded source notes reinforce urgent escalation in postpartum cases.",
                    ],
                    "next_steps": [
                        "Seek emergency care now.",
                        "Tell the care team the patient is 10 days postpartum.",
                        "Do not rely on home monitoring alone.",
                    ],
                    "follow_up_questions": [
                        "Has there been heavy bleeding?",
                        "Is there any shortness of breath or blurred vision?",
                        "Who can help with transport right now?",
                    ],
                }
            ),
        ]
    )

    response = GemmaBackend(generator=generator).assess(request, retrieved)

    assert response.mode == "gemma"
    assert response.summary.startswith("This is still a postpartum emergency pattern")
    assert len(generator.calls) == 2


def test_parse_gemma_draft_accepts_labeled_sections() -> None:
    draft = parse_gemma_draft(
        """SUMMARY: This is a postpartum emergency pattern and the family should escalate now.
WHY_NOW_IN_AMERICA: Postpartum care gaps in the United States make clear escalation guidance more important.
WHY_THIS_MATTERS_LONG_TERM: Better family triage habits protect mothers and infants across generations.
RATIONALE:
- Chest pain, severe headache, and swelling are red flags after birth.
- Distance and weak connectivity raise the cost of delay.
- The grounded notes reinforce postpartum escalation.
NEXT_STEPS:
- Seek emergency care now.
- Tell the care team the patient is 10 days postpartum.
- Do not rely on home monitoring alone.
FOLLOW_UP_QUESTIONS:
- Has there been heavy bleeding?
- Is there any shortness of breath or blurred vision?
- Who can help with transport right now?
"""
    )

    assert draft.summary.startswith("This is a postpartum emergency pattern")
    assert len(draft.rationale) == 3


def test_parse_gemma_draft_allows_partial_lists_for_baseline_backfill() -> None:
    draft = parse_gemma_draft(
        """SUMMARY: This is a postpartum emergency pattern and the family should escalate now.
WHY_NOW_IN_AMERICA: Postpartum care gaps in the United States make clear escalation guidance more important.
WHY_THIS_MATTERS_LONG_TERM: Better family triage habits protect mothers and infants across generations.
RATIONALE:
- Chest pain, severe headache, and swelling are red flags after birth.
"""
    )

    assert draft.summary.startswith("This is a postpartum emergency pattern")
    assert draft.next_steps == []


def test_parse_gemma_draft_allows_missing_long_term_field_for_backfill() -> None:
    draft = parse_gemma_draft(
        """SUMMARY: This is a postpartum emergency pattern and the family should escalate now.
WHY_NOW_IN_AMERICA: Postpartum care gaps in the United States make clear escalation guidance more important.
"""
    )

    assert draft.summary.startswith("This is a postpartum emergency pattern")
    assert draft.why_this_matters_long_term == ""


def test_collapse_system_into_user_preserves_chat_payload() -> None:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Return JSON only."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Say hello."}],
        },
    ]

    collapsed = collapse_system_into_user(messages)

    assert len(collapsed) == 1
    assert collapsed[0]["role"] == "user"
    content = collapsed[0]["content"]
    assert isinstance(content, list)
    assert "Return JSON only." in content[0]["text"]
    assert "Say hello." in content[1]["text"]


def test_normalize_gemma4_response_prefers_parsed_content() -> None:
    parsed = {
        "role": "assistant",
        "thinking": "hidden reasoning",
        "content": "{\"summary\": \"Hello\"}",
    }

    assert normalize_gemma4_response(parsed, "raw text") == "{\"summary\": \"Hello\"}"
