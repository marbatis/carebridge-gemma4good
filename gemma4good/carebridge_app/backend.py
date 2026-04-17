from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from .gemma_adapter import TextGenerator, TransformersGemmaGenerator, generate_gemma_assessment
from .knowledge_base import KnowledgeBase, RetrievedDocument
from .models import AssessmentRequest, AssessmentResponse, HandoffCard, SourceExcerpt
from .runtime import detect_runtime


PROJECT_NAME = "CareBridge"
SCENARIO_FOCUS = {
    "general-family": "Family health navigation",
    "maternal-postpartum": "Maternal and postpartum safety",
    "pediatric-fever": "Child fever and dehydration safety",
    "climate-respiratory": "Heat, smoke, and respiratory resilience",
}
SCENARIO_WHY_NOW = {
    "general-family": (
        "CareBridge is built for a United States where caregivers often have to make first-response "
        "decisions before they can reach a clinic, especially in rural, multilingual, and low-trust settings."
    ),
    "maternal-postpartum": (
        "This matters right now in the U.S. because many communities are losing local obstetric and postpartum coverage, "
        "while preventable pregnancy-related emergencies still happen when warning signs are missed or dismissed."
    ),
    "pediatric-fever": (
        "This matters right now in the U.S. because families are juggling pediatric access gaps, urgent-care costs, "
        "and the need to tell the difference between safe home monitoring and a child who is starting to decompensate."
    ),
    "climate-respiratory": (
        "This matters right now in the U.S. because heat waves, wildfire smoke, and respiratory triggers are hitting "
        "families more often, especially where transport, air quality, and clinic access are fragile."
    ),
}
SCENARIO_LONG_TERM = {
    "general-family": (
        "A tool like this can become a first-line caregiving habit for future generations: clear escalation, plain-language "
        "guidance, and grounded evidence instead of panic or guesswork."
    ),
    "maternal-postpartum": (
        "For future generations, better postpartum decision support protects mothers, infants, and the wider family system "
        "at one of the most vulnerable moments in life."
    ),
    "pediatric-fever": (
        "For future generations, giving caregivers better child-triage support means fewer dangerous delays, clearer home "
        "care routines, and stronger health confidence in the household."
    ),
    "climate-respiratory": (
        "For future generations, climate-linked health navigation will matter even more as families need resilient, local-first "
        "support during smoke events, heat emergencies, and disrupted services."
    ),
}
DISCLAIMER = (
    "CareBridge is a prototype educational tool, not a diagnosis service. "
    "If symptoms are severe, worsening, or you are unsure, contact a licensed clinician "
    "or emergency services right away."
)

EMERGENCY_TERMS = {
    "trouble breathing",
    "shortness of breath",
    "chest pain",
    "blue lips",
    "unconscious",
    "seizure",
    "fainting",
    "confusion",
    "severe allergic reaction",
}
URGENT_TERMS = {
    "high fever",
    "persistent vomiting",
    "dehydration",
    "blood in cough",
    "pregnant",
    "infant",
    "worsening",
}
MATERNAL_EMERGENCY_TERMS = {
    "heavy bleeding",
    "soaking pads",
    "postpartum bleeding",
    "blurred vision",
    "severe headache",
    "swelling",
}
PEDIATRIC_EMERGENCY_TERMS = {
    "hard to wake",
    "very sleepy",
    "stiff neck",
    "no wet diapers",
    "dry mouth",
}
CLIMATE_URGENT_TERMS = {
    "wildfire smoke",
    "smoke",
    "heat",
    "heat wave",
    "asthma",
    "inhaler",
}


class Backend(Protocol):
    def assess(self, request: AssessmentRequest, retrieved: list[RetrievedDocument]) -> AssessmentResponse:
        ...


def _contains_phrase(text: str, phrases: set[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _matched_phrases(text: str, phrases: set[str]) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in sorted(phrases) if phrase in lowered]


def _detect_scenario(request: AssessmentRequest, text: str) -> str:
    if request.care_scenario != "general-family":
        return request.care_scenario
    if request.pregnancy_postpartum_status != "not-applicable" or _contains_phrase(text, MATERNAL_EMERGENCY_TERMS):
        return "maternal-postpartum"
    if request.patient_age_band in {"infant", "child"} or _contains_phrase(text, PEDIATRIC_EMERGENCY_TERMS):
        return "pediatric-fever"
    if _contains_phrase(text, CLIMATE_URGENT_TERMS):
        return "climate-respiratory"
    return "general-family"


def _care_barriers(request: AssessmentRequest, scenario: str) -> list[str]:
    barriers: list[str] = []
    location = request.location_context.lower()

    if request.transport_access in {"limited", "none"} or any(term in location for term in {"remote", "rural", "far", "limited clinic"}):
        barriers.append("distance or transport limits could delay in-person care")
    if request.connectivity in {"spotty", "offline-first"}:
        barriers.append("the household may need guidance that still works with weak connectivity")
    if request.language.lower() != "english":
        barriers.append(f"the guidance should be ready for {request.language}-first communication")
    if request.patient_age_band in {"infant", "child", "older-adult"}:
        barriers.append(f"the patient is in a higher-attention age group: {request.patient_age_band.replace('-', ' ')}")
    if request.pregnancy_postpartum_status != "not-applicable":
        barriers.append(
            f"pregnancy/postpartum context raises the cost of missing escalation cues: "
            f"{request.pregnancy_postpartum_status.replace('-', ' ')}"
        )
    if scenario == "climate-respiratory":
        barriers.append("climate-driven smoke or heat can worsen symptoms faster than families expect")

    return barriers


def _grounded_sources(retrieved: list[RetrievedDocument]) -> list[SourceExcerpt]:
    return [
        SourceExcerpt(
            source_id=item.document.source_id,
            title=item.document.title,
            summary=item.document.summary,
            score=item.score,
        )
        for item in retrieved
    ]


def _build_handoff_card(
    request: AssessmentRequest,
    urgency: str,
    scenario: str,
    matched_alerts: list[str],
    next_steps: list[str],
) -> HandoffCard:
    context = [
        f"Caregiver role: {request.caregiver_role.replace('-', ' ')}",
        f"Patient age band: {request.patient_age_band.replace('-', ' ')}",
    ]
    if request.pregnancy_postpartum_status != "not-applicable":
        context.append(
            f"Pregnancy/postpartum: {request.pregnancy_postpartum_status.replace('-', ' ')}"
        )
    if request.location_context:
        context.append(f"Location context: {request.location_context}")
    if request.zip_code:
        context.append(f"ZIP code: {request.zip_code}")
    context.append(f"Transport access: {request.transport_access}")
    context.append(f"Connectivity: {request.connectivity}")

    return HandoffCard(
        headline=f"{SCENARIO_FOCUS[scenario]} handoff",
        risk_level=urgency,  # type: ignore[arg-type]
        top_concerns=matched_alerts[:3] or [SCENARIO_FOCUS[scenario]],
        immediate_actions=next_steps[:3],
        handoff_context=context,
    )


class SmokeBackend:
    def assess(self, request: AssessmentRequest, retrieved: list[RetrievedDocument]) -> AssessmentResponse:
        text = " ".join(
            [
                request.question,
                request.user_profile,
                request.location_context,
                request.patient_age_band,
                request.pregnancy_postpartum_status,
            ]
        ).lower()
        scenario = _detect_scenario(request, text)
        barriers = _care_barriers(request, scenario)
        matched_alerts = _matched_phrases(
            text,
            EMERGENCY_TERMS | URGENT_TERMS | MATERNAL_EMERGENCY_TERMS | PEDIATRIC_EMERGENCY_TERMS | CLIMATE_URGENT_TERMS,
        )

        if (
            _contains_phrase(text, EMERGENCY_TERMS)
            or (scenario == "maternal-postpartum" and _contains_phrase(text, MATERNAL_EMERGENCY_TERMS))
            or (scenario == "pediatric-fever" and _contains_phrase(text, PEDIATRIC_EMERGENCY_TERMS))
        ):
            urgency = "emergency"
            if scenario == "maternal-postpartum":
                summary = (
                    "This looks like a maternal or postpartum emergency pattern, where waiting for a routine callback "
                    "would be unsafe."
                )
                next_steps = [
                    "Escalate to emergency care now or call emergency services immediately.",
                    "Name the pregnancy or postpartum context clearly when asking for help.",
                    "Do not stay in home-monitoring mode if bleeding, severe headache, chest symptoms, or confusion are involved.",
                ]
                warning_signs = [
                    "heavy bleeding or rapidly increasing bleeding",
                    "severe headache, chest pain, or shortness of breath",
                    "confusion, fainting, or visual changes after pregnancy or birth",
                ]
            elif scenario == "pediatric-fever":
                summary = "This looks like a child-safety emergency pattern rather than a routine fever question."
                next_steps = [
                    "Seek emergency care now, especially if the child is hard to wake, struggling to breathe, or not staying hydrated.",
                    "Keep the child with an adult and limit extra movement while arranging help.",
                    "Bring a clear list of symptom timing, temperature, and fluid intake if you leave home.",
                ]
                warning_signs = [
                    "trouble breathing or blue lips",
                    "very sleepy, confused, or hard to wake",
                    "seizure activity or dehydration signs such as no wet diapers",
                ]
            else:
                summary = (
                    "The symptoms described include red-flag warning signs that need immediate "
                    "in-person evaluation."
                )
                next_steps = [
                    "Call local emergency services or go to the nearest emergency department now.",
                    "Do not rely on a text-only tool if breathing, consciousness, or severe chest symptoms are involved.",
                    "If someone is with the person, ask them to stay nearby and monitor changes while help is on the way.",
                ]
                warning_signs = [
                    "breathing difficulty",
                    "severe chest symptoms",
                    "confusion, fainting, or seizure activity",
                ]
        elif (
            _contains_phrase(text, URGENT_TERMS)
            or (scenario == "maternal-postpartum" and request.pregnancy_postpartum_status != "not-applicable")
            or (scenario == "pediatric-fever" and request.patient_age_band in {"infant", "child"})
            or (scenario == "climate-respiratory" and _contains_phrase(text, CLIMATE_URGENT_TERMS))
        ):
            urgency = "urgent"
            if scenario == "maternal-postpartum":
                summary = (
                    "Maternal and postpartum issues can worsen quickly, so this should move to same-day clinical review."
                )
                next_steps = [
                    "Contact a clinician, nurse line, urgent care, or OB team today rather than waiting.",
                    "Track bleeding, fever, pain, swelling, and blood-pressure-type symptoms while arranging care.",
                    "Escalate to emergency care immediately if chest symptoms, heavy bleeding, or confusion appear.",
                ]
                warning_signs = [
                    "fever after delivery or worsening pain",
                    "headache with swelling, vision changes, or chest symptoms",
                    "bleeding that is increasing instead of tapering",
                ]
            elif scenario == "pediatric-fever":
                summary = (
                    "The child may still be stable, but the combination of age and symptoms needs prompt review rather than wait-and-see alone."
                )
                next_steps = [
                    "Seek same-day pediatric advice through a clinic, urgent care, or telehealth line.",
                    "Track temperature, fluids, urination, and alertness while arranging care.",
                    "Escalate faster if the child becomes harder to wake, cannot drink, or breathing worsens.",
                ]
                warning_signs = [
                    "fewer wet diapers or signs of dehydration",
                    "persistent high fever or worsening lethargy",
                    "new breathing difficulty",
                ]
            elif scenario == "climate-respiratory":
                summary = (
                    "Smoke, asthma, and heat exposure can push families from ordinary symptoms into urgent care needs faster than expected."
                )
                next_steps = [
                    "Move to cleaner, cooler air if possible and seek same-day clinical advice.",
                    "Check whether rescue inhalers, fluids, cooling, or transport plans are available now.",
                    "Escalate immediately if breathing or confusion gets worse.",
                ]
                warning_signs = [
                    "worsening breathing after smoke or heat exposure",
                    "weakness, dizziness, or dehydration in the heat",
                    "limited transport making delayed care more likely",
                ]
            else:
                summary = (
                    "The situation may not require emergency transport, but it does warrant prompt "
                    "clinical review rather than home monitoring alone."
                )
                next_steps = [
                    "Seek urgent clinical advice today through a clinic, urgent care, or telehealth service.",
                    "Keep tracking fluids, temperature, and symptom changes while arranging care.",
                    "Escalate immediately if breathing, alertness, or pain worsens.",
                ]
                warning_signs = [
                    "unable to keep fluids down",
                    "symptoms lasting longer than expected",
                    "higher-risk patient context such as pregnancy or infancy",
                ]
        else:
            urgency = "routine"
            if scenario == "maternal-postpartum":
                summary = (
                    "The current description sounds appropriate for close monitoring, but postpartum and pregnancy concerns still need a lower threshold for escalation."
                )
                next_steps = [
                    "Monitor symptoms closely and keep a written log of pain, fever, bleeding, or swelling changes.",
                    "Use trusted postpartum guidance and make sure the household knows the red flags to watch for.",
                    "Escalate if symptoms worsen instead of improving over the next several hours.",
                ]
                warning_signs = [
                    "heavy bleeding, fever, or worsening pain",
                    "headache with swelling or vision changes",
                    "new chest symptoms, shortness of breath, or fainting",
                ]
            elif scenario == "pediatric-fever":
                summary = (
                    "This sounds appropriate for cautious home monitoring, but pediatric symptoms should be rechecked frequently for hydration and alertness."
                )
                next_steps = [
                    "Monitor temperature, fluids, urination, and energy level.",
                    "Use child-focused supportive care and make sure another caregiver knows the warning signs.",
                    "Escalate if the child is drinking less, urinating less, or becoming harder to wake.",
                ]
                warning_signs = [
                    "dehydration, fewer wet diapers, or dry mouth",
                    "new trouble breathing",
                    "worsening lethargy or persistent fever",
                ]
            elif scenario == "climate-respiratory":
                summary = (
                    "This sounds manageable at home for now, but smoke and heat exposure should be watched closely because they can accelerate respiratory strain."
                )
                next_steps = [
                    "Reduce smoke or heat exposure, increase fluids, and keep monitoring symptoms.",
                    "Check whether cleaner indoor air, shade, cooling, or inhaler access is available.",
                    "Escalate if breathing, dizziness, or exhaustion worsens.",
                ]
                warning_signs = [
                    "worsening shortness of breath",
                    "increasing weakness, confusion, or poor fluid intake",
                    "limited access to transport, cooling, or medications",
                ]
            else:
                summary = (
                    "The symptoms described sound appropriate for cautious home monitoring with a clear "
                    "plan for when to seek care if things worsen."
                )
                next_steps = [
                    "Monitor symptoms, rest, and hydration.",
                    "Use trusted local care guidance for supportive steps.",
                    "Escalate if red-flag symptoms appear or the condition gets worse instead of better.",
                ]
                warning_signs = [
                    "new breathing difficulty",
                    "new confusion or fainting",
                    "rapid worsening or inability to drink fluids",
                ]

        follow_up_questions = {
            "maternal-postpartum": [
                "How many days or weeks pregnant/postpartum is the patient?",
                "Has there been bleeding, swelling, severe headache, or fever?",
                "Is there a clinician, doula, nurse line, or hospital that the family can contact today?",
            ],
            "pediatric-fever": [
                "What is the child’s age and how high is the fever?",
                "Has the child been drinking fluids and urinating normally?",
                "Is the child alert, interactive, and breathing comfortably between coughs or crying?",
            ],
            "climate-respiratory": [
                "Was there recent smoke, wildfire, asthma, or heat exposure?",
                "Does the household have access to cooling, clean indoor air, or inhalers?",
                "Would transport or distance make same-day care hard if symptoms get worse?",
            ],
            "general-family": [
                "How long have these symptoms been present?",
                "Has the person been able to drink fluids and urinate normally?",
                "Are there risk factors such as pregnancy, infancy, older age, or chronic illness?",
            ],
        }[scenario]

        if request.language.lower() != "english":
            follow_up_questions.append(
                f"Should the final guidance be rewritten for a {request.language}-speaking caregiver?"
            )

        grounded_sources = _grounded_sources(retrieved)

        return AssessmentResponse(
            mode="smoke",
            project=PROJECT_NAME,
            urgency=urgency,
            care_focus=SCENARIO_FOCUS[scenario],
            summary=summary,
            why_now_in_america=SCENARIO_WHY_NOW[scenario],
            why_this_matters_long_term=SCENARIO_LONG_TERM[scenario],
            rationale=[
                f"Care focus selected: {SCENARIO_FOCUS[scenario]}.",
                f"Matched risk cues: {', '.join(matched_alerts[:4])}." if matched_alerts else "No explicit red-flag phrases were matched beyond the broader context.",
                f"Barriers considered: {'; '.join(barriers)}." if barriers else "No major access barriers were declared in the intake.",
                (
                    "Grounded by: " + ", ".join(source.title for source in grounded_sources[:3]) + "."
                )
                if grounded_sources
                else "No supporting note was retrieved from the local knowledge base.",
            ],
            care_barriers=barriers,
            next_steps=next_steps,
            warning_signs=warning_signs,
            follow_up_questions=follow_up_questions,
            handoff_card=_build_handoff_card(request, urgency, scenario, matched_alerts, next_steps),
            grounded_sources=grounded_sources,
            disclaimer=DISCLAIMER,
        )


class GemmaBackend:
    def __init__(self, generator: TextGenerator | None = None):
        self.generator = generator or TransformersGemmaGenerator()
        self.max_new_tokens = int(os.getenv("CAREBRIDGE_GEMMA_MAX_NEW_TOKENS", "320"))
        self.temperature = float(os.getenv("CAREBRIDGE_GEMMA_TEMPERATURE", "0.2"))

    def assess(self, request: AssessmentRequest, retrieved: list[RetrievedDocument]) -> AssessmentResponse:
        baseline = SmokeBackend().assess(request, retrieved)
        try:
            return generate_gemma_assessment(
                request=request,
                baseline=baseline,
                grounded_sources=baseline.grounded_sources,
                generator=self.generator,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            print(f"[CareBridge] Gemma backend fallback: {exc}")
            return baseline.model_copy(update={"mode": "gemma-fallback"})


class CareBridgeService:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        runtime = detect_runtime()
        self.backend: Backend = SmokeBackend() if runtime["backend"] == "smoke" else GemmaBackend()

    def assess(self, request: AssessmentRequest) -> AssessmentResponse:
        retrieval_query = " ".join(
            filter(
                None,
                [
                    request.question,
                    request.user_profile,
                    request.location_context,
                    request.care_scenario,
                    request.patient_age_band,
                    request.pregnancy_postpartum_status,
                ],
            )
        )
        retrieved = self.knowledge_base.retrieve(retrieval_query, top_k=4)
        return self.backend.assess(request, retrieved)


def build_service(base_dir: Path) -> CareBridgeService:
    docs_dir = base_dir / "knowledge"
    return CareBridgeService(KnowledgeBase(docs_dir))
