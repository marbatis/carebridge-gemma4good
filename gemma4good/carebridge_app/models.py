from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Urgency = Literal["emergency", "urgent", "routine"]
CareScenario = Literal["general-family", "maternal-postpartum", "pediatric-fever", "climate-respiratory"]
PatientAgeBand = Literal["adult", "older-adult", "infant", "child", "teen"]
CaregiverRole = Literal["self", "parent", "adult-child", "partner", "community-health-worker", "teacher", "other"]
PregnancyStatus = Literal["not-applicable", "pregnant", "postpartum-6-weeks", "postpartum-1-year"]
TransportAccess = Literal["reliable", "limited", "none"]
Connectivity = Literal["stable", "spotty", "offline-first"]


class AssessmentRequest(BaseModel):
    question: str = Field(..., min_length=8, max_length=3000)
    language: str = Field(default="English", max_length=100)
    caregiver_role: CaregiverRole = "self"
    care_scenario: CareScenario = "general-family"
    patient_age_band: PatientAgeBand = "adult"
    pregnancy_postpartum_status: PregnancyStatus = "not-applicable"
    transport_access: TransportAccess = "reliable"
    connectivity: Connectivity = "stable"
    zip_code: str = Field(default="", max_length=20)
    location_context: str = Field(default="", max_length=300)
    user_profile: str = Field(default="", max_length=300)


class SourceExcerpt(BaseModel):
    source_id: str
    title: str
    summary: str
    score: float


class HandoffCard(BaseModel):
    headline: str
    risk_level: Urgency
    top_concerns: list[str]
    immediate_actions: list[str]
    handoff_context: list[str]


class AssessmentResponse(BaseModel):
    mode: str
    project: str
    urgency: Urgency
    care_focus: str
    summary: str
    why_now_in_america: str
    why_this_matters_long_term: str
    rationale: list[str]
    care_barriers: list[str]
    next_steps: list[str]
    warning_signs: list[str]
    follow_up_questions: list[str]
    handoff_card: HandoffCard
    grounded_sources: list[SourceExcerpt]
    disclaimer: str
