from fastapi.testclient import TestClient

from gemma4good.carebridge_app.app import app, service
from gemma4good.carebridge_app.models import AssessmentRequest


client = TestClient(app)


def test_home_page_loads() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "CareBridge" in response.text
    assert "maternal-and-family resilience assistant" in response.text


def test_retrieval_finds_urgent_doc() -> None:
    retrieved = service.knowledge_base.retrieve("The patient has chest pain and trouble breathing.")
    assert retrieved
    assert retrieved[0].document.source_id == "urgent-warning-signs"


def test_api_escalates_emergency_case() -> None:
    payload = {
        "question": "My father has chest pain, shortness of breath, and seems confused.",
        "language": "English",
        "location_context": "Remote village",
        "user_profile": "Older adult",
    }
    response = client.post("/api/assess", json=payload)
    body = response.json()
    assert response.status_code == 200
    assert body["urgency"] == "emergency"
    assert body["care_focus"] == "Family health navigation"
    assert body["handoff_card"]["risk_level"] == "emergency"
    assert body["grounded_sources"]


def test_service_handles_routine_case() -> None:
    payload = AssessmentRequest(
        question="I have a sore throat and mild fever since yesterday.",
        language="English",
    )
    response = service.assess(payload)
    assert response.urgency == "routine"
    assert response.mode == "smoke"


def test_postpartum_path_gets_maternal_focus() -> None:
    payload = {
        "question": "My wife gave birth 10 days ago and now has a severe headache, chest pain, and swelling.",
        "language": "English",
        "caregiver_role": "partner",
        "care_scenario": "maternal-postpartum",
        "patient_age_band": "adult",
        "pregnancy_postpartum_status": "postpartum-6-weeks",
        "transport_access": "limited",
        "connectivity": "spotty",
        "location_context": "Rural county with limited clinic access",
        "user_profile": "10 days postpartum",
    }
    response = client.post("/api/assess", json=payload)
    body = response.json()
    assert response.status_code == 200
    assert body["urgency"] == "emergency"
    assert body["care_focus"] == "Maternal and postpartum safety"
    assert "postpartum" in body["why_now_in_america"].lower()
    assert any(source["source_id"] == "postpartum-warning-signs" for source in body["grounded_sources"])


def test_runtime_api_returns_structured_runtime() -> None:
    response = client.get("/api/runtime")
    body = response.json()
    assert response.status_code == 200
    assert "runtime" in body
    assert "backend" in body
