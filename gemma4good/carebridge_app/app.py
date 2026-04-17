from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .backend import build_service
from .models import AssessmentRequest
from .runtime import detect_runtime, runtime_report_json


BASE_DIR = Path(__file__).resolve().parent
service = build_service(BASE_DIR)

app = FastAPI(title="CareBridge", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _empty_state() -> dict[str, object]:
    return {
        "request_payload": {
            "question": "",
            "language": "English",
            "caregiver_role": "self",
            "care_scenario": "general-family",
            "patient_age_band": "adult",
            "pregnancy_postpartum_status": "not-applicable",
            "transport_access": "reliable",
            "connectivity": "stable",
            "zip_code": "",
            "location_context": "",
            "user_profile": "",
        },
        "response_payload": None,
        "runtime_report": runtime_report_json(),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", _empty_state())


@app.post("/", response_class=HTMLResponse)
def assess_form(
    request: Request,
    question: str = Form(...),
    language: str = Form(default="English"),
    caregiver_role: str = Form(default="self"),
    care_scenario: str = Form(default="general-family"),
    patient_age_band: str = Form(default="adult"),
    pregnancy_postpartum_status: str = Form(default="not-applicable"),
    transport_access: str = Form(default="reliable"),
    connectivity: str = Form(default="stable"),
    zip_code: str = Form(default=""),
    location_context: str = Form(default=""),
    user_profile: str = Form(default=""),
) -> HTMLResponse:
    payload = AssessmentRequest(
        question=question,
        language=language,
        caregiver_role=caregiver_role,
        care_scenario=care_scenario,
        patient_age_band=patient_age_band,
        pregnancy_postpartum_status=pregnancy_postpartum_status,
        transport_access=transport_access,
        connectivity=connectivity,
        zip_code=zip_code,
        location_context=location_context,
        user_profile=user_profile,
    )
    response_payload = service.assess(payload)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request_payload": payload.model_dump(),
            "response_payload": response_payload.model_dump(),
            "runtime_report": runtime_report_json(),
        },
    )


@app.post("/api/assess")
def assess_api(payload: AssessmentRequest) -> dict[str, object]:
    return service.assess(payload).model_dump()


@app.get("/api/runtime")
def runtime_api() -> dict[str, object]:
    runtime = detect_runtime()
    return {"runtime": runtime, "backend": runtime["backend"]}


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}
