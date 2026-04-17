from __future__ import annotations

import json
import os
from functools import cached_property
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from .models import AssessmentRequest, AssessmentResponse, SourceExcerpt
from .prompts import SYSTEM_PROMPT, build_gemma_messages
from .runtime import find_gemma_model_ref


class TextGenerator(Protocol):
    def generate(
        self,
        messages: list[dict[str, object]],
        *,
        max_new_tokens: int = 700,
        temperature: float = 0.2,
    ) -> str:
        ...


class GemmaNarrativeDraft(BaseModel):
    summary: str = Field(default="", max_length=700)
    why_now_in_america: str = Field(default="", max_length=700)
    why_this_matters_long_term: str = Field(default="", max_length=700)
    rationale: list[str] = Field(default_factory=list, max_length=5)
    next_steps: list[str] = Field(default_factory=list, max_length=5)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=5)


def _preferred_dtype(torch_module: Any) -> Any:
    if _cuda_is_usable(torch_module):
        if hasattr(torch_module.cuda, "is_bf16_supported") and torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        return torch_module.float16
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return torch_module.float16
    return torch_module.float32


def _force_cpu() -> bool:
    return os.getenv("CAREBRIDGE_FORCE_CPU", "").lower() in {"1", "true", "yes", "on"}


def _cuda_is_usable(torch_module: Any) -> bool:
    if _force_cpu() or not torch_module.cuda.is_available():
        return False
    try:
        major, _minor = torch_module.cuda.get_device_capability(0)
    except Exception:
        return False
    return major >= 7


def _preferred_device(torch_module: Any) -> str:
    if _cuda_is_usable(torch_module):
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _candidate_tasks(model_ref: str) -> list[str]:
    lowered = model_ref.lower()
    if "gemma-3" in lowered or "gemma3" in lowered or "gemma-4" in lowered or "gemma4" in lowered:
        return ["image-text-to-text", "text-generation"]
    return ["text-generation"]


def _is_gemma4(model_ref: str) -> bool:
    lowered = model_ref.lower()
    return "gemma-4" in lowered or "gemma4" in lowered


def _extract_text_from_chat_parts(parts: list[object]) -> str:
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, str):
            chunks.append(part)
            continue
        if not isinstance(part, dict):
            continue
        content = part.get("content")
        if isinstance(content, str):
            chunks.append(content)
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
    return "\n".join(chunk.strip() for chunk in chunks if chunk).strip()


def normalize_generation_payload(payload: object) -> str:
    current = payload
    if isinstance(current, list) and current:
        current = current[0]
    if isinstance(current, dict):
        generated = current.get("generated_text", current)
    else:
        generated = current

    if isinstance(generated, str):
        return generated.strip()
    if isinstance(generated, list):
        return _extract_text_from_chat_parts(generated)
    return str(generated).strip()


def normalize_gemma4_response(payload: object, raw_text: str) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            extracted = _extract_text_from_chat_parts(content)
            if extracted:
                return extracted
        for key in ("response", "text", "answer"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return raw_text.strip()


def _extract_text_parts(content: object) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return parts


def collapse_system_into_user(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    system_text: list[str] = []
    rewritten: list[dict[str, object]] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            system_text.extend(_extract_text_parts(message.get("content")))
            continue
        rewritten.append(message)

    if not system_text or not rewritten:
        return messages

    prefix = "Follow these system instructions exactly before answering:\n" + "\n".join(system_text)
    first = dict(rewritten[0])
    existing_content = first.get("content")
    merged_content: list[dict[str, str]] = [{"type": "text", "text": prefix}]
    if isinstance(existing_content, list):
        for item in existing_content:
            if isinstance(item, dict):
                merged_content.append(item)
    elif isinstance(existing_content, str):
        merged_content.append({"type": "text", "text": existing_content})
    first["content"] = merged_content
    rewritten[0] = first
    return rewritten


def extract_json_object(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found in model output")

    return json.loads(cleaned[first : last + 1])


def _normalize_section_header(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def _parse_list_block(text: str) -> list[str]:
    items: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line[:1].isdigit() and "." in line:
            _, _, line = line.partition(".")
            line = line.strip()
        if line.startswith(("-", "*")):
            line = line[1:].strip()
        if line:
            items.append(line)
    if items:
        return items
    return [chunk.strip() for chunk in text.split(";") if chunk.strip()]


def parse_labeled_draft(text: str) -> GemmaNarrativeDraft:
    canonical = {
        "summary": "summary",
        "why_now_in_america": "why_now_in_america",
        "why_this_matters_long_term": "why_this_matters_long_term",
        "rationale": "rationale",
        "next_steps": "next_steps",
        "follow_up_questions": "follow_up_questions",
    }
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_lines
        if current_key is not None:
            sections[current_key] = "\n".join(current_lines).strip()
        current_key = None
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_key is not None:
                current_lines.append("")
            continue
        if ":" in line:
            candidate, _, remainder = line.partition(":")
            normalized = _normalize_section_header(candidate)
            if normalized in canonical:
                flush()
                current_key = canonical[normalized]
                if remainder.strip():
                    current_lines.append(remainder.strip())
                continue
        if current_key is not None:
            current_lines.append(line)
    flush()

    if not sections:
        raise ValueError("No labeled sections found in model output")

    payload: dict[str, object] = {}
    for key in ("summary", "why_now_in_america", "why_this_matters_long_term"):
        payload[key] = sections.get(key, "").strip()
    for key in ("rationale", "next_steps", "follow_up_questions"):
        payload[key] = _parse_list_block(sections.get(key, ""))

    try:
        return GemmaNarrativeDraft.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Could not parse labeled Gemma draft: {exc}") from exc


def parse_gemma_draft(text: str) -> GemmaNarrativeDraft:
    try:
        payload = extract_json_object(text)
        return GemmaNarrativeDraft.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError):
        try:
            return parse_labeled_draft(text)
        except ValueError as exc:
            raise ValueError(f"Could not parse Gemma JSON draft: {exc}") from exc


class TransformersGemmaGenerator:
    def __init__(self, model_ref: str | None = None):
        self.model_ref = model_ref or find_gemma_model_ref()

    @cached_property
    def torch_module(self) -> Any:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "The Transformers Gemma stack is unavailable. "
                "Install a compatible transformers/huggingface-hub/torch set first."
            ) from exc
        return torch

    @cached_property
    def pipe(self) -> Any:
        from transformers import pipeline

        device = _preferred_device(self.torch_module)
        load_kwargs: dict[str, object] = {
            "model": self.model_ref,
            "torch_dtype": _preferred_dtype(self.torch_module),
        }
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            load_kwargs["token"] = token

        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        elif device == "mps":
            load_kwargs["device"] = self.torch_module.device("mps")
        else:
            load_kwargs["device"] = -1

        last_error: Exception | None = None
        for task in _candidate_tasks(self.model_ref):
            try:
                return pipeline(task=task, **load_kwargs)
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Unable to initialize a Gemma pipeline for {self.model_ref}. "
            f"Last error: {last_error}"
        )

    @cached_property
    def gemma4_bundle(self) -> tuple[Any, Any]:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = _preferred_device(self.torch_module)
        load_kwargs: dict[str, object] = {
            "torch_dtype": _preferred_dtype(self.torch_module),
            "low_cpu_mem_usage": True,
        }
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            load_kwargs["token"] = token

        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        elif device == "cpu":
            load_kwargs["device_map"] = "cpu"

        processor = AutoProcessor.from_pretrained(
            self.model_ref,
            token=token,
            padding_side="left",
        )
        model = AutoModelForImageTextToText.from_pretrained(self.model_ref, **load_kwargs)

        if device != "cuda":
            if device == "mps":
                model = model.to(self.torch_module.device("mps"))
            else:
                model = model.to("cpu")

        return processor, model

    def generate(
        self,
        messages: list[dict[str, object]],
        *,
        max_new_tokens: int = 700,
        temperature: float = 0.2,
    ) -> str:
        if _is_gemma4(self.model_ref):
            processor, model = self.gemma4_bundle
            chat_kwargs: dict[str, object] = {
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            try:
                inputs = processor.apply_chat_template(messages, **chat_kwargs)
            except TypeError:
                chat_kwargs.pop("enable_thinking", None)
                inputs = processor.apply_chat_template(messages, **chat_kwargs)
            inputs = inputs.to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            generation_kwargs: dict[str, object] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = 0.95
                generation_kwargs["top_k"] = 64

            outputs = model.generate(**inputs, **generation_kwargs)
            raw_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            if hasattr(processor, "parse_response"):
                parsed = processor.parse_response(raw_text)
                return normalize_gemma4_response(parsed, raw_text)
            return raw_text.strip()

        call_kwargs = {
            "return_full_text": False,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            call_kwargs["temperature"] = temperature

        try:
            result = self.pipe(messages, **call_kwargs)
        except Exception as exc:
            if "System role not supported" not in str(exc):
                raise
            result = self.pipe(collapse_system_into_user(messages), **call_kwargs)
        return normalize_generation_payload(result)


def merge_gemma_draft(
    *,
    request: AssessmentRequest,
    baseline: AssessmentResponse,
    draft: GemmaNarrativeDraft,
) -> AssessmentResponse:
    summary = draft.summary.strip()
    if len(summary) < 12:
        summary = baseline.summary

    why_now_in_america = draft.why_now_in_america.strip()
    if len(why_now_in_america) < 12:
        why_now_in_america = baseline.why_now_in_america

    why_this_matters_long_term = draft.why_this_matters_long_term.strip()
    if len(why_this_matters_long_term) < 12:
        why_this_matters_long_term = baseline.why_this_matters_long_term

    next_steps = [item.strip() for item in draft.next_steps if item.strip()]
    if len(next_steps) < 2:
        next_steps = baseline.next_steps

    rationale = [item.strip() for item in draft.rationale if item.strip()]
    if len(rationale) < 2:
        rationale = baseline.rationale

    follow_up_questions = [
        item.strip() for item in draft.follow_up_questions if item.strip()
    ]
    if len(follow_up_questions) < 2:
        follow_up_questions = baseline.follow_up_questions

    return baseline.model_copy(
        update={
            "mode": "gemma",
            "summary": summary,
            "why_now_in_america": why_now_in_america,
            "why_this_matters_long_term": why_this_matters_long_term,
            "rationale": rationale[:5],
            "next_steps": next_steps[:5],
            "follow_up_questions": follow_up_questions[:5],
            "handoff_card": baseline.handoff_card.model_copy(
                update={
                    "immediate_actions": next_steps[:3],
                }
            ),
        }
    )


def generate_gemma_assessment(
    *,
    request: AssessmentRequest,
    baseline: AssessmentResponse,
    grounded_sources: list[SourceExcerpt],
    generator: TextGenerator,
    max_new_tokens: int = 700,
    temperature: float = 0.2,
) -> AssessmentResponse:
    messages = build_gemma_messages(
        system_prompt=SYSTEM_PROMPT,
        request=request,
        baseline=baseline,
        grounded_sources=grounded_sources,
    )
    raw = generator.generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    try:
        draft = parse_gemma_draft(raw)
    except ValueError:
        repair_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Rewrite the assistant output into a strict labeled format only.\n"
                            "Use exactly these section headers:\n"
                            "SUMMARY:\n"
                            "WHY_NOW_IN_AMERICA:\n"
                            "WHY_THIS_MATTERS_LONG_TERM:\n"
                            "RATIONALE:\n"
                            "NEXT_STEPS:\n"
                            "FOLLOW_UP_QUESTIONS:\n"
                            "Use bullet points for the last three list sections."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": raw,
                    }
                ],
            },
        ]
        repaired = generator.generate(
            repair_messages,
            max_new_tokens=max_new_tokens,
            temperature=0,
        )
        try:
            draft = parse_gemma_draft(repaired)
        except ValueError as exc:
            raise ValueError(
                f"{exc}; raw={raw[:800]!r}; repaired={repaired[:800]!r}"
            ) from exc
    return merge_gemma_draft(
        request=request,
        baseline=baseline,
        draft=draft,
    )
