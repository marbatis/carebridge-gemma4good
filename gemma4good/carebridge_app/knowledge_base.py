from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{2,}")
STOPWORDS = {
    "adult",
    "and",
    "are",
    "but",
    "family",
    "for",
    "from",
    "has",
    "have",
    "into",
    "not",
    "patient",
    "person",
    "since",
    "the",
    "their",
    "them",
    "they",
    "this",
    "with",
    "yesterday",
}


def normalize_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS
    }


@dataclass(frozen=True)
class KnowledgeDocument:
    source_id: str
    title: str
    summary: str
    body: str
    path: Path

    @property
    def tokens(self) -> set[str]:
        return normalize_tokens(" ".join([self.title, self.summary, self.body]))


@dataclass(frozen=True)
class RetrievedDocument:
    document: KnowledgeDocument
    score: float


def parse_markdown_document(path: Path) -> KnowledgeDocument:
    raw = path.read_text(encoding="utf-8")
    title = path.stem.replace("-", " ").title()
    summary = ""
    body_lines: list[str] = []

    for line in raw.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("Summary:"):
            summary = line.partition(":")[2].strip()
            continue
        body_lines.append(line)

    body = "\n".join(body_lines).strip()
    if not summary:
        summary = body.splitlines()[0].strip() if body else title

    return KnowledgeDocument(
        source_id=path.stem,
        title=title,
        summary=summary,
        body=body,
        path=path,
    )


class KnowledgeBase:
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self.documents = [
            parse_markdown_document(path)
            for path in sorted(docs_dir.glob("*.md"))
        ]

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDocument]:
        query_tokens = normalize_tokens(query)
        query_lower = query.lower()
        hits: list[RetrievedDocument] = []

        for document in self.documents:
            overlap = query_tokens & document.tokens
            if not overlap:
                continue

            title_hits = query_tokens & normalize_tokens(document.title)
            phrase_bonus = 0.0
            for phrase in (
                "chest pain",
                "trouble breathing",
                "shortness of breath",
                "high fever",
                "postpartum bleeding",
                "heavy bleeding",
                "severe headache",
                "no wet diapers",
                "wildfire smoke",
                "heat wave",
                "asthma",
            ):
                if phrase in query_lower and phrase in document.body.lower():
                    phrase_bonus += 3.0

            score = float(len(overlap) + (2 * len(title_hits)) + phrase_bonus)
            hits.append(RetrievedDocument(document=document, score=score))

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]
