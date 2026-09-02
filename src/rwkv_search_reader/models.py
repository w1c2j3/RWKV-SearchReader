from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SearchHit:
    title: str
    url: str
    source_id: str = ""
    snippet: str = ""
    content: str = ""
    provider: str = ""
    score: float = 0.0
    published_date: str = ""
    error: str = ""
    retrieval_point: str = ""
    retrieval_query: str = ""


@dataclass(slots=True)
class DocumentChunk:
    source_id: str
    title: str
    url: str
    text: str
    score: float = 0.0
    chunk_index: int = 0
    published_date: str = ""
    retrieval_point: str = ""
    retrieval_query: str = ""


@dataclass(slots=True)
class Evidence:
    source_id: str
    title: str
    url: str
    claim: str
    quote: str
    score: float = 0.0
    source_relation: str = "unclear"
    published_date: str = ""
    retrieval_point: str = ""
    retrieval_query: str = ""


@dataclass(slots=True)
class EvidenceAudit:
    sufficient: bool
    queries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchRecord:
    subject: str
    question: str
    fields: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchPlan:
    decision: str
    goal: str
    records: list[ResearchRecord] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    scope_mode: str = "fixed"
    clarification: str = ""
    fallback: bool = False
    retrieval_points: list[str] = field(default_factory=list)
    direct_urls: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FieldEvidenceSelection:
    subject: str
    field: str
    source_id: str
    status: str
    fact: str = ""
    source_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PipelineResult:
    request_id: str
    session_id: str
    question: str
    answer: str
    queries: list[str]
    sources: list[SearchHit]
    evidence: list[Evidence]
    retrieval_mode: str = "forced"
    search_provider: str = ""
    reader_mode: str = ""
    degraded: bool = False
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    conversation_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchReaderError(RuntimeError):
    """Base error carrying the pipeline stage that failed."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
