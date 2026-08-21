from __future__ import annotations

import time
import uuid

from .config import Settings, get_settings
from .models import PipelineResult, SearchReaderError
from .reader import RWKVReader, citation_warnings
from .retrieval import Retriever
from .search import create_search_provider
from .tracing import append_trace


class SearchReaderPipeline:
    """Forced Tavily -> parallel RWKV summaries -> ordered reduce -> final answer."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.provider = create_search_provider(self.settings)
        self.retriever = Retriever(self.settings, self.provider)
        self.reader = RWKVReader(self.settings)

    def answer(self, question: str) -> PipelineResult:
        normalized = " ".join(str(question or "").split()).strip()
        if not normalized:
            raise SearchReaderError("input", "question is empty")
        request_id = uuid.uuid4().hex
        timings: dict[str, float] = {}
        started = time.perf_counter()

        stage = time.perf_counter()
        queries, sources, chunks, retrieval_warnings = self.retriever.run(normalized)
        timings["retrieve"] = round((time.perf_counter() - stage) * 1000, 1)

        stage = time.perf_counter()
        evidence, summary_warnings = self.reader.summarize(normalized, chunks)
        timings["summarize"] = round((time.perf_counter() - stage) * 1000, 1)

        stage = time.perf_counter()
        answer, reduce_warnings = self.reader.finalize(normalized, evidence)
        timings["reduce"] = round((time.perf_counter() - stage) * 1000, 1)
        timings["total"] = round((time.perf_counter() - started) * 1000, 1)

        warnings = [
            *retrieval_warnings,
            *summary_warnings,
            *reduce_warnings,
            *citation_warnings(answer, evidence),
        ]
        result = PipelineResult(
            request_id=request_id,
            question=normalized,
            answer=answer,
            queries=queries,
            sources=sources,
            evidence=evidence,
            search_provider=self.provider.name,
            reader_mode="rwkv-parallel-map-reduce",
            degraded=bool(warnings),
            warnings=warnings,
            timings_ms=timings,
        )
        if self.settings.trace_enabled:
            append_trace(self.settings.trace_directory, result.to_dict())
        return result
