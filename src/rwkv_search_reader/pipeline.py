from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable

from .config import Settings, get_settings
from .models import PipelineResult, SearchReaderError
from .reader import RWKVReader, citation_warnings
from .retrieval import Retriever
from .search import create_search_provider
from .tracing import append_trace


def _conversation_reply(question: str) -> str | None:
    """Handle a tiny exact-match small-talk allow-list without search or a model."""

    canonical = re.sub(r"[\s,，。.!！?？~～]+", "", question).casefold()
    replies = {
        "你好": "你好！有什么想了解的，直接问我就好。",
        "您好": "你好！有什么想了解的，直接问我就好。",
        "嗨": "嗨！有什么想了解的吗？",
        "哈喽": "你好！有什么想了解的吗？",
        "hello": "你好！有什么想了解的吗？",
        "hi": "你好！有什么想了解的吗？",
        "在吗": "在的。你想了解什么？",
        "早上好": "早上好！今天想了解什么？",
        "下午好": "下午好！有什么可以帮你了解的？",
        "晚上好": "晚上好！有什么想了解的吗？",
        "谢谢": "不客气。还有什么想了解的吗？",
        "谢谢你": "不客气。还有什么想了解的吗？",
        "多谢": "不客气。还有什么想了解的吗？",
        "再见": "再见！需要时随时来问。",
        "拜拜": "再见！需要时随时来问。",
        "你是谁": "我是寻答，可以帮你查找资料、阅读来源并整理答案。",
        "你能做什么": "我可以帮你查找资料、阅读来源并整理答案。直接问一个具体问题就好。",
    }
    return replies.get(canonical)


class SearchReaderPipeline:
    """Deterministic greeting reply or Tavily -> parallel RWKV reader pipeline."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.provider = create_search_provider(self.settings)
        self.retriever = Retriever(self.settings, self.provider)
        self.reader = RWKVReader(self.settings)

    def answer(
        self,
        question: str,
        on_progress: Callable[[str, dict[str, object]], None] | None = None,
        session_id: str = "",
    ) -> PipelineResult:
        normalized = " ".join(str(question or "").split()).strip()
        if not normalized:
            raise SearchReaderError("input", "question is empty")
        request_id = uuid.uuid4().hex
        timings: dict[str, float] = {}
        started = time.perf_counter()
        trace: list[dict[str, object]] = []
        self.reader.reset_call_records()

        configuration = {
            "search_provider": self.settings.search_provider,
            "search_max_results": self.settings.search_max_results,
            "model": self.settings.rwkv_model,
            "protocol": self.settings.rwkv_protocol,
            "summary_workers": self.settings.rwkv_summary_workers,
            "reduce_orders": self.settings.rwkv_reduce_orders,
            "summary_max_tokens": self.settings.rwkv_summary_max_tokens,
            "max_tokens": self.settings.rwkv_max_tokens,
            "temperature": self.settings.rwkv_temperature,
        }

        def emit(stage_name: str, detail: dict[str, object]) -> None:
            at_ms = round((time.perf_counter() - started) * 1000, 1)
            event = {"stage": stage_name, "at_ms": at_ms, "detail": detail}
            trace.append(event)
            if on_progress:
                on_progress(stage_name, {**detail, "at_ms": at_ms})

        try:
            conversation_reply = _conversation_reply(normalized)
            if conversation_reply is not None:
                emit("responding", {"message": "直接回应简单对话"})
                timings["total"] = round((time.perf_counter() - started) * 1000, 1)
                emit("completed", {"duration_ms": timings["total"]})
                result = PipelineResult(
                    request_id=request_id,
                    session_id=session_id,
                    question=normalized,
                    answer=conversation_reply,
                    queries=[],
                    sources=[],
                    evidence=[],
                    retrieval_mode="local-conversation",
                    search_provider="",
                    reader_mode="deterministic-conversation",
                    degraded=False,
                    warnings=[],
                    timings_ms=timings,
                    trace=trace,
                )
                if self.settings.trace_enabled:
                    append_trace(
                        self.settings.trace_directory,
                        {
                            "schema_version": "rwkv-search-reader.trace.v2",
                            "status": "ok",
                            **result.to_dict(),
                            "configuration": configuration,
                            "model_calls": [],
                        },
                    )
                return result

            emit("searching", {"message": "正在查找相关资料"})
            stage = time.perf_counter()
            queries, sources, chunks, retrieval_warnings = self.retriever.run(
                normalized
            )
            timings["retrieve"] = round((time.perf_counter() - stage) * 1000, 1)
            emit(
                "sources_found",
                {
                    "queries": queries,
                    "source_count": len(chunks),
                    "sources": [
                        {
                            "source_id": f"S{index}",
                            "title": source.title,
                            "url": source.url,
                            "score": source.score,
                        }
                        for index, source in enumerate(sources[: len(chunks)], start=1)
                    ],
                    "duration_ms": timings["retrieve"],
                },
            )

            emit("reading", {"source_count": len(chunks)})
            stage = time.perf_counter()
            evidence, summary_warnings = self.reader.summarize(
                normalized, chunks, on_progress=emit
            )
            timings["summarize"] = round((time.perf_counter() - stage) * 1000, 1)

            emit(
                "organizing",
                {
                    "source_count": len(evidence),
                    "order_count": self.settings.rwkv_reduce_orders,
                },
            )
            stage = time.perf_counter()
            answer, reduce_warnings = self.reader.finalize(
                normalized, evidence, on_progress=emit
            )
            timings["reduce"] = round((time.perf_counter() - stage) * 1000, 1)
            timings["total"] = round((time.perf_counter() - started) * 1000, 1)
            emit("completed", {"duration_ms": timings["total"]})

            warnings = [
                *retrieval_warnings,
                *summary_warnings,
                *reduce_warnings,
                *citation_warnings(answer, evidence),
            ]
            result = PipelineResult(
                request_id=request_id,
                session_id=session_id,
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
                trace=trace,
            )
            if self.settings.trace_enabled:
                append_trace(
                    self.settings.trace_directory,
                    {
                        "schema_version": "rwkv-search-reader.trace.v2",
                        "status": "ok",
                        **result.to_dict(),
                        "configuration": configuration,
                        "model_calls": self.reader.call_records(),
                    },
                )
            return result
        except Exception as exc:
            timings["total"] = round((time.perf_counter() - started) * 1000, 1)
            error_stage = exc.stage if isinstance(exc, SearchReaderError) else "unexpected"
            emit(
                "failed",
                {
                    "error_stage": error_stage,
                    "message": str(exc)[:500],
                    "duration_ms": timings["total"],
                },
            )
            if self.settings.trace_enabled:
                try:
                    append_trace(
                        self.settings.trace_directory,
                        {
                            "schema_version": "rwkv-search-reader.trace.v2",
                            "status": "error",
                            "request_id": request_id,
                            "session_id": session_id,
                            "question": normalized,
                            "error": {
                                "stage": error_stage,
                                "type": type(exc).__name__,
                                "message": str(exc)[:1000],
                            },
                            "timings_ms": timings,
                            "trace": trace,
                            "configuration": configuration,
                            "model_calls": self.reader.call_records(),
                        },
                    )
                except OSError:
                    pass
            raise
