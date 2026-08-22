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
from .tracing import append_trace, session_turns


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
        conversation_history: list[dict[str, str]] | None = None,
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
            "review_max_rounds": self.settings.review_max_rounds,
            "review_max_queries": self.settings.review_max_queries,
            "conversation_turns": self.settings.conversation_turns,
            "model": self.settings.rwkv_model,
            "protocol": self.settings.rwkv_protocol,
            "summary_workers": self.settings.rwkv_summary_workers,
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

            pipeline_warnings: list[str] = []
            history = (
                conversation_history[-self.settings.conversation_turns :]
                if conversation_history is not None
                else session_turns(
                    self.settings.trace_directory,
                    session_id,
                    self.settings.conversation_turns,
                )
            )
            effective_question = normalized
            if history:
                emit(
                    "contextualizing",
                    {"turn_count": len(history), "message": "正在衔接此前对话"},
                )
                try:
                    effective_question = self.reader.contextualize(normalized, history)
                except SearchReaderError as exc:
                    pipeline_warnings.append(f"conversation context degraded: {exc}"[:500])
                emit(
                    "context_resolved",
                    {
                        "turn_count": len(history),
                        "resolved_question": effective_question,
                    },
                )

            emit("searching", {"message": "正在查找相关资料"})
            stage = time.perf_counter()
            queries, sources, chunks, retrieval_warnings = self.retriever.run(
                effective_question
            )
            timings["retrieve"] = round((time.perf_counter() - stage) * 1000, 1)
            emit(
                "sources_found",
                {
                    "queries": queries,
                    "source_count": len(chunks),
                    "sources": [
                        {
                            "source_id": source.source_id or f"S{index}",
                            "title": source.title,
                            "url": source.url,
                            "score": source.score,
                            "published_date": source.published_date,
                        }
                        for index, source in enumerate(sources[: len(chunks)], start=1)
                    ],
                    "duration_ms": timings["retrieve"],
                },
            )

            emit("reading", {"source_count": len(chunks)})
            stage = time.perf_counter()
            evidence, summary_warnings = self.reader.summarize(
                effective_question, chunks, on_progress=emit
            )
            timings["summarize"] = round((time.perf_counter() - stage) * 1000, 1)

            audit_warnings: list[str] = []
            audit_started = time.perf_counter()
            for review_index in range(self.settings.review_max_rounds + 1):
                emit(
                    "auditing",
                    {
                        "round": review_index + 1,
                        "source_count": len(evidence),
                        "message": "正在检查材料是否足够",
                    },
                )
                try:
                    audit = self.reader.audit(effective_question, evidence, queries)
                except SearchReaderError as exc:
                    audit_warnings.append(f"evidence audit degraded: {exc}"[:500])
                    emit(
                        "audit_completed",
                        {
                            "round": review_index + 1,
                            "sufficient": True,
                            "degraded": True,
                            "message": "审核结果不可用，使用现有材料",
                        },
                    )
                    break

                if audit.sufficient:
                    emit(
                        "audit_completed",
                        {
                            "round": review_index + 1,
                            "sufficient": True,
                            "message": "现有材料足以回答",
                        },
                    )
                    break

                tried = {query.casefold() for query in queries}
                supplemental_queries = [
                    query for query in audit.queries if query.casefold() not in tried
                ]
                emit(
                    "audit_completed",
                    {
                        "round": review_index + 1,
                        "sufficient": False,
                        "queries": supplemental_queries,
                        "message": "材料仍有缺口，需要补充查找",
                    },
                )
                if review_index >= self.settings.review_max_rounds:
                    audit_warnings.append(
                        "evidence remained incomplete after the maximum review rounds"
                    )
                    break
                if not supplemental_queries:
                    audit_warnings.append(
                        "evidence audit requested retrieval without a new query"
                    )
                    break

                supplement_round = review_index + 1
                emit(
                    "supplemental_searching",
                    {
                        "round": supplement_round,
                        "queries": supplemental_queries,
                        "message": "正在补充缺失资料",
                    },
                )
                stage = time.perf_counter()
                try:
                    (
                        new_queries,
                        new_sources,
                        new_chunks,
                        new_retrieval_warnings,
                    ) = self.retriever.run_queries(
                        supplemental_queries,
                        question=effective_question,
                        excluded_urls={source.url for source in sources},
                        start_index=len(sources) + 1,
                    )
                except SearchReaderError as exc:
                    audit_warnings.append(
                        f"supplemental retrieval round {supplement_round} failed: {exc}"[
                            :500
                        ]
                    )
                    emit(
                        "supplemental_sources_found",
                        {
                            "round": supplement_round,
                            "source_count": 0,
                            "total_source_count": len(sources),
                            "message": "没有找到新的可用材料",
                        },
                    )
                    break
                timings["retrieve"] += round(
                    (time.perf_counter() - stage) * 1000, 1
                )
                queries.extend(new_queries)
                sources.extend(new_sources)
                retrieval_warnings.extend(new_retrieval_warnings)
                emit(
                    "supplemental_sources_found",
                    {
                        "round": supplement_round,
                        "queries": new_queries,
                        "source_count": len(new_chunks),
                        "total_source_count": len(sources),
                        "sources": [
                            {
                                "source_id": source.source_id,
                                "title": source.title,
                                "url": source.url,
                                "score": source.score,
                                "published_date": source.published_date,
                            }
                            for source in new_sources[: len(new_chunks)]
                        ],
                    },
                )
                emit(
                    "supplemental_reading",
                    {
                        "round": supplement_round,
                        "source_count": len(new_chunks),
                        "total_source_count": len(sources),
                    },
                )
                stage = time.perf_counter()
                new_evidence, new_summary_warnings = self.reader.summarize(
                    effective_question, new_chunks, on_progress=emit
                )
                timings["summarize"] += round(
                    (time.perf_counter() - stage) * 1000, 1
                )
                evidence.extend(new_evidence)
                summary_warnings.extend(new_summary_warnings)

            timings["audit"] = round(
                (time.perf_counter() - audit_started) * 1000, 1
            )

            stage = time.perf_counter()
            answer, final_warnings = self.reader.finalize(
                effective_question, evidence, on_progress=emit
            )
            timings["finalize"] = round((time.perf_counter() - stage) * 1000, 1)
            timings["total"] = round((time.perf_counter() - started) * 1000, 1)
            emit("completed", {"duration_ms": timings["total"]})

            warnings = [
                *pipeline_warnings,
                *retrieval_warnings,
                *summary_warnings,
                *audit_warnings,
                *final_warnings,
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
                reader_mode="rwkv-reviewed-parallel-read-final-summary",
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
