from __future__ import annotations

from dataclasses import replace

import pytest

from rwkv_search_reader.config import Settings
from rwkv_search_reader.models import (
    DocumentChunk,
    Evidence,
    EvidenceAudit,
    SearchHit,
    SearchReaderError,
)
from rwkv_search_reader.pipeline import SearchReaderPipeline, _conversation_reply
from rwkv_search_reader.tracing import append_trace, recent_traces


def test_pipeline_order_and_result_contract(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    calls = []
    hit = SearchHit(title="Source", url="https://example.com", snippet="Fact")
    chunk = DocumentChunk(
        source_id="S1", title=hit.title, url=hit.url, text="Fact", score=1
    )
    evidence = Evidence(
        source_id="S1", title=hit.title, url=hit.url, claim="Fact", quote="Fact"
    )

    def retrieve(question):
        calls.append("retrieve")
        return [question], [hit], [chunk], []

    def summarize(question, chunks, on_progress=None):
        calls.append("summarize")
        if on_progress:
            on_progress(
                "source_read",
                {"source_id": "S1", "completed": 1, "total": 1},
            )
        return [evidence], []

    def finalize(question, rows, on_progress=None):
        calls.append("finalize")
        if on_progress:
            on_progress("finalizing", {"source_count": 1})
        return "Answer [S1]", []

    def audit(question, rows, queries):
        calls.append("audit")
        return EvidenceAudit(sufficient=True)

    pipeline.retriever.run = retrieve
    pipeline.reader.summarize = summarize
    pipeline.reader.audit = audit
    pipeline.reader.finalize = finalize

    progress = []
    result = pipeline.answer(
        "Question",
        on_progress=lambda stage, detail: progress.append((stage, detail)),
        session_id="session-test",
    )

    assert calls == ["retrieve", "summarize", "audit", "finalize"]
    assert result.retrieval_mode == "forced"
    assert result.reader_mode == "rwkv-reviewed-parallel-read-final-summary"
    assert result.session_id == "session-test"
    assert result.queries == ["Question"]
    assert result.answer == "Answer [S1]"
    assert result.warnings == []
    assert [stage for stage, _ in progress] == [
        "searching",
        "sources_found",
        "reading",
        "source_read",
        "auditing",
        "audit_completed",
        "finalizing",
        "completed",
    ]
    assert progress[1][1]["sources"][0] == {
        "source_id": "S1",
        "title": "Source",
        "url": "https://example.com",
        "score": 0.0,
        "published_date": "",
    }
    assert result.trace[-1]["stage"] == "completed"


def test_pipeline_persists_failed_runs_with_trace(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=True,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)

    def fail_retrieval(question):
        raise SearchReaderError("search", "provider unavailable")

    pipeline.retriever.run = fail_retrieval

    with pytest.raises(SearchReaderError):
        pipeline.answer("Question", session_id="failed-session")

    record = recent_traces(tmp_path, 1)[0]
    assert record["status"] == "error"
    assert record["session_id"] == "failed-session"
    assert record["error"]["stage"] == "search"
    assert record["trace"][-1]["stage"] == "failed"


def test_exact_greeting_skips_search_and_model_calls(tmp_path) -> None:
    settings = replace(
        Settings(),
        trace_enabled=True,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    pipeline.retriever.run = lambda question: pytest.fail("search must not run")
    pipeline.reader.summarize = lambda *args, **kwargs: pytest.fail(
        "reader must not run"
    )
    progress = []

    result = pipeline.answer(
        "你好！",
        on_progress=lambda stage, detail: progress.append((stage, detail)),
        session_id="greeting-session",
    )

    assert result.answer == "你好！有什么想了解的，直接问我就好。"
    assert result.retrieval_mode == "local-conversation"
    assert result.queries == []
    assert result.sources == []
    assert [stage for stage, _ in progress] == ["responding", "completed"]
    saved = recent_traces(tmp_path, 1)[0]
    assert saved["trace"][-1]["stage"] == "completed"


def test_greeting_with_real_question_does_not_match_small_talk_rule() -> None:
    assert _conversation_reply("你好，请帮我查询 Python 最新版本") is None


def test_pipeline_uses_audit_query_to_add_new_sources(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
        review_max_rounds=2,
    )
    pipeline = SearchReaderPipeline(settings)
    first_hit = SearchHit(
        source_id="S1", title="First", url="https://example.com/1", content="A"
    )
    second_hit = SearchHit(
        source_id="S2", title="Second", url="https://example.com/2", content="B"
    )
    first_chunk = DocumentChunk("S1", "First", first_hit.url, "A")
    second_chunk = DocumentChunk("S2", "Second", second_hit.url, "B")

    pipeline.retriever.run = lambda question: (
        [question],
        [first_hit],
        [first_chunk],
        [],
    )
    supplemental_calls = []

    def supplement(queries, **kwargs):
        supplemental_calls.append((queries, kwargs))
        return queries, [second_hit], [second_chunk], []

    pipeline.retriever.run_queries = supplement

    def summarize(question, chunks, on_progress=None):
        del question, on_progress
        return [
            Evidence(
                source_id=chunk.source_id,
                title=chunk.title,
                url=chunk.url,
                claim=chunk.text,
                quote=chunk.text,
            )
            for chunk in chunks
        ], []

    pipeline.reader.summarize = summarize
    audits = iter(
        [
            EvidenceAudit(sufficient=False, queries=["missing fact query"]),
            EvidenceAudit(sufficient=True),
        ]
    )
    pipeline.reader.audit = lambda *args: next(audits)
    pipeline.reader.finalize = lambda question, rows, on_progress=None: (
        f"Answer from {len(rows)} sources [S1][S2]",
        [],
    )

    result = pipeline.answer("Question")

    assert result.answer == "Answer from 2 sources [S1][S2]"
    assert [source.source_id for source in result.sources] == ["S1", "S2"]
    assert result.queries == ["Question", "missing fact query"]
    assert supplemental_calls[0][1]["excluded_urls"] == {first_hit.url}
    assert supplemental_calls[0][1]["start_index"] == 2
    assert [event["stage"] for event in result.trace].count("auditing") == 2


def test_pipeline_resolves_followup_from_same_session_before_search(tmp_path) -> None:
    append_trace(
        tmp_path,
        {
            "status": "ok",
            "session_id": "continuous-chat",
            "question": "RWKV 最新版本是什么？",
            "answer": "最新版本是 RWKV-7-G1。",
        },
    )
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    contextualized = []

    def contextualize(question, history):
        contextualized.append((question, history))
        return "RWKV-7-G1 是什么时候发布的？"

    pipeline.reader.contextualize = contextualize
    hit = SearchHit(source_id="S1", title="Release", url="https://example.com/1")
    chunk = DocumentChunk("S1", hit.title, hit.url, "release fact")

    def retrieve(question):
        assert question == "RWKV-7-G1 是什么时候发布的？"
        return [question], [hit], [chunk], []

    pipeline.retriever.run = retrieve
    pipeline.reader.summarize = lambda *args, **kwargs: (
        [Evidence("S1", hit.title, hit.url, "release fact", "release fact")],
        [],
    )
    pipeline.reader.audit = lambda *args: EvidenceAudit(sufficient=True)
    pipeline.reader.finalize = lambda question, rows, on_progress=None: (
        "发布日期回答 [S1]",
        [],
    )

    result = pipeline.answer("它是什么时候发布的？", session_id="continuous-chat")

    assert contextualized[0][0] == "它是什么时候发布的？"
    assert contextualized[0][1][0]["question"] == "RWKV 最新版本是什么？"
    assert result.question == "它是什么时候发布的？"
    assert result.queries == ["RWKV-7-G1 是什么时候发布的？"]
    assert [event["stage"] for event in result.trace][:2] == [
        "contextualizing",
        "context_resolved",
    ]
