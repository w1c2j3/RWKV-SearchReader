from __future__ import annotations

from dataclasses import replace

import pytest

from rwkv_search_reader.config import Settings
from rwkv_search_reader.models import (
    DocumentChunk,
    Evidence,
    SearchHit,
    SearchReaderError,
)
from rwkv_search_reader.pipeline import SearchReaderPipeline, _conversation_reply
from rwkv_search_reader.tracing import recent_traces


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
        calls.append("reduce")
        if on_progress:
            on_progress(
                "draft_complete",
                {"order": 1, "completed": 1, "total": 1},
            )
            on_progress("finalizing", {"candidate_groups": 1})
        return "Answer [S1]", []

    pipeline.retriever.run = retrieve
    pipeline.reader.summarize = summarize
    pipeline.reader.finalize = finalize

    progress = []
    result = pipeline.answer(
        "Question",
        on_progress=lambda stage, detail: progress.append((stage, detail)),
        session_id="session-test",
    )

    assert calls == ["retrieve", "summarize", "reduce"]
    assert result.retrieval_mode == "forced"
    assert result.reader_mode == "rwkv-parallel-map-reduce"
    assert result.session_id == "session-test"
    assert result.queries == ["Question"]
    assert result.answer == "Answer [S1]"
    assert result.warnings == []
    assert [stage for stage, _ in progress] == [
        "searching",
        "sources_found",
        "reading",
        "source_read",
        "organizing",
        "draft_complete",
        "finalizing",
        "completed",
    ]
    assert progress[1][1]["sources"][0] == {
        "source_id": "S1",
        "title": "Source",
        "url": "https://example.com",
        "score": 0.0,
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
