from __future__ import annotations

from dataclasses import replace

from rwkv_search_reader.config import Settings
from rwkv_search_reader.models import DocumentChunk, Evidence, SearchHit
from rwkv_search_reader.pipeline import SearchReaderPipeline


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

    def summarize(question, chunks):
        calls.append("summarize")
        return [evidence], []

    def finalize(question, rows):
        calls.append("reduce")
        return "Answer [S1]", []

    pipeline.retriever.run = retrieve
    pipeline.reader.summarize = summarize
    pipeline.reader.finalize = finalize

    progress = []
    result = pipeline.answer(
        "Question", on_progress=lambda stage, detail: progress.append((stage, detail))
    )

    assert calls == ["retrieve", "summarize", "reduce"]
    assert result.retrieval_mode == "forced"
    assert result.reader_mode == "rwkv-parallel-map-reduce"
    assert result.queries == ["Question"]
    assert result.answer == "Answer [S1]"
    assert result.warnings == []
    assert [stage for stage, _ in progress] == [
        "searching",
        "reading",
        "organizing",
    ]
    assert progress[1][1]["sources"] == [
        {"title": "Source", "url": "https://example.com"}
    ]
