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

    def compress(question, chunks):
        calls.append("compress")
        return [evidence], "deterministic", []

    def read(question, rows):
        calls.append("read")
        return "Answer [S1]"

    pipeline.retriever.run = retrieve
    pipeline.compressor.run = compress
    pipeline.reader.run = read

    result = pipeline.answer("Question")

    assert calls == ["retrieve", "compress", "read"]
    assert result.retrieval_mode == "forced"
    assert result.queries == ["Question"]
    assert result.answer == "Answer [S1]"
    assert result.warnings == []
