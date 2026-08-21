from __future__ import annotations

from dataclasses import replace

from rwkv_search_reader.compression import EvidenceCompressor, deterministic_compress
from rwkv_search_reader.config import Settings
from rwkv_search_reader.models import DocumentChunk


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            source_id="S1",
            title="Release notes",
            url="https://example.com/release",
            text="Version 1.0 was released on 2026-01-02. This is the official release note.",
            score=8.0,
        )
    ]


def test_deterministic_compression_keeps_literal_text_and_source() -> None:
    evidence = deterministic_compress(_chunks(), 4000)
    assert evidence[0].source_id == "S1"
    assert "released on 2026-01-02" in evidence[0].quote


def test_unconfigured_compressor_is_deterministic() -> None:
    settings = replace(
        Settings(),
        compressor_enabled=True,
        compressor_base_url="",
        compressor_api_key="",
        compressor_model="",
    )
    evidence, mode, warnings = EvidenceCompressor(settings).run(
        "release date?", _chunks()
    )
    assert mode == "deterministic"
    assert evidence
    assert warnings == []


def test_compressor_replaces_non_literal_model_quote() -> None:
    settings = replace(
        Settings(),
        compressor_enabled=True,
        compressor_base_url="https://model.example/v1",
        compressor_api_key="test",
        compressor_model="compressor",
    )
    compressor = EvidenceCompressor(settings)
    compressor.client.chat = lambda **kwargs: (
        '{"evidence":[{"source_id":"S1","claim":"released in 2026",'
        '"quote":"invented quotation"}]}'
    )
    evidence, mode, warnings = compressor.run("release date?", _chunks())
    assert mode == "model"
    assert evidence[0].quote == _chunks()[0].text
    assert evidence[0].claim == _chunks()[0].text[:500]
    assert warnings == []
