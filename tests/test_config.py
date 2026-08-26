from __future__ import annotations

from rwkv_search_reader.config import get_settings


def test_final_answer_generation_is_hard_capped_at_4090(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_READER_RWKV_MAX_TOKENS", "9000")

    assert get_settings().rwkv_max_tokens == 4090


def test_per_page_summary_defaults_to_about_one_thousand_tokens(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SEARCH_READER_RWKV_SUMMARY_MAX_TOKENS", raising=False)

    settings = get_settings()

    assert settings.rwkv_summary_max_tokens == 1024
    assert settings.readiness()["summary_max_tokens"] == 1024


def test_funnel_chunk_budgets_are_exposed_in_readiness(monkeypatch) -> None:
    monkeypatch.delenv("SEARCH_READER_RWKV_REDUCE_CHUNK_CHARS", raising=False)
    monkeypatch.delenv("SEARCH_READER_RWKV_REDUCE_FAN_IN", raising=False)
    monkeypatch.delenv("SEARCH_READER_RWKV_FINAL_MATERIAL_CHARS", raising=False)

    readiness = get_settings().readiness()

    assert readiness["reduce_chunk_chars"] == 6000
    assert readiness["reduce_fan_in"] == 4
    assert readiness["final_material_chars"] == 9000


def test_single_page_input_is_capped_below_the_16k_context(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_READER_MAX_EVIDENCE_CHARS", "80000")

    assert get_settings().max_evidence_chars == 12000
