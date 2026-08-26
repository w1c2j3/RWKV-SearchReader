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
