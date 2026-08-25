from __future__ import annotations

from rwkv_search_reader.config import get_settings


def test_final_answer_generation_is_hard_capped_at_4090(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_READER_RWKV_MAX_TOKENS", "9000")

    assert get_settings().rwkv_max_tokens == 4090
