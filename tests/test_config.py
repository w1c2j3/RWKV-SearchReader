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


def test_only_one_tavily_key_is_active(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "active-key")
    monkeypatch.setenv("TAVILY_API_KEYS", '["pool-key-1", "pool-key-2"]')

    settings = get_settings()

    assert settings.tavily_api_key == "active-key"
    assert settings.tavily_api_keys == ("active-key",)


def test_legacy_tavily_list_activates_only_its_first_key(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEYS", '["first-key", "second-key"]')

    settings = get_settings()

    assert settings.tavily_api_key == "first-key"
    assert settings.tavily_api_keys == ("first-key",)


def test_cloudflare_access_headers_are_configurable_without_logging_values(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_READER_RWKV_CF_ACCESS_CLIENT_ID", "test-id")
    monkeypatch.setenv("SEARCH_READER_RWKV_CF_ACCESS_CLIENT_SECRET", "test-secret")
    monkeypatch.delenv("SEARCH_READER_RWKV_FINAL_CF_ACCESS_CLIENT_ID", raising=False)
    monkeypatch.delenv("SEARCH_READER_RWKV_FINAL_CF_ACCESS_CLIENT_SECRET", raising=False)

    settings = get_settings()

    assert settings.rwkv_access_headers == {
        "CF-Access-Client-Id": "test-id",
        "CF-Access-Client-Secret": "test-secret",
    }
    assert settings.final_access_headers == settings.rwkv_access_headers


def test_final_cloudflare_credentials_can_override_primary(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_READER_RWKV_CF_ACCESS_CLIENT_ID", "primary-id")
    monkeypatch.setenv("SEARCH_READER_RWKV_CF_ACCESS_CLIENT_SECRET", "primary-secret")
    monkeypatch.setenv("SEARCH_READER_RWKV_FINAL_CF_ACCESS_CLIENT_ID", "final-id")
    monkeypatch.setenv("SEARCH_READER_RWKV_FINAL_CF_ACCESS_CLIENT_SECRET", "final-secret")

    assert get_settings().final_access_headers == {
        "CF-Access-Client-Id": "final-id",
        "CF-Access-Client-Secret": "final-secret",
    }


def test_conversation_context_supports_up_to_100_turns(monkeypatch) -> None:
    monkeypatch.delenv("SEARCH_READER_CONVERSATION_TURNS", raising=False)
    assert get_settings().conversation_turns == 100

    monkeypatch.setenv("SEARCH_READER_CONVERSATION_TURNS", "150")
    assert get_settings().conversation_turns == 100


def test_evidence_review_round_limit_is_configurable_to_10(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_READER_REVIEW_MAX_ROUNDS", "8")
    assert get_settings().review_max_rounds == 8

    monkeypatch.setenv("SEARCH_READER_REVIEW_MAX_ROUNDS", "20")
    assert get_settings().review_max_rounds == 10
