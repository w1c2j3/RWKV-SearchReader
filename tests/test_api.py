from __future__ import annotations

from dataclasses import replace

from rwkv_search_reader import api
from rwkv_search_reader.config import Settings


def test_request_slots_reject_work_past_the_concurrency_limit() -> None:
    assert api._acquire_request_slot(1, 0)
    try:
        assert not api._acquire_request_slot(1, 0)
    finally:
        api._release_request_slot()


def test_runtime_readiness_requires_reachable_model_endpoints(monkeypatch) -> None:
    settings = replace(
        Settings(),
        tavily_api_key="configured",
        rwkv_model="reader",
        rwkv_final_model="final",
        rwkv_final_base_url="http://127.0.0.1:29999/v1",
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "_endpoint_reachable", lambda base_url: False)

    payload = api.readiness()

    assert payload["rwkv_configured"] is True
    assert payload["ready"] is False
    assert payload["reader_model_reachable"] is False
    assert payload["final_model_reachable"] is False
