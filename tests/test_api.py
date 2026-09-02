from __future__ import annotations

import json
from dataclasses import replace

from rwkv_search_reader import api
from rwkv_search_reader.config import Settings
from rwkv_search_reader.documents import MarkdownPassage


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


def test_document_selection_reads_all_chunks_only_for_whole_document_questions(
    monkeypatch,
) -> None:
    settings = replace(
        Settings(), document_max_chunks=2, document_full_read_max_chunks=10
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    passages = [
        MarkdownPassage("doc_alpha123", "alpha.pdf", index, "", f"段落 {index}")
        for index in range(5)
    ]

    class Store:
        def document_passages(self, conversation_id, client_id):
            assert conversation_id == "conv_alpha123"
            assert client_id == "client_alpha123"
            return passages

    full = api._select_document_passages(
        Store(), "conv_alpha123", "client_alpha123", "概括整篇论文", []
    )
    targeted = api._select_document_passages(
        Store(), "conv_alpha123", "client_alpha123", "第三节的结论是什么？", []
    )

    assert len(full) == 5
    assert len(targeted) == 2


def test_document_upload_stream_emits_progress_and_result(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_settings", Settings)

    class Store:
        def ensure(self, conversation_id, client_id):
            return None

        def add_document(self, *args, progress=None, **kwargs):
            assert progress is not None
            progress("pdf_converting", {"message": "正在还原标题"})
            return {"id": "doc_alpha123", "chunk_count": 7}

    monkeypatch.setattr(api, "_conversation_store", lambda: Store())
    request = api.DocumentUploadRequest(
        client_id="client_alpha123",
        conversation_id="conv_alpha123",
        filename="alpha.pdf",
        content_base64="JVBERi0xLjcK",
    )

    events = [json.loads(line) for line in api._document_upload_events(request)]

    assert events[0]["stage"] == "upload_received"
    assert any(event.get("stage") == "pdf_converting" for event in events)
    assert events[-1] == {
        "type": "result",
        "result": {"id": "doc_alpha123", "chunk_count": 7},
    }
