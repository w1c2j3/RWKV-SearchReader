from __future__ import annotations

import pytest

from rwkv_search_reader.conversations import ConversationStore


def test_conversation_store_persists_ordered_message_items(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation_id = "conv_test123"
    client_id = "client_test123"

    created = store.ensure(conversation_id, client_id)
    store.append_turn(
        conversation_id,
        client_id,
        "第一问",
        {
            "request_id": "request-1",
            "answer": "第一答",
            "sources": [{"source_id": "S1", "title": "Source", "url": "https://x"}],
            "trace": [{"stage": "completed"}],
            "timings_ms": {"total": 10},
        },
    )
    store.append_turn(
        conversation_id,
        client_id,
        "第二问",
        {"request_id": "request-2", "answer": "第二答"},
    )

    conversation = store.get(conversation_id, client_id)
    listed = store.list(client_id)

    assert created["object"] == "conversation"
    assert conversation["title"] == "第一问"
    assert [item["role"] for item in conversation["items"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [item["content"] for item in conversation["items"]] == [
        "第一问",
        "第一答",
        "第二问",
        "第二答",
    ]
    assert conversation["items"][1]["payload"]["sources"][0]["source_id"] == "S1"
    assert listed[0]["id"] == conversation_id
    assert listed[0]["message_count"] == 4
    assert store.history(conversation_id, client_id) == [
        {"question": "第一问", "answer": "第一答"},
        {"question": "第二问", "answer": "第二答"},
    ]


def test_conversation_store_imports_legacy_session_once(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    records = [
        {
            "status": "ok",
            "recorded_at": "2026-08-22T02:00:00+00:00",
            "request_id": "request-2",
            "question": "第二问",
            "answer": "第二答",
        },
        {
            "status": "ok",
            "recorded_at": "2026-08-22T01:00:00+00:00",
            "request_id": "request-1",
            "question": "第一问",
            "answer": "第一答",
        },
    ]

    imported = store.ensure(
        "legacy_session123", "client_test123", legacy_records=records
    )
    repeated = store.ensure(
        "legacy_session123", "client_test123", legacy_records=records
    )

    assert imported["title"] == "第一问"
    assert imported["message_count"] == 4
    assert repeated["message_count"] == 4
    assert store.history("legacy_session123", "client_test123")[-1] == {
        "question": "第二问",
        "answer": "第二答",
    }


def test_conversation_store_enforces_browser_ownership(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    store.ensure("conv_test123", "client_owner123")

    with pytest.raises(PermissionError):
        store.ensure("conv_test123", "client_other123")
