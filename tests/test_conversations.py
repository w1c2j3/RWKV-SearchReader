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


def test_conversation_store_persists_and_searches_markdown(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation_id = "conv_documents123"
    client_id = "client_documents123"
    store.ensure(conversation_id, client_id)

    document = store.add_document(
        conversation_id,
        client_id,
        "tea.md",
        "# 偏好\n\n我喜欢回甘明显的茶。\n\n# 预算\n\n每斤预算五百元。",
        max_bytes=10_000,
        chunk_chars=600,
    )
    repeated = store.add_document(
        conversation_id,
        client_id,
        "tea-copy.md",
        "# 偏好\n\n我喜欢回甘明显的茶。\n\n# 预算\n\n每斤预算五百元。",
        max_bytes=10_000,
        chunk_chars=600,
    )

    passages = store.search_documents(
        conversation_id, client_id, "预算是多少", limit=2
    )
    listed = store.list(client_id)
    restored = store.get(conversation_id, client_id)

    assert document["id"] == repeated["id"]
    assert document["chunk_count"] == 2
    assert passages[0].heading == "预算"
    assert listed[0]["document_count"] == 1
    assert listed[0]["message_count"] == 0
    assert restored["documents"][0]["filename"] == "tea.md"


def test_conversation_history_keeps_relevant_older_turns(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation_id = "conv_context123"
    client_id = "client_context123"
    store.ensure(conversation_id, client_id)
    for index in range(8):
        question = "我喜欢回甘明显的茶" if index == 0 else f"普通问题 {index}"
        store.append_turn(
            conversation_id,
            client_id,
            question,
            {"request_id": f"request-{index}", "answer": f"回答 {index}"},
        )

    history = store.history(
        conversation_id, client_id, limit=5, query="按我的回甘偏好推荐"
    )

    assert history[0]["question"] == "我喜欢回甘明显的茶"
    assert [turn["question"] for turn in history[-4:]] == [
        "普通问题 4",
        "普通问题 5",
        "普通问题 6",
        "普通问题 7",
    ]
