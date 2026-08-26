from __future__ import annotations

import os
import time

from rwkv_search_reader.tracing import (
    append_trace,
    recent_traces,
    session_records,
    session_turns,
)


def test_trace_records_are_durable_and_history_omits_raw_evidence(tmp_path) -> None:
    append_trace(
        tmp_path,
        {
            "status": "ok",
            "request_id": "request-1",
            "session_id": "session-1",
            "question": "Question",
            "answer": "Answer [S1]",
            "sources": [{"title": "Source", "url": "https://example.com"}],
            "evidence": [{"quote": "raw private page material"}],
            "model_calls": [{"input": "private prompt", "output": "raw output"}],
            "trace": [{"stage": "completed", "at_ms": 123.4, "detail": {}}],
            "timings_ms": {"total": 123.4},
        },
    )

    records = recent_traces(tmp_path)

    assert len(records) == 1
    assert records[0]["session_id"] == "session-1"
    assert records[0]["sources"] == [
        {"source_id": "", "title": "Source", "url": "https://example.com"}
    ]
    assert "evidence" not in records[0]
    assert "model_calls" not in records[0]


def test_session_turns_returns_only_matching_successful_conversation(tmp_path) -> None:
    append_trace(
        tmp_path,
        {
            "status": "ok",
            "session_id": "chat-a",
            "question": "第一问",
            "answer": "第一答",
        },
    )
    append_trace(
        tmp_path,
        {
            "status": "ok",
            "session_id": "chat-b",
            "question": "其他问题",
            "answer": "其他回答",
        },
    )
    append_trace(
        tmp_path,
        {
            "status": "ok",
            "session_id": "chat-a",
            "question": "第二问",
            "answer": "第二答",
        },
    )

    assert session_turns(tmp_path, "chat-a", 6) == [
        {"question": "第一问", "answer": "第一答"},
        {"question": "第二问", "answer": "第二答"},
    ]
    records = session_records(tmp_path, "chat-a", 6)
    assert [record["question"] for record in records] == ["第二问", "第一问"]
    assert all(record["session_id"] == "chat-a" for record in records)


def test_trace_retention_removes_expired_daily_files(tmp_path) -> None:
    expired = tmp_path / "runs-2020-01-01.jsonl"
    expired.write_text("{}\n", encoding="utf-8")
    old = time.time() - 5 * 86400
    os.utime(expired, (old, old))

    append_trace(
        tmp_path,
        {
            "status": "ok",
            "question": "新记录",
            "answer": "回答",
            "configuration": {
                "trace_retention_days": 1,
                "trace_max_total_bytes": 10_000_000,
            },
        },
    )

    assert not expired.exists()
