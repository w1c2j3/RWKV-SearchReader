from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _prune_trace_files(
    directory: Path,
    current: Path,
    *,
    retention_days: int,
    max_total_bytes: int,
) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, retention_days))
    files = sorted(directory.glob("runs-*.jsonl"), key=lambda item: item.stat().st_mtime)
    for path in list(files):
        if path == current:
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified < cutoff:
            path.unlink(missing_ok=True)
            files.remove(path)
    total = sum(path.stat().st_size for path in files if path.exists())
    for path in files:
        if total <= max_total_bytes:
            break
        if path == current or not path.exists():
            continue
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        total -= size


def _compact_trace(row: dict[str, Any]) -> dict[str, Any]:
    sources = [
        {
            "source_id": str(source.get("source_id") or ""),
            "title": str(source.get("title") or ""),
            "url": str(source.get("url") or ""),
        }
        for source in row.get("sources") or []
        if isinstance(source, dict)
    ]
    return {
        "recorded_at": row.get("recorded_at"),
        "status": row.get("status", "ok"),
        "request_id": row.get("request_id"),
        "session_id": row.get("session_id"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "sources": sources,
        "trace": row.get("trace") or [],
        "timings_ms": row.get("timings_ms") or {},
        "degraded": bool(row.get("degraded")),
        "warnings": row.get("warnings") or [],
        "error": row.get("error"),
        "conversation_state": row.get("conversation_state") or {},
    }


def append_trace(directory: Path, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"runs-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        configuration = payload.get("configuration") or {}
        _prune_trace_files(
            directory,
            path,
            retention_days=int(configuration.get("trace_retention_days") or 30),
            max_total_bytes=int(
                configuration.get("trace_max_total_bytes") or 1_000_000_000
            ),
        )
    return path


def recent_traces(directory: Path, limit: int = 50) -> list[dict[str, Any]]:
    """Read newest durable run records without exposing raw page material."""

    records: list[dict[str, Any]] = []
    paths = sorted(directory.glob("runs-*.jsonl"), reverse=True)
    with _LOCK:
        for path in paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                records.append(_compact_trace(row))
                if len(records) >= limit:
                    return records
    return records


def session_records(
    directory: Path, session_id: str, limit: int = 60
) -> list[dict[str, Any]]:
    """Read compact records for one opaque session without exposing other chats."""

    identity = str(session_id or "").strip()
    if not identity:
        return []
    records: list[dict[str, Any]] = []
    paths = sorted(directory.glob("runs-*.jsonl"), reverse=True)
    with _LOCK:
        for path in paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    not isinstance(row, dict)
                    or str(row.get("session_id") or "") != identity
                ):
                    continue
                records.append(_compact_trace(row))
                if len(records) >= limit:
                    return records
    return records


def session_turns(
    directory: Path, session_id: str, limit: int = 6
) -> list[dict[str, Any]]:
    """Return prior successful turns for one opaque browser session, oldest first."""

    identity = str(session_id or "").strip()
    if not identity:
        return []
    turns: list[dict[str, Any]] = []
    paths = sorted(directory.glob("runs-*.jsonl"), reverse=True)
    with _LOCK:
        for path in paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    not isinstance(row, dict)
                    or row.get("status") != "ok"
                    or str(row.get("session_id") or "") != identity
                ):
                    continue
                question = str(row.get("question") or "").strip()
                answer = str(row.get("answer") or "").strip()
                if question and answer:
                    turn: dict[str, Any] = {"question": question, "answer": answer}
                    state = row.get("conversation_state")
                    if isinstance(state, dict) and state:
                        turn["state"] = state
                    turns.append(turn)
                if len(turns) >= limit:
                    return list(reversed(turns))
    return list(reversed(turns))
