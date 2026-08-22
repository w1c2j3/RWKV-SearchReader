from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


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
                sources = [
                    {
                        "title": str(source.get("title") or ""),
                        "url": str(source.get("url") or ""),
                    }
                    for source in row.get("sources") or []
                    if isinstance(source, dict)
                ]
                records.append(
                    {
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
                    }
                )
                if len(records) >= limit:
                    return records
    return records


def session_turns(
    directory: Path, session_id: str, limit: int = 6
) -> list[dict[str, str]]:
    """Return prior successful turns for one opaque browser session, oldest first."""

    identity = str(session_id or "").strip()
    if not identity:
        return []
    turns: list[dict[str, str]] = []
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
                    turns.append({"question": question, "answer": answer})
                if len(turns) >= limit:
                    return list(reversed(turns))
    return list(reversed(turns))
