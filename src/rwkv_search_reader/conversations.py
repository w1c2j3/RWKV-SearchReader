from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_IDENTITY_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _identity(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not _IDENTITY_RE.fullmatch(text):
        raise ValueError(f"invalid {label}")
    return text


def new_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex}"


def _answer_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "request_id",
            "sources",
            "queries",
            "reader_mode",
            "degraded",
            "warnings",
            "timings_ms",
            "trace",
        )
    }


class ConversationStore:
    """Durable ChatGPT-style conversations with ordered message items."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversations_client_updated
                    ON conversations(client_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    request_id TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    UNIQUE(conversation_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS messages_conversation_ordinal
                    ON messages(conversation_id, ordinal);
                """
            )

    @staticmethod
    def _conversation(row: sqlite3.Row, message_count: int = 0) -> dict[str, Any]:
        return {
            "id": row["id"],
            "object": "conversation",
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": message_count,
        }

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        return {
            "id": row["id"],
            "object": "conversation.message",
            "conversation_id": row["conversation_id"],
            "ordinal": row["ordinal"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
            "request_id": row["request_id"],
            "payload": payload,
        }

    def ensure(
        self,
        conversation_id: str,
        client_id: str,
        *,
        legacy_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        identity = _identity(conversation_id, "conversation id")
        owner = _identity(client_id, "client id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (identity,)
            ).fetchone()
            if row is not None:
                if row["client_id"] != owner:
                    raise PermissionError("conversation belongs to another client")
                count = connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (identity,),
                ).fetchone()[0]
                return self._conversation(row, count)

            timestamp = _now()
            ordered_legacy = list(reversed(legacy_records or []))
            first_question = next(
                (
                    str(record.get("question") or "").strip()
                    for record in ordered_legacy
                    if str(record.get("question") or "").strip()
                ),
                "",
            )
            title = first_question[:60] or "新的对话"
            connection.execute(
                """INSERT INTO conversations
                   (id, client_id, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (identity, owner, title, timestamp, timestamp),
            )
            ordinal = 0
            last_time = timestamp
            for record in ordered_legacy:
                question = str(record.get("question") or "").strip()
                answer = str(record.get("answer") or "").strip()
                if not question or not answer or record.get("status", "ok") != "ok":
                    continue
                created_at = str(record.get("recorded_at") or timestamp)
                request_id = str(record.get("request_id") or "")
                for role, content, payload in (
                    ("user", question, {}),
                    ("assistant", answer, _answer_payload(record)),
                ):
                    connection.execute(
                        """INSERT INTO messages
                           (id, conversation_id, ordinal, role, content, created_at,
                            request_id, payload_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            f"msg_{uuid.uuid4().hex}",
                            identity,
                            ordinal,
                            role,
                            content,
                            created_at,
                            request_id,
                            json.dumps(payload, ensure_ascii=False),
                        ),
                    )
                    ordinal += 1
                last_time = created_at
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (last_time, identity),
            )
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (identity,)
            ).fetchone()
            return self._conversation(row, ordinal)

    def append_turn(
        self,
        conversation_id: str,
        client_id: str,
        question: str,
        result: dict[str, Any],
    ) -> None:
        identity = _identity(conversation_id, "conversation id")
        owner = _identity(client_id, "client id")
        with self._lock, self._connect() as connection:
            conversation = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (identity,)
            ).fetchone()
            if conversation is None or conversation["client_id"] != owner:
                raise PermissionError("conversation is unavailable")
            ordinal = connection.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM messages "
                "WHERE conversation_id = ?",
                (identity,),
            ).fetchone()[0]
            timestamp = _now()
            request_id = str(result.get("request_id") or "")
            answer = str(result.get("answer") or "").strip()
            for offset, role, content, payload in (
                (0, "user", question, {}),
                (1, "assistant", answer, _answer_payload(result)),
            ):
                connection.execute(
                    """INSERT INTO messages
                       (id, conversation_id, ordinal, role, content, created_at,
                        request_id, payload_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"msg_{uuid.uuid4().hex}",
                        identity,
                        ordinal + offset,
                        role,
                        content,
                        timestamp,
                        request_id,
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
            title = conversation["title"]
            if ordinal == 0 or title == "新的对话":
                title = question[:60]
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, timestamp, identity),
            )

    def list(self, client_id: str, limit: int = 60) -> list[dict[str, Any]]:
        owner = _identity(client_id, "client id")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT c.*, COUNT(m.id) AS message_count
                   FROM conversations c
                   LEFT JOIN messages m ON m.conversation_id = c.id
                   WHERE c.client_id = ?
                   GROUP BY c.id
                   HAVING COUNT(m.id) > 0
                   ORDER BY c.updated_at DESC
                   LIMIT ?""",
                (owner, limit),
            ).fetchall()
            return [self._conversation(row, row["message_count"]) for row in rows]

    def get(self, conversation_id: str, client_id: str) -> dict[str, Any]:
        identity = _identity(conversation_id, "conversation id")
        owner = _identity(client_id, "client id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND client_id = ?",
                (identity, owner),
            ).fetchone()
            if row is None:
                raise KeyError("conversation not found")
            message_rows = connection.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY ordinal",
                (identity,),
            ).fetchall()
            conversation = self._conversation(row, len(message_rows))
            conversation["items"] = [self._message(item) for item in message_rows]
            return conversation

    def history(
        self, conversation_id: str, client_id: str, limit: int = 6
    ) -> list[dict[str, str]]:
        conversation = self.get(conversation_id, client_id)
        turns: list[dict[str, str]] = []
        pending = ""
        for item in conversation["items"]:
            if item["role"] == "user":
                pending = item["content"]
            elif item["role"] == "assistant" and pending:
                turns.append({"question": pending, "answer": item["content"]})
                pending = ""
        return turns[-limit:]
