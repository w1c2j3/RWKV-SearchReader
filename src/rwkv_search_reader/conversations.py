from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .documents import (
    MarkdownPassage,
    PreparedDocument,
    prepare_document,
    rank_markdown_passages,
    split_markdown,
)

_IDENTITY_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_STORE_LOCKS_GUARD = threading.Lock()
_STORE_LOCKS: dict[str, threading.RLock] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _identity(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not _IDENTITY_RE.fullmatch(text):
        raise ValueError(f"invalid {label}")
    return text


def new_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex}"


def _context_terms(value: str) -> set[str]:
    text = str(value or "").casefold()
    terms = set(re.findall(r"[a-z][a-z0-9_.+-]{1,}", text))
    for segment in re.findall(r"[\u3400-\u9fff]{2,}", text):
        terms.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return terms


def _compact_source(source: object) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    return {
        key: source.get(key)
        for key in (
            "source_id",
            "title",
            "url",
            "provider",
            "score",
            "published_date",
            "error",
            "retrieval_point",
            "retrieval_query",
        )
        if source.get(key) not in (None, "")
    }


def _compact_trace(trace: object) -> list[dict[str, Any]]:
    if not isinstance(trace, list):
        return []
    compact: list[dict[str, Any]] = []
    for event in trace:
        if not isinstance(event, dict):
            continue
        row = dict(event)
        detail = row.get("detail")
        if isinstance(detail, dict):
            detail = dict(detail)
            if isinstance(detail.get("sources"), list):
                detail["sources"] = [
                    compact_source
                    for source in detail["sources"]
                    if (compact_source := _compact_source(source))
                ]
            row["detail"] = detail
        compact.append(row)
    return compact


def _compact_answer_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compact = {
        key: payload.get(key)
        for key in (
            "request_id",
            "queries",
            "reader_mode",
            "degraded",
            "warnings",
            "timings_ms",
            "conversation_state",
        )
        if key in payload
    }
    compact["sources"] = [
        compact_source
        for source in payload.get("sources") or []
        if (compact_source := _compact_source(source))
    ]
    compact["trace"] = _compact_trace(payload.get("trace"))
    return compact


def _answer_payload(result: dict[str, Any]) -> dict[str, Any]:
    return _compact_answer_payload(
        {
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
                "conversation_state",
            )
        }
    )


class ConversationStore:
    """Durable ChatGPT-style conversations with ordered message items."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        lock_key = str(self.path)
        with _STORE_LOCKS_GUARD:
            self._lock = _STORE_LOCKS.setdefault(lock_key, threading.RLock())
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
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    source_mime TEXT NOT NULL DEFAULT 'text/markdown',
                    source_blob BLOB,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    UNIQUE(conversation_id, sha256)
                );
                CREATE INDEX IF NOT EXISTS documents_conversation_created
                    ON documents(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    heading TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    UNIQUE(document_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS document_chunks_conversation
                    ON document_chunks(conversation_id, document_id, ordinal);
                """
            )
            document_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(documents)")
            }
            if "source_mime" not in document_columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN source_mime "
                    "TEXT NOT NULL DEFAULT 'text/markdown'"
                )
            if "source_blob" not in document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN source_blob BLOB")

    @staticmethod
    def _conversation(
        row: sqlite3.Row, message_count: int = 0, document_count: int = 0
    ) -> dict[str, Any]:
        return {
            "id": row["id"],
            "object": "conversation",
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": message_count,
            "document_count": document_count,
        }

    @staticmethod
    def _document(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "object": "conversation.document",
            "conversation_id": row["conversation_id"],
            "filename": row["filename"],
            "size_bytes": row["size_bytes"],
            "char_count": row["char_count"],
            "chunk_count": row["chunk_count"],
            "source_mime": row["source_mime"] or "text/markdown",
            "created_at": row["created_at"],
        }

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        payload = _compact_answer_payload(payload)
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
                document_count = connection.execute(
                    "SELECT COUNT(*) FROM documents WHERE conversation_id = ?",
                    (identity,),
                ).fetchone()[0]
                return self._conversation(row, count, document_count)

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
            return self._conversation(row, ordinal, 0)

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
            connection.execute("BEGIN IMMEDIATE")
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
                """SELECT c.*,
                          (SELECT COUNT(*) FROM messages m
                           WHERE m.conversation_id = c.id) AS message_count,
                          (SELECT COUNT(*) FROM documents d
                           WHERE d.conversation_id = c.id) AS document_count
                   FROM conversations c
                   WHERE c.client_id = ?
                     AND ((SELECT COUNT(*) FROM messages m
                           WHERE m.conversation_id = c.id) > 0
                          OR (SELECT COUNT(*) FROM documents d
                              WHERE d.conversation_id = c.id) > 0)
                   ORDER BY c.updated_at DESC
                   LIMIT ?""",
                (owner, limit),
            ).fetchall()
            return [
                self._conversation(row, row["message_count"], row["document_count"])
                for row in rows
            ]

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
            document_rows = connection.execute(
                "SELECT * FROM documents WHERE conversation_id = ? ORDER BY created_at",
                (identity,),
            ).fetchall()
            conversation["documents"] = [self._document(item) for item in document_rows]
            conversation["document_count"] = len(document_rows)
            return conversation

    def history(
        self,
        conversation_id: str,
        client_id: str,
        limit: int = 10,
        query: str = "",
    ) -> list[dict[str, Any]]:
        conversation = self.get(conversation_id, client_id)
        turns: list[dict[str, str]] = []
        pending = ""
        for item in conversation["items"]:
            if item["role"] == "user":
                pending = item["content"]
            elif item["role"] == "assistant" and pending:
                turn: dict[str, Any] = {
                    "question": pending,
                    "answer": item["content"],
                }
                state = item.get("payload", {}).get("conversation_state")
                if isinstance(state, dict) and state:
                    turn["state"] = state
                turns.append(turn)
                pending = ""
        if len(turns) <= limit:
            return turns
        recent_count = min(4, limit)
        recent_start = len(turns) - recent_count
        selected = set(range(recent_start, len(turns)))
        query_terms = _context_terms(query)
        preference_re = re.compile(r"我(?:喜欢|偏好|希望|需要|想要|不要)|我的")

        def score(index: int) -> tuple[float, int]:
            turn = turns[index]
            text = f"{turn['question']} {turn['answer']}"
            terms = _context_terms(text)
            overlap = len(query_terms & terms) / max(1, len(query_terms))
            preference = 0.2 if preference_re.search(turn["question"]) else 0.0
            return (overlap + preference, index)

        older_limit = max(0, limit - recent_count)
        ranked = sorted(range(recent_start), key=score, reverse=True)
        selected.update(ranked[:older_limit])
        return [turn for index, turn in enumerate(turns) if index in selected]

    def add_document(
        self,
        conversation_id: str,
        client_id: str,
        filename: str,
        content: str,
        *,
        max_bytes: int,
        chunk_chars: int,
        content_base64: str = "",
        max_pdf_bytes: int = 20_000_000,
        max_pdf_pages: int = 200,
    ) -> dict[str, Any]:
        prepared = prepare_document(
            filename,
            content,
            content_base64,
            max_markdown_bytes=max_bytes,
            max_pdf_bytes=max_pdf_bytes,
            max_pdf_pages=max_pdf_pages,
        )
        return self.add_prepared_document(
            conversation_id,
            client_id,
            prepared,
            chunk_chars=chunk_chars,
        )

    def add_prepared_document(
        self,
        conversation_id: str,
        client_id: str,
        prepared: PreparedDocument,
        *,
        chunk_chars: int,
    ) -> dict[str, Any]:
        """Persist a validated document that has already been converted."""

        identity = _identity(conversation_id, "conversation id")
        owner = _identity(client_id, "client id")
        clean_name = prepared.filename
        clean_content = prepared.markdown
        chunks = split_markdown(clean_content, chunk_chars)
        if not chunks:
            raise ValueError("document has no readable content")
        digest = hashlib.sha256(prepared.source_bytes).hexdigest()
        with self._lock, self._connect() as connection:
            conversation = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND client_id = ?",
                (identity, owner),
            ).fetchone()
            if conversation is None:
                raise PermissionError("conversation is unavailable")
            existing = connection.execute(
                "SELECT * FROM documents WHERE conversation_id = ? AND sha256 = ?",
                (identity, digest),
            ).fetchone()
            if existing is not None:
                return self._document(existing)
            document_id = f"doc_{uuid.uuid4().hex}"
            timestamp = _now()
            connection.execute(
                """INSERT INTO documents
                   (id, conversation_id, filename, content, sha256, size_bytes,
                    char_count, chunk_count, source_mime, source_blob, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    identity,
                    clean_name,
                    clean_content,
                    digest,
                    len(prepared.source_bytes),
                    len(clean_content),
                    len(chunks),
                    prepared.source_mime,
                    (
                        prepared.source_bytes
                        if prepared.source_mime == "application/pdf"
                        else None
                    ),
                    timestamp,
                ),
            )
            for ordinal, (heading, chunk) in enumerate(chunks):
                connection.execute(
                    """INSERT INTO document_chunks
                       (id, document_id, conversation_id, ordinal, heading, content)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        f"chunk_{uuid.uuid4().hex}",
                        document_id,
                        identity,
                        ordinal,
                        heading,
                        chunk,
                    ),
                )
            title = conversation["title"]
            if title == "新的对话":
                title = clean_name[:60]
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, timestamp, identity),
            )
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            return self._document(row)

    def list_documents(
        self, conversation_id: str, client_id: str
    ) -> list[dict[str, Any]]:
        conversation = self.get(conversation_id, client_id)
        return conversation["documents"]

    def search_documents(
        self,
        conversation_id: str,
        client_id: str,
        query: str,
        limit: int,
    ) -> list[MarkdownPassage]:
        identity = _identity(conversation_id, "conversation id")
        owner = _identity(client_id, "client id")
        with self._lock, self._connect() as connection:
            conversation = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND client_id = ?",
                (identity, owner),
            ).fetchone()
            if conversation is None:
                raise PermissionError("conversation is unavailable")
            rows = connection.execute(
                """SELECT d.id AS document_id, d.filename, c.ordinal,
                          c.heading, c.content
                   FROM document_chunks c
                   JOIN documents d ON d.id = c.document_id
                   WHERE c.conversation_id = ?
                   ORDER BY d.created_at, c.ordinal""",
                (identity,),
            ).fetchall()
        passages = [
            MarkdownPassage(
                document_id=row["document_id"],
                filename=row["filename"],
                ordinal=row["ordinal"],
                heading=row["heading"],
                content=row["content"],
            )
            for row in rows
        ]
        return rank_markdown_passages(query, passages, limit)
