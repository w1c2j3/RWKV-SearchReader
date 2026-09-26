"""Durable, owner-scoped jobs. SQLite transactions serialize queue claims."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Response
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from typing import Literal

from .config import get_settings
from .retrieval import normalize_url


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class JobRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=201)
    locale: str = Field(default="zh-CN", pattern=r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*$", max_length=35)

    @model_validator(mode="after")
    def validate_history(self):
        if self.messages[-1].role != "user" or any(not m.content.strip() for m in self.messages):
            raise ValueError("messages must end with a non-empty user message")
        if sum(len(m.content) for m in self.messages) > 64000:
            raise ValueError("messages exceed 64000 characters")
        if any(m.role != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(self.messages)):
            raise ValueError("messages must alternate user and assistant")
        return self


class SourceResponse(BaseModel):
    id: str
    rank: int
    title: str
    url: str
    canonical_url: str
    snippet: str
    content_markdown: str
    content_status: Literal["fetched", "snippet_only"]
    published_at: int | None
    updated_at: int | None
    retrieved_at: str
    language: str | None


class AnswerResponse(BaseModel):
    role: Literal["assistant"]
    content: str
    results: list[SourceResponse]


class RunningResponse(BaseModel):
    job_id: str
    status: Literal["running"]


class DoneResponse(BaseModel):
    job_id: str
    status: Literal["done"]
    http_status: Literal[200]
    result: AnswerResponse


class ErrorResponse(BaseModel):
    code: str
    message: str


class FailedResponse(BaseModel):
    job_id: str
    status: Literal["failed"]
    http_status: int
    error: ErrorResponse


JobResponse = RunningResponse | DoneResponse | FailedResponse


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, idem TEXT, request TEXT NOT NULL,
                state TEXT NOT NULL, created REAL NOT NULL, deadline REAL NOT NULL,
                heartbeat REAL, finished REAL, payload TEXT, UNIQUE(owner, idem))""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, created)")

            db.execute("""CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL
                REFERENCES jobs(id) ON DELETE CASCADE, event TEXT NOT NULL, data TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS job_events_job ON job_events(job_id, id)")

    @staticmethod
    def _event(db, ident, event, data):
        db.execute("INSERT INTO job_events(job_id,event,data) VALUES(?,?,?)",
                   (ident, event, json.dumps(data, ensure_ascii=False)))

    def append_event(self, ident, event, data):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state,deadline FROM jobs WHERE id=?", (ident,)).fetchone()
            if row and row["state"] == "active" and row["deadline"] > time.time():
                self._event(db, ident, event, data)

    def events(self, owner, ident, after=0):
        with self.connection() as db:
            self.maintenance(db, time.time())
            row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=?", (ident, owner)).fetchone()
            if row is None:
                raise HTTPException(404, "Job not found")
            maximum = db.execute("SELECT COALESCE(MAX(id),0) FROM job_events WHERE job_id=?", (ident,)).fetchone()[0]
            if after > maximum:
                raise HTTPException(400, "Last-Event-ID exceeds this job's latest event")
            rows = db.execute("SELECT id,event,data FROM job_events WHERE job_id=? AND id>? ORDER BY id LIMIT 128", (ident, after)).fetchall()
            return self.public(row), [dict(item) for item in rows]

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def maintenance(self, db, now):
        if not db.in_transaction:
            db.execute("BEGIN IMMEDIATE")
        for code, message, condition in (
            ("job_timeout", "任务超过 15 分钟", "deadline <= ?"),
            ("worker_lost", "任务执行进程中断，请重新提交", "state = 'active' AND heartbeat < ?"),
        ):
            threshold = now if code == "job_timeout" else now - 60
            expired = db.execute(f"SELECT id FROM jobs WHERE state IN ('queued','active') AND {condition}", (threshold,)).fetchall()
            payload = {"http_status": 504 if code == "job_timeout" else 502, "error": {"code": code, "message": message}}
            for row in expired:
                db.execute("UPDATE jobs SET state='failed',finished=?,payload=? WHERE id=?", (now, json.dumps(payload), row["id"]))
                self._event(db, row["id"], "failed", {"job_id": row["id"], "status": "failed", **payload})
        db.execute("DELETE FROM jobs WHERE finished < ?", (now - 86400,))

    def submit(self, owner, request, idem=None):
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False)
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self.maintenance(db, now)
            if idem:
                old = db.execute("SELECT * FROM jobs WHERE owner=? AND idem=?", (owner, idem)).fetchone()
                if old:
                    if old["request"] != encoded:
                        raise HTTPException(409, "Idempotency-Key already used for another request")
                    return self.public(old)
            if db.execute("SELECT count(*) FROM jobs WHERE owner=? AND created>?", (owner, now-60)).fetchone()[0] >= 20:
                raise HTTPException(429, "Submission rate exceeded", headers={"Retry-After": "20"})
            if db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','active')").fetchone()[0] >= 100:
                raise HTTPException(503, "Queue full", headers={"Retry-After": "20"})
            ident = uuid.uuid4().hex
            db.execute("INSERT INTO jobs(id,owner,idem,request,state,created,deadline) VALUES(?,?,?,?,'queued',?,?)", (ident, owner, idem, encoded, now, now+900))
            self._event(db, ident, "progress", {"stage": "queued", "message": "任务已排队"})
        return {"job_id": ident, "status": "running"}

    @staticmethod
    def public(row):
        return {"job_id": row["id"], "status": "running" if row["state"] in {"queued", "active"} else row["state"], **(json.loads(row["payload"]) if row["payload"] else {})}

    def get(self, owner, ident):
        with self.connection() as db:
            self.maintenance(db, time.time())
            row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=?", (ident, owner)).fetchone()
            if row is None:
                raise HTTPException(404, "Job not found")
            return self.public(row)

    def claim(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            self.maintenance(db, now)
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET state='active', heartbeat=? WHERE id=?", (now, row["id"]))
                return dict(row)

    def heartbeat(self, ident):
        with self.connection() as db:
            return db.execute("UPDATE jobs SET heartbeat=? WHERE id=? AND state='active' AND deadline>?", (time.time(), ident, time.time())).rowcount == 1

    def finish(self, ident, payload, state="done"):
        with self.connection() as db:
            self.maintenance(db, time.time())
            changed = db.execute("UPDATE jobs SET state=?,payload=?,finished=? WHERE id=? AND state='active'", (state, json.dumps(payload, ensure_ascii=False), time.time(), ident)).rowcount
            if changed:
                self._event(db, ident, state, {"job_id": ident, "status": state, **payload})


def store():
    return JobStore(Path(os.environ.get("SEARCH_READER_JOB_DB", str(get_settings().trace_directory / "jobs.sqlite3"))))


def owner_for(authorization):
    keys = [v.strip() for v in os.environ.get("SEARCH_READER_API_KEYS", "").split(",") if v.strip()]
    import hmac
    token = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
    if not keys:
        raise HTTPException(503, "Job API keys are not configured")
    if not any(hmac.compare_digest(token, key) for key in keys):
        raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})
    return hashlib.sha256(token.encode()).hexdigest()


router = APIRouter()


_DECISION_SLOTS = threading.BoundedSemaphore(2)


class SearchDecisionResponse(BaseModel):
    needs_search: bool


@router.post("/v1/search/decision", response_model=SearchDecisionResponse)
def search_decision(body: JobRequest, response: Response, authorization: str = Header(default="")):
    owner_for(authorization)
    from .routing import decide_search
    from .models import SearchReaderError
    if not _DECISION_SLOTS.acquire(blocking=False):
        raise HTTPException(503, "Decision service is busy", headers={"Retry-After": "3"})
    try:
        decision = decide_search([m.model_dump() for m in body.messages])
    except SearchReaderError as exc:
        raise HTTPException(502, "Unable to determine whether search is needed; please retry") from exc
    finally:
        _DECISION_SLOTS.release()
    response.headers["Cache-Control"] = "no-store"
    return {"needs_search": decision}


@router.post("/v1/jobs", status_code=202, response_model=JobResponse)
def submit_job(body: JobRequest, response: Response, authorization: str = Header(default=""), idempotency_key: str | None = Header(default=None, max_length=128)):
    owner = owner_for(authorization)
    result = store().submit(owner, body.model_dump(), idempotency_key)
    response.headers["Retry-After"] = "20"
    response.headers["Location"] = f"/v1/jobs/{result['job_id']}"
    response.headers["Cache-Control"] = "no-store"
    return result


@router.get("/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, response: Response, authorization: str = Header(default="")):
    result = store().get(owner_for(authorization), job_id)
    response.headers["Cache-Control"] = "no-store"
    if result["status"] == "running":
        response.headers["Retry-After"] = "20"
    return result


@router.get("/v1/jobs/{job_id}/events")
def job_events(job_id: str, authorization: str = Header(default=""), last_event_id: str = Header(default="0")):
    owner = owner_for(authorization)
    if not last_event_id.isascii() or not last_event_id.isdigit() or len(last_event_id) > 18:
        raise HTTPException(400, "Last-Event-ID must be a non-negative integer")
    cursor = int(last_event_id)
    queue = store()
    queue.events(owner, job_id, cursor)  # Authenticate and validate before opening the stream.

    async def stream():
        nonlocal cursor
        heartbeat = time.monotonic()
        yield "retry: 2000\n\n"
        while True:
            try:
                job, events = await asyncio.to_thread(queue.events, owner, job_id, cursor)
            except HTTPException:
                yield 'event: failed\ndata: {"status":"failed","error":{"code":"job_expired","message":"任务已过期"}}\n\n'
                return
            for event in events:
                cursor = event["id"]
                yield f"id: {cursor}\nevent: {event['event']}\ndata: {event['data']}\n\n"
                heartbeat = time.monotonic()
            if job["status"] != "running" and len(events) < 128:
                # Jobs created before SSE was deployed have no terminal event.
                if not events and cursor == 0:
                    yield f"event: {job['status']}\ndata: {json.dumps(job, ensure_ascii=False)}\n\n"
                return
            if time.monotonic() - heartbeat >= 15:
                yield ": heartbeat\n\n"
                heartbeat = time.monotonic()
            await asyncio.sleep(0.25 if events else 0.5)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
    })


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int(parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp() * 1000)
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def serialize_result(result):
    return {"role": "assistant", "content": result.answer, "results": serialize_sources(result.sources)}


def serialize_sources(sources):
    return [
        {"id": hit.source_id, "rank": rank, "title": hit.title, "url": hit.url,
         "canonical_url": hit.canonical_url or normalize_url(hit.url), "snippet": hit.snippet,
         "content_markdown": hit.content_markdown, "content_status": hit.content_status,
         "published_at": timestamp(hit.published_date), "updated_at": None,
         "retrieved_at": hit.retrieved_at, "language": hit.language}
        for rank, hit in enumerate(sources, 1)]
