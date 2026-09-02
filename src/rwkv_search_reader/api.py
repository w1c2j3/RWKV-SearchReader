from __future__ import annotations

import ipaddress
import json
import queue
import re
import socket
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.gzip import GZipMiddleware

from . import __version__
from .config import PROJECT_ROOT, get_settings
from .conversations import ConversationStore, new_conversation_id
from .documents import (
    MarkdownPassage,
    full_document_read_requested,
    limit_full_document_passages,
    rank_markdown_passages,
)
from .models import SearchReaderError
from .pipeline import SearchReaderPipeline
from .tracing import recent_traces, session_records

_DOCUMENT_CONTEXT_REFERENCE_RE = re.compile(
    r"^\s*(?:它|它们|这个|那个|这些|那些|这份|这篇|该(?:文档|版本|项目))|"
    r"上述|前者|后者"
)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(default="", max_length=128)
    conversation_id: str = Field(default="", max_length=128)
    client_id: str = Field(default="", max_length=128)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False


class SessionHistoryRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=128)
    limit: int = Field(default=60, ge=1, le=200)


class ConversationAccessRequest(BaseModel):
    client_id: str = Field(min_length=8, max_length=128)
    conversation_id: str = Field(min_length=8, max_length=128)


class ConversationListRequest(BaseModel):
    client_id: str = Field(min_length=8, max_length=128)
    limit: int = Field(default=60, ge=1, le=200)


class DocumentUploadRequest(BaseModel):
    client_id: str = Field(min_length=8, max_length=128)
    conversation_id: str = Field(min_length=8, max_length=128)
    filename: str = Field(min_length=1, max_length=180)
    content: str = Field(default="", max_length=30_000_000)
    content_base64: str = Field(default="", max_length=70_000_000)


app = FastAPI(
    title="RWKV SearchReader",
    version=__version__,
    description="Deterministic small-talk bypass plus retrieve-then-read answers.",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

_STORE_GUARD = threading.Lock()
_STORES: dict[str, ConversationStore] = {}
_REQUEST_CONDITION = threading.Condition()
_ACTIVE_REQUESTS = 0
_RATE_GUARD = threading.Lock()
_RATE_WINDOWS: dict[str, deque[float]] = defaultdict(deque)


def _new_pipeline() -> SearchReaderPipeline:
    return SearchReaderPipeline(get_settings())


def _conversation_store() -> ConversationStore:
    settings = get_settings()
    path = (settings.trace_directory / "conversations.sqlite3").resolve()
    key = str(path)
    with _STORE_GUARD:
        store = _STORES.get(key)
        if store is None:
            store = ConversationStore(path)
            _STORES[key] = store
        return store


def _select_document_passages(
    store: ConversationStore,
    conversation_id: str,
    client_id: str,
    question: str,
    history: list[dict[str, Any]],
) -> list[MarkdownPassage]:
    settings = get_settings()
    passages = store.document_passages(conversation_id, client_id)
    if not passages:
        return []
    if full_document_read_requested(question):
        return limit_full_document_passages(
            passages, settings.document_full_read_max_chunks
        )
    context_questions = (
        [turn["question"] for turn in history[-1:]]
        if _DOCUMENT_CONTEXT_REFERENCE_RE.search(question)
        else []
    )
    document_query = " ".join([*context_questions, question])
    return rank_markdown_passages(
        document_query, passages, settings.document_max_chunks
    )


def _request_identity(request: Request) -> str:
    forwarded = str(request.headers.get("cf-connecting-ip") or "").strip()
    return forwarded or (request.client.host if request.client else "unknown")


def _enforce_rate_limit(request: Request) -> None:
    settings = get_settings()
    identity = _request_identity(request)
    now = time.monotonic()
    with _RATE_GUARD:
        window = _RATE_WINDOWS[identity]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= settings.api_rate_limit_per_minute:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
        window.append(now)


def _acquire_request_slot(limit: int, timeout: int) -> bool:
    global _ACTIVE_REQUESTS
    deadline = time.monotonic() + timeout
    with _REQUEST_CONDITION:
        while _ACTIVE_REQUESTS >= limit:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _REQUEST_CONDITION.wait(timeout=remaining)
        _ACTIVE_REQUESTS += 1
        return True


def _release_request_slot() -> None:
    global _ACTIVE_REQUESTS
    with _REQUEST_CONDITION:
        _ACTIVE_REQUESTS = max(0, _ACTIVE_REQUESTS - 1)
        _REQUEST_CONDITION.notify()


def _endpoint_reachable(base_url: str, timeout: float = 0.6) -> bool:
    parsed = urlparse(str(base_url or ""))
    if not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


async def _answer(
    question: str,
    session_id: str = "",
    conversation_history: list[dict[str, Any]] | None = None,
    conversation_id: str = "",
    client_id: str = "",
) -> dict[str, Any]:
    try:
        if conversation_history is not None and not (conversation_id or client_id):
            result = await run_in_threadpool(
                _new_pipeline().answer,
                question,
                None,
                "",
                conversation_history,
            )
            return result.to_dict()

        settings = get_settings()
        identity = conversation_id or session_id or new_conversation_id()
        owner = client_id or "client_api_anonymous"
        store = _conversation_store()
        store.ensure(
            identity,
            owner,
            legacy_records=session_records(settings.trace_directory, identity, 200),
        )
        history = store.history(
            identity,
            owner,
            settings.conversation_turns,
            query=question,
        )
        document_passages = _select_document_passages(
            store, identity, owner, question, history
        )
        result = await run_in_threadpool(
            _new_pipeline().answer,
            question,
            None,
            identity,
            history,
            document_passages,
        )
        payload = result.to_dict()
        payload["conversation_id"] = identity
        store.append_turn(identity, owner, question, payload)
    except SearchReaderError as exc:
        status = 422 if exc.stage == "input" else 502
        raise HTTPException(
            status_code=status,
            detail={"stage": exc.stage, "message": str(exc)},
        ) from exc
    except (KeyError, PermissionError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return payload


def _answer_events(
    question: str,
    session_id: str = "",
    conversation_id: str = "",
    client_id: str = "",
    cancel_event: threading.Event | None = None,
) -> Iterator[str]:
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    cancelled = cancel_event or threading.Event()

    def progress(stage: str, detail: dict[str, object]) -> None:
        if cancelled.is_set():
            raise SearchReaderError("cancelled", "request was cancelled")
        events.put({"type": "progress", "stage": stage, "detail": detail})

    def work() -> None:
        try:
            settings = get_settings()
            identity = conversation_id or session_id or new_conversation_id()
            owner = client_id or "client_api_anonymous"
            store = _conversation_store()
            store.ensure(
                identity,
                owner,
                legacy_records=session_records(settings.trace_directory, identity, 200),
            )
            history = store.history(
                identity,
                owner,
                settings.conversation_turns,
                query=question,
            )
            document_passages = _select_document_passages(
                store, identity, owner, question, history
            )
            result = _new_pipeline().answer(
                question,
                on_progress=progress,
                session_id=identity,
                conversation_history=history,
                document_passages=document_passages,
            )
            payload = result.to_dict()
            payload["conversation_id"] = identity
            if cancelled.is_set():
                return
            store.append_turn(identity, owner, question, payload)
            events.put({"type": "result", "result": payload})
        except SearchReaderError as exc:
            events.put(
                {
                    "type": "error",
                    "error": {"stage": exc.stage, "message": str(exc)},
                }
            )
        except Exception:  # noqa: BLE001 - preserve the stream response contract
            events.put(
                {
                    "type": "error",
                    "error": {"stage": "unexpected", "message": "服务暂时不可用"},
                }
            )

    threading.Thread(target=work, daemon=True).start()
    try:
        while True:
            try:
                event = events.get(timeout=1.0)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] in {"result", "error"}:
                break
    finally:
        cancelled.set()
        _release_request_slot()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "static" / "index.html")


@app.get("/healthz")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "retrieval_mode": "web-or-attached-documents",
        "tool_calling": False,
    }


@app.get("/readyz")
def readiness() -> dict[str, Any]:
    settings = get_settings()
    payload = settings.readiness()
    reader_reachable = _endpoint_reachable(settings.rwkv_base_url)
    final_reachable = (
        reader_reachable
        if settings.final_base_url == settings.rwkv_base_url
        else _endpoint_reachable(settings.final_base_url)
    )
    payload["reader_model_reachable"] = reader_reachable
    payload["final_model_reachable"] = final_reachable
    payload["ready"] = bool(payload["ready"] and reader_reachable and final_reachable)
    return payload


@app.post("/v1/answer")
async def answer(payload: AnswerRequest, request: Request) -> dict[str, Any]:
    _enforce_rate_limit(request)
    settings = get_settings()
    acquired = await run_in_threadpool(
        _acquire_request_slot,
        settings.api_max_concurrent_requests,
        settings.api_queue_timeout_seconds,
    )
    if not acquired:
        raise HTTPException(status_code=503, detail="当前请求较多，请稍后再试")
    try:
        return await _answer(
            payload.question,
            payload.session_id,
            conversation_id=payload.conversation_id,
            client_id=payload.client_id,
        )
    finally:
        _release_request_slot()


@app.post("/v1/answer/stream")
async def answer_stream(payload: AnswerRequest, request: Request) -> StreamingResponse:
    _enforce_rate_limit(request)
    settings = get_settings()
    acquired = await run_in_threadpool(
        _acquire_request_slot,
        settings.api_max_concurrent_requests,
        settings.api_queue_timeout_seconds,
    )
    if not acquired:
        raise HTTPException(status_code=503, detail="当前请求较多，请稍后再试")
    return StreamingResponse(
        _answer_events(
            payload.question,
            payload.session_id,
            payload.conversation_id,
            payload.client_id,
            threading.Event(),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1/history")
def history(
    request: Request,
    limit: int = Query(default=40, ge=1, le=200),
) -> dict[str, Any]:
    host = request.client.host if request.client else ""
    try:
        is_local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_local = host.casefold() == "localhost"
    if not is_local:
        raise HTTPException(status_code=403, detail="history is available locally only")
    settings = get_settings()
    return {
        "records": recent_traces(settings.trace_directory, limit),
        "storage": str(settings.trace_directory),
    }


@app.post("/v1/session/history")
def session_history(payload: SessionHistoryRequest, request: Request) -> dict[str, Any]:
    host = request.client.host if request.client else ""
    try:
        is_local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_local = host.casefold() == "localhost"
    if not is_local:
        raise HTTPException(status_code=403, detail="session history is local only")
    settings = get_settings()
    return {
        "session_id": payload.session_id,
        "records": session_records(
            settings.trace_directory, payload.session_id, payload.limit
        ),
    }


@app.post("/v1/conversations/create")
def create_conversation(request: ConversationAccessRequest) -> dict[str, Any]:
    settings = get_settings()
    try:
        return _conversation_store().ensure(
            request.conversation_id,
            request.client_id,
            legacy_records=session_records(
                settings.trace_directory, request.conversation_id, 200
            ),
        )
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/conversations/list")
def list_conversations(request: ConversationListRequest) -> dict[str, Any]:
    try:
        conversations = _conversation_store().list(request.client_id, request.limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"object": "list", "data": conversations}


@app.post("/v1/conversations/get")
def get_conversation(request: ConversationAccessRequest) -> dict[str, Any]:
    try:
        return _conversation_store().get(request.conversation_id, request.client_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/documents/upload")
def upload_document(request: DocumentUploadRequest) -> dict[str, Any]:
    settings = get_settings()
    store = _conversation_store()
    try:
        store.ensure(request.conversation_id, request.client_id)
        return store.add_document(
            request.conversation_id,
            request.client_id,
            request.filename,
            request.content,
            max_bytes=settings.document_max_bytes,
            chunk_chars=settings.document_chunk_chars,
            content_base64=request.content_base64,
            max_pdf_bytes=settings.document_pdf_max_bytes,
            max_pdf_pages=settings.document_pdf_max_pages,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _document_upload_events(request: DocumentUploadRequest) -> Iterator[str]:
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    started = time.monotonic()
    finished = threading.Event()
    current_stage = "upload_received"
    current_message = "文件已接收"

    def progress(stage: str, detail: dict[str, Any]) -> None:
        nonlocal current_stage, current_message
        current_stage = stage
        current_message = str(detail.get("message") or current_message)
        events.put(
            {
                "type": "progress",
                "stage": stage,
                "detail": {
                    **detail,
                    "elapsed_seconds": round(time.monotonic() - started),
                },
            }
        )

    def work() -> None:
        settings = get_settings()
        store = _conversation_store()
        try:
            store.ensure(request.conversation_id, request.client_id)
            document = store.add_document(
                request.conversation_id,
                request.client_id,
                request.filename,
                request.content,
                max_bytes=settings.document_max_bytes,
                chunk_chars=settings.document_chunk_chars,
                content_base64=request.content_base64,
                max_pdf_bytes=settings.document_pdf_max_bytes,
                max_pdf_pages=settings.document_pdf_max_pages,
                progress=progress,
            )
            progress(
                "document_ready",
                {
                    "message": f"已完成解析，共 {document['chunk_count']} 个分块",
                    "chunk_count": document["chunk_count"],
                },
            )
            events.put({"type": "result", "result": document})
        except (KeyError, PermissionError, RuntimeError, ValueError) as exc:
            events.put(
                {
                    "type": "error",
                    "error": {"message": str(exc) or "文档上传失败"},
                }
            )
        finally:
            finished.set()

    threading.Thread(target=work, daemon=True).start()
    yield json.dumps(
        {
            "type": "progress",
            "stage": "upload_received",
            "detail": {"message": "文件已接收", "elapsed_seconds": 0},
        },
        ensure_ascii=False,
    ) + "\n"
    while not (finished.is_set() and events.empty()):
        try:
            event = events.get(timeout=1.0)
        except queue.Empty:
            event = {
                "type": "progress",
                "stage": current_stage,
                "detail": {
                    "message": current_message,
                    "elapsed_seconds": round(time.monotonic() - started),
                },
            }
        yield json.dumps(event, ensure_ascii=False) + "\n"


@app.post("/v1/documents/upload/stream")
def upload_document_stream(request: DocumentUploadRequest) -> StreamingResponse:
    return StreamingResponse(
        _document_upload_events(request),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/documents/list")
def list_documents(request: ConversationAccessRequest) -> dict[str, Any]:
    try:
        documents = _conversation_store().list_documents(
            request.conversation_id, request.client_id
        )
    except (KeyError, PermissionError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"object": "list", "data": documents}


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest, request: Request
) -> dict[str, Any]:
    _enforce_rate_limit(request)
    if payload.stream:
        raise HTTPException(
            status_code=400, detail="streaming is not implemented in v0.1"
        )
    user_indexes = [
        index
        for index, message in enumerate(payload.messages)
        if message.role.casefold() == "user" and message.content.strip()
    ]
    question = payload.messages[user_indexes[-1]].content if user_indexes else ""
    if not question:
        raise HTTPException(
            status_code=422, detail="a non-empty user message is required"
        )
    history: list[dict[str, str]] = []
    for position, user_index in enumerate(user_indexes[:-1]):
        next_user_index = user_indexes[position + 1]
        answer = next(
            (
                message.content.strip()
                for message in payload.messages[user_index + 1 : next_user_index]
                if message.role.casefold() == "assistant" and message.content.strip()
            ),
            "",
        )
        if answer:
            history.append(
                {
                    "question": payload.messages[user_index].content.strip(),
                    "answer": answer,
                }
            )
    settings = get_settings()
    acquired = await run_in_threadpool(
        _acquire_request_slot,
        settings.api_max_concurrent_requests,
        settings.api_queue_timeout_seconds,
    )
    if not acquired:
        raise HTTPException(status_code=503, detail="当前请求较多，请稍后再试")
    try:
        result = await _answer(question, conversation_history=history)
    finally:
        _release_request_slot()
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model or get_settings().rwkv_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["answer"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "search_reader": {
            key: result[key]
            for key in (
                "request_id",
                "queries",
                "sources",
                "evidence",
                "retrieval_mode",
                "search_provider",
                "reader_mode",
                "degraded",
                "warnings",
                "timings_ms",
            )
        },
    }


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "rwkv_search_reader.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
