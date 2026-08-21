from __future__ import annotations

import ipaddress
import json
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import PROJECT_ROOT, get_settings
from .models import SearchReaderError
from .pipeline import SearchReaderPipeline
from .tracing import recent_traces


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(default="", max_length=128)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False


app = FastAPI(
    title="RWKV SearchReader",
    version=__version__,
    description="Mandatory retrieve-then-read API. RWKV never receives tools.",
)


def _new_pipeline() -> SearchReaderPipeline:
    return SearchReaderPipeline(get_settings())


async def _answer(question: str, session_id: str = "") -> dict[str, Any]:
    try:
        result = await run_in_threadpool(
            _new_pipeline().answer, question, None, session_id
        )
    except SearchReaderError as exc:
        status = 422 if exc.stage == "input" else 502
        raise HTTPException(
            status_code=status,
            detail={"stage": exc.stage, "message": str(exc)},
        ) from exc
    return result.to_dict()


def _answer_events(question: str, session_id: str = "") -> Iterator[str]:
    events: queue.Queue[dict[str, Any]] = queue.Queue()

    def progress(stage: str, detail: dict[str, object]) -> None:
        events.put({"type": "progress", "stage": stage, "detail": detail})

    def work() -> None:
        try:
            result = _new_pipeline().answer(
                question, on_progress=progress, session_id=session_id
            )
            events.put({"type": "result", "result": result.to_dict()})
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
    while True:
        event = events.get()
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        if event["type"] in {"result", "error"}:
            break


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "static" / "index.html")


@app.get("/healthz")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "retrieval_mode": "forced",
        "tool_calling": False,
    }


@app.get("/readyz")
def readiness() -> dict[str, Any]:
    return get_settings().readiness()


@app.post("/v1/answer")
async def answer(request: AnswerRequest) -> dict[str, Any]:
    return await _answer(request.question, request.session_id)


@app.post("/v1/answer/stream")
async def answer_stream(request: AnswerRequest) -> StreamingResponse:
    return StreamingResponse(
        _answer_events(request.question, request.session_id),
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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(
            status_code=400, detail="streaming is not implemented in v0.1"
        )
    question = next(
        (
            message.content
            for message in reversed(request.messages)
            if message.role.casefold() == "user" and message.content.strip()
        ),
        "",
    )
    if not question:
        raise HTTPException(
            status_code=422, detail="a non-empty user message is required"
        )
    result = await _answer(question)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or get_settings().rwkv_model,
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
