from __future__ import annotations

import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import PROJECT_ROOT, get_settings
from .models import SearchReaderError
from .pipeline import SearchReaderPipeline


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)


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


async def _answer(question: str) -> dict[str, Any]:
    try:
        result = await run_in_threadpool(_new_pipeline().answer, question)
    except SearchReaderError as exc:
        status = 422 if exc.stage == "input" else 502
        raise HTTPException(
            status_code=status,
            detail={"stage": exc.stage, "message": str(exc)},
        ) from exc
    return result.to_dict()


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
    return await _answer(request.question)


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
