"""A read-only event view over one ordinary SearchReader question."""
from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable, Iterator

from .config import Settings
from .pipeline import SearchReaderPipeline


def agent_events(
    task: str, settings: Settings, stopped: threading.Event,
    on_finished: Callable[[], None] | None = None,
) -> Iterator[str]:
    events: queue.Queue = queue.Queue()
    started = time.monotonic()

    def emit(kind: str, **data: object) -> None:
        if not stopped.is_set():
            events.put({"type": kind, "at_ms": round((time.monotonic() - started) * 1000), **data})

    def work() -> None:
        pipeline = None
        try:
            pipeline = SearchReaderPipeline(settings)
            pipeline.reader.call_observer = lambda event: emit("call", **event)
            emit("started", question=task)
            result = pipeline.answer(task, on_progress=lambda stage, detail: emit("progress", stage=stage, detail=detail))
            # Return the ordinary pipeline result verbatim: no extra model calls,
            # voting, independently generated questions, or publication policy.
            emit("result", result=result.to_dict())
        except Exception as exc:
            emit("error", stage=getattr(exc, "stage", "unexpected"),
                 message="问答执行失败，请查看检索或模型服务状态。")
        finally:
            if pipeline is not None:
                pipeline.provider.session.close()
            emit("end")
            if on_finished is not None:
                on_finished()

    def iterate() -> Iterator[str]:
        try:
            while not stopped.is_set():
                try:
                    event = events.get(timeout=.5)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if event["type"] == "end":
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            stopped.set()

    # Start immediately so a released/disconnected response cannot leak a slot.
    threading.Thread(target=work, daemon=True).start()
    return iterate()
