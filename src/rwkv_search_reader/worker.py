"""Single-server supervisor with bounded, killable job processes (Linux)."""
from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import signal
import threading
import time

from .jobs import serialize_result, serialize_sources, store
from .models import SearchReaderError
from .pipeline import SearchReaderPipeline


def execute(job, parent_pid):
    # Exit even if the supervisor is killed without running its cleanup handler.
    def watch_parent():
        while True:
            if os.getppid() != parent_pid or time.time() >= job["deadline"]:
                os._exit(1)
            time.sleep(0.5)
    threading.Thread(target=watch_parent, daemon=True).start()
    request = json.loads(job["request"])
    messages = request["messages"]
    history = [{"question": messages[i]["content"], "answer": messages[i+1]["content"]}
               for i in range(0, len(messages)-1, 2)]
    question = messages[-1]["content"]
    queue = store()
    previous_stage = None
    def progress(stage, detail):
        nonlocal previous_stage
        if stage in {"completed", "failed", "error"}:
            return
        if stage == "finalizing":
            public_stage, message = "generating", "正在生成回答"
        elif "search" in stage or "retriev" in stage:
            public_stage, message = "searching", "正在搜索相关资料"
        elif "read" in stage or "summar" in stage or "fetch" in stage:
            public_stage, message = "reading", "正在阅读资料"
        else:
            public_stage, message = "verifying", "正在整理与核对资料"
        if public_stage != previous_stage:
            queue.append_event(job["id"], "progress", {"stage": public_stage, "message": message})
            previous_stage = public_stage

    def delta(text):
        queue.append_event(job["id"], "delta", {"content": text})

    try:
        result = SearchReaderPipeline().answer(
            question, conversation_history=history, locale=request.get("locale", "zh-CN"),
            on_progress=progress, on_delta=delta,
            on_sources=lambda sources: queue.append_event(job["id"], "sources", {"results": [{k: v for k, v in source.items() if k != "content_markdown"} for source in serialize_sources(sources)]}),
        )
        store().finish(job["id"], {"http_status": 200, "result": serialize_result(result)})
    except Exception as exc:
        code = exc.stage if isinstance(exc, SearchReaderError) else "internal_error"
        store().finish(job["id"], {"http_status": 502, "error": {"code": code, "message": "任务执行失败，请稍后重试"}}, "failed")


def stop_process(process):
    if process.is_alive():
        process.terminate()
        process.join(3)
        if process.is_alive():
            process.kill()
    process.join()
    process.close()


def main():
    queue = store()
    concurrency = max(1, min(4, int(os.environ.get("SEARCH_READER_JOB_CONCURRENCY", "1"))))
    lock = open(str(queue.path) + ".worker.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A worker supervisor is already running for this database")
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    context = multiprocessing.get_context("spawn")
    active = {}
    try:
        while not stopped.is_set():
            for ident, (process, deadline) in list(active.items()):
                alive = process.is_alive()
                if not alive or time.time() >= deadline or not queue.heartbeat(ident):
                    stop_process(process)
                    queue.finish(ident, {"http_status": 502, "error": {"code": "worker_lost", "message": "任务执行进程中断"}}, "failed")
                    del active[ident]
            while len(active) < concurrency and not stopped.is_set():
                job = queue.claim()
                if not job:
                    break
                process = context.Process(target=execute, args=(job, os.getpid()))
                process.start()
                active[job["id"]] = (process, job["deadline"])
            stopped.wait(0.5)
    finally:
        for ident, (process, _) in active.items():
            stop_process(process)
            queue.finish(ident, {"http_status": 502, "error": {"code": "worker_shutdown", "message": "服务重启，请重新提交"}}, "failed")
        lock.close()


if __name__ == "__main__":
    main()
