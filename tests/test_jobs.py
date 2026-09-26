import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response
from pydantic import ValidationError

from rwkv_search_reader import jobs
from rwkv_search_reader.models import SearchHit


def request():
    return {"messages": [{"role": "user", "content": "问题"}], "locale": "zh-CN"}


def test_submission_owner_and_idempotency(tmp_path):
    store = jobs.JobStore(tmp_path / "jobs.db")
    first = store.submit("a", request(), "key")
    assert store.submit("a", request(), "key") == first
    with pytest.raises(HTTPException) as exc:
        store.submit("a", {**request(), "locale": "en"}, "key")
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc:
        store.get("b", first["job_id"])
    assert exc.value.status_code == 404
    assert jobs.JobStore(store.path).get("a", first["job_id"]) == first


def test_atomic_claim_and_completion(tmp_path):
    store = jobs.JobStore(tmp_path / "jobs.db")
    ident = store.submit("a", request())["job_id"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda _: store.claim(), range(4)))
    assert sum(item is not None for item in claims) == 1
    payload = {"http_status": 200, "result": {"role": "assistant", "content": "回答", "results": []}}
    store.finish(ident, payload)
    assert store.get("a", ident) == {"job_id": ident, "status": "done", **payload}
    store.finish(ident, {"error": "late"}, "failed")
    assert store.get("a", ident)["status"] == "done"


@pytest.mark.parametrize("expired", [True, False])
def test_timeout_and_worker_loss_are_terminal(tmp_path, expired):
    store = jobs.JobStore(tmp_path / "jobs.db")
    ident = store.submit("a", request())["job_id"]
    store.claim()
    with store.connection() as db:
        if expired:
            db.execute("UPDATE jobs SET deadline=?", (time.time()-1,))
        else:
            db.execute("UPDATE jobs SET heartbeat=?", (time.time()-61,))
    store.finish(ident, {"http_status": 200})
    result = store.get("a", ident)
    assert result["status"] == "failed"
    assert result["error"]["code"] == ("job_timeout" if expired else "worker_lost")


def test_protocol_and_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_READER_API_KEYS", "secret")
    monkeypatch.setenv("SEARCH_READER_JOB_DB", str(tmp_path / "jobs.db"))
    response = Response()
    result = jobs.submit_job(jobs.JobRequest(**request()), response, "Bearer secret", "idem")
    assert response.headers["Retry-After"] == "20"
    assert jobs.get_job(result["job_id"], Response(), "Bearer secret") == result
    with pytest.raises(HTTPException) as exc:
        jobs.owner_for("Bearer wrong")
    assert exc.value.status_code == 401
    for messages in ([], [{"role": "assistant", "content": "a"}], [{"role": "user", "content": " "}]):
        with pytest.raises(ValidationError):
            jobs.JobRequest(messages=messages)


def test_source_metadata_does_not_claim_snippet_is_body():
    hit = SearchHit(title="页面", url="https://example.com/#fragment", source_id="S1", content="fallback snippet", published_date="2024-10-27T03:33:20Z")
    result = jobs.serialize_result(SimpleNamespace(answer="回答 [S1]", sources=[hit]))
    source = result["results"][0]
    assert source["content_status"] == "snippet_only"
    assert source["content_markdown"] == ""
    assert source["published_at"] == 1730000000000
    assert source["canonical_url"] == "https://example.com/"
    assert source["language"] is None
    assert source["id"] == "S1"


def test_worker_passes_history_and_locale(monkeypatch):
    from rwkv_search_reader import worker
    captured = {}
    class Pipeline:
        def answer(self, question, conversation_history, **kwargs):
            captured.update(question=question, history=conversation_history, **kwargs)
            return SimpleNamespace(answer="answer", sources=[])
    class Store:
        def finish(self, ident, payload, state="done"):
            captured.update(payload=payload, state=state)
    monkeypatch.setattr(worker, "SearchReaderPipeline", Pipeline)
    monkeypatch.setattr(worker, "store", Store)
    monkeypatch.setattr(worker.threading.Thread, "start", lambda self: None)
    body = {"messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "previous"}, {"role": "user", "content": "next"}], "locale": "en-US"}
    worker.execute({"id": "x", "deadline": time.time()+900, "request": json.dumps(body)}, 0)
    assert captured["history"] == [{"question": "first", "answer": "previous"}]
    assert captured["question"] == "next"
    assert captured["locale"] == "en-US"
    assert captured["state"] == "done"


def test_hard_stop_terminates_child():
    import multiprocessing
    from rwkv_search_reader.worker import stop_process
    process = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(60,))
    process.start()
    started = time.monotonic()
    stop_process(process)
    assert time.monotonic() - started < 10


def test_http_contract(monkeypatch, tmp_path):
    import socket
    import threading
    import urllib.request
    import urllib.error
    import uvicorn
    from rwkv_search_reader.api import app
    monkeypatch.setenv("SEARCH_READER_API_KEYS", "http-secret")
    monkeypatch.setenv("SEARCH_READER_JOB_DB", str(tmp_path / "http.db"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        base = f"http://127.0.0.1:{port}/v1/jobs"
        headers = {"Authorization": "Bearer http-secret", "Content-Type": "application/json"}
        with urllib.request.urlopen(urllib.request.Request(base, data=json.dumps(request()).encode(), headers=headers), timeout=5) as response:
            assert response.status == 202
            created = json.load(response)
        ident = created["job_id"]
        jobs.store().claim()
        jobs.store().finish(ident, {"http_status": 200, "result": {"role": "assistant", "content": "回答", "results": []}})
        with urllib.request.urlopen(urllib.request.Request(base + "/" + ident, headers=headers), timeout=5) as response:
            result = json.load(response)
            assert result["status"] == "done"
            assert result["result"]["content"] == "回答"
            assert response.headers["Cache-Control"] == "no-store"
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/" + ident, timeout=5)
        assert exc.value.code == 401
    finally:
        server.should_exit = True
        thread.join(5)
        sock.close()
