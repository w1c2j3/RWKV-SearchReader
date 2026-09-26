import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from rwkv_search_reader import agent_demo
from rwkv_search_reader.config import Settings
from rwkv_search_reader.reader import RWKVReader


def test_observatory_runs_one_original_question_and_returns_result_unchanged(monkeypatch):
    received = []
    payload = {"answer": "原流程的回答 [S1]", "degraded": True, "warnings": ["recovered material"], "sources": []}
    settings = Settings(rwkv_model="test")

    class Pipeline:
        def __init__(self, current):
            assert current is settings
            self.reader = SimpleNamespace(call_observer=None)
            self.provider = SimpleNamespace(session=SimpleNamespace(close=lambda: None))

        def answer(self, question, on_progress):
            received.append(question)
            on_progress("searching", {"queries": [question]})
            self.reader.call_observer({"call_id": 1, "stage": "summarize", "state": "queued"})
            self.reader.call_observer({"call_id": 1, "stage": "summarize", "state": "returned", "output": "节点输出"})
            return SimpleNamespace(to_dict=lambda: payload)

    monkeypatch.setattr(agent_demo, "SearchReaderPipeline", Pipeline)
    released = threading.Event()
    events = [json.loads(line[6:]) for line in agent_demo.agent_events("原来的问题", settings, threading.Event(), released.set) if line.startswith("data: ")]
    assert received == ["原来的问题"]
    assert events[-1] == {"type": "result", "at_ms": events[-1]["at_ms"], "result": payload}
    assert [e["state"] for e in events if e["type"] == "call"] == ["queued", "returned"]
    assert released.wait(1)


def test_reader_observer_tracks_actual_parallel_calls_without_changing_prompts(monkeypatch):
    import re
    import requests
    barrier = threading.Barrier(8)
    posted = []

    def post(self, url, **kwargs):
        prompt = kwargs["json"]["prompt"]
        posted.append(prompt)
        marker = re.search(r"marker_\d+", prompt).group()
        barrier.wait(timeout=3)
        return SimpleNamespace(status_code=200, content=json.dumps({"choices": [{"text": marker}]}).encode())

    monkeypatch.setattr(requests.Session, "post", post)
    reader = RWKVReader(Settings(rwkv_model="test"))
    events = []
    reader.call_observer = events.append
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(lambda i: reader._complete(f"marker_{i}", max_tokens=32, stage="summarize"), range(8)))
    assert outputs == [f"marker_{i}" for i in range(8)]
    assert len(posted) == 8
    for i in range(8):
        assert f"User: marker_{i}\n\nAssistant: <think></think>" in posted
    assert len({e["call_id"] for e in events}) == 8
    for call_id in range(1, 9):
        assert [e["state"] for e in events if e["call_id"] == call_id] == ["queued", "running", "returned"]


def test_broken_observer_cannot_break_model_answer(monkeypatch):
    import requests

    def post(self, url, **kwargs):
        return SimpleNamespace(status_code=200, content=b'{"choices":[{"text":"answer"}]}')

    monkeypatch.setattr(requests.Session, "post", post)
    reader = RWKVReader(Settings(rwkv_model="test"))
    def broken(event):
        raise RuntimeError("UI observer failed")
    reader.call_observer = broken
    assert reader._complete("question", max_tokens=32, stage="summarize") == "answer"
    assert reader.call_records()[0]["status"] == "ok"
