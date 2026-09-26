from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response

from rwkv_search_reader import jobs, routing
from rwkv_search_reader.models import SearchReaderError


@pytest.fixture
def classifier(monkeypatch):
    from dataclasses import replace
    from rwkv_search_reader.config import Settings
    settings = replace(Settings(), router_base_url="http://router/v1", router_model="g1j-1.5b")
    monkeypatch.setattr(routing, "get_settings", lambda: settings)
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["model"] == "g1j-1.5b"
            self.session = SimpleNamespace(close=lambda: None)
        def completion(self, **kwargs):
            calls.append(kwargs)
            return state["output"]
    state = {"output": "false"}
    monkeypatch.setattr(routing, "OpenAICompatibleClient", Client)
    return calls, state


@pytest.mark.parametrize("question", ["你好", "hello!", "谢谢你", "明天的天气如何？", "后天出门需要带伞吗？"])
def test_every_question_uses_model(classifier, question):
    assert routing.decide_search([{"role": "user", "content": question}]) is False
    assert len(classifier[0]) == 1
    assert question in classifier[0][0]["user"]
    assert classifier[0][0]["temperature"] == 0.0


def test_boolean_and_history(classifier):
    # Model output, not question text, determines the returned boolean.
    classifier[1]["output"] = "true"
    messages = [{"role": "user", "content": "明天北京天气"}, {"role": "assistant", "content": "晴"}, {"role": "user", "content": "后天呢？"}]
    assert routing.decide_search(messages) is True
    assert "明天北京天气" in classifier[0][0]["user"]


def test_invalid_decision_is_error(classifier):
    classifier[1]["output"] = "false, because it is simple"
    with pytest.raises(SearchReaderError):
        routing.decide_search([{"role": "user", "content": "你好"}])


def test_missing_dedicated_model_does_not_fall_back(monkeypatch):
    from rwkv_search_reader.config import Settings
    monkeypatch.setattr(routing, "get_settings", Settings)
    with pytest.raises(SearchReaderError, match="Dedicated routing model"):
        routing.decide_search([{"role": "user", "content": "你好"}])


def test_endpoint_never_creates_job(monkeypatch):
    monkeypatch.setenv("SEARCH_READER_API_KEYS", "secret")
    monkeypatch.setattr(routing, "decide_search", lambda messages: True)
    def fail():
        raise AssertionError("Decision must not create a job or search")
    monkeypatch.setattr(jobs, "store", fail)
    body = jobs.JobRequest(messages=[{"role": "user", "content": "查询天气"}])
    assert jobs.search_decision(body, Response(), "Bearer secret") == {"needs_search": True}
    with pytest.raises(HTTPException) as exc:
        jobs.search_decision(body, Response(), "Bearer wrong")
    assert exc.value.status_code == 401


def test_classifier_failure_is_retryable_and_releases_slot(monkeypatch):
    monkeypatch.setenv("SEARCH_READER_API_KEYS", "secret")
    def fail(messages):
        raise SearchReaderError("routing", "timeout")
    monkeypatch.setattr(routing, "decide_search", fail)
    body = jobs.JobRequest(messages=[{"role": "user", "content": "问题"}])
    for _ in range(3):
        with pytest.raises(HTTPException) as exc:
            jobs.search_decision(body, Response(), "Bearer secret")
        assert exc.value.status_code == 502


def test_material_precedes_classification_requirements():
    prompt = routing.build_decision_prompt([{"role": "user", "content": "明天的天气如何？"}])
    assert prompt.index("明天的天气如何？") < prompt.index("判断要求：")
    assert prompt.endswith("不要解释。")
