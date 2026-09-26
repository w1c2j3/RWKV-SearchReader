import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("api_client_example", Path(__file__).parents[1] / "examples/api_client_example.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def test_false_does_not_submit_job(monkeypatch):
    monkeypatch.setattr(client, "needs_search", lambda *args: False)
    def unexpected(*args):
        pytest.fail("false must not submit a search job")
    monkeypatch.setattr(client, "search_answer", unexpected)
    assert client.answer_if_search_needed([{"role": "user", "content": "你好"}]) is None


def test_true_returns_server_answer(monkeypatch):
    messages = [{"role": "user", "content": "查询最新版本"}]
    result = {"role": "assistant", "content": "回答", "results": []}
    monkeypatch.setattr(client, "needs_search", lambda *args: True)
    def search(actual, locale):
        assert actual == messages
        assert locale == "zh-CN"
        return result
    monkeypatch.setattr(client, "search_answer", search)
    assert client.answer_if_search_needed(messages) is result


def test_decision_error_is_not_false(monkeypatch):
    def fail(*args):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(client, "needs_search", fail)
    with pytest.raises(RuntimeError, match="model unavailable"):
        client.answer_if_search_needed([])
