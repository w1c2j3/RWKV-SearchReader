from __future__ import annotations

import json
from dataclasses import replace

from rwkv_search_reader.config import Settings
from rwkv_search_reader.search import TavilyProvider, _reset_tavily_health_for_tests


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.content = json.dumps(payload).encode()
        self.text = self.content.decode()


def test_tavily_key_pool_fails_over_without_changing_query() -> None:
    _reset_tavily_health_for_tests()
    settings = replace(
        Settings(), tavily_api_key="", tavily_api_keys=("first-key", "second-key")
    )
    provider = TavilyProvider(settings)
    calls = []

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if kwargs["headers"]["Authorization"] == "Bearer first-key":
            return FakeResponse(432, {"error": "quota"})
        return FakeResponse(
            200,
            {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com/result",
                        "content": "Relevant snippet",
                        "score": 0.8,
                    }
                ]
            },
        )

    provider.session.post = post
    question = "完整用户问题"
    hits = provider.search(question, 6)

    assert len(calls) == 2
    assert calls[0]["json"]["query"] == question
    assert calls[1]["json"]["query"] == question
    assert calls[0]["headers"]["Authorization"] == "Bearer first-key"
    assert calls[1]["headers"]["Authorization"] == "Bearer second-key"
    assert hits[0].url == "https://example.com/result"

    calls.clear()
    provider.search(question, 6)
    assert len(calls) == 1
    assert calls[0]["headers"]["Authorization"] == "Bearer second-key"
    _reset_tavily_health_for_tests()
