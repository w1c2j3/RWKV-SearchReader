from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rwkv_search_reader.config import Settings
from rwkv_search_reader.models import SearchReaderError
from rwkv_search_reader.search import (
    TavilyProvider,
    _tavily_topic,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.content = json.dumps(payload).encode()
        self.text = self.content.decode()


def test_tavily_stops_after_quota_error_without_trying_another_key() -> None:
    settings = replace(
        Settings(),
        tavily_api_key="active-key",
        tavily_api_keys=("active-key", "next-key"),
    )
    provider = TavilyProvider(settings)
    calls = []

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse(432, {"error": "quota"})

    provider.session.post = post
    with pytest.raises(SearchReaderError, match="HTTP 432"):
        provider.search("user query", 5)

    assert len(calls) == 1
    assert calls[0]["headers"]["Authorization"] == "Bearer active-key"
    assert calls[0]["json"]["query"] == "user query"


def test_tavily_uses_news_topic_when_timestamp_is_material() -> None:
    assert _tavily_topic("宇树股价") == "general"
    assert _tavily_topic("今天有哪些 AI 新闻") == "news"
    assert _tavily_topic("概括本周最重要的科技动态") == "news"
    assert _tavily_topic("Python 列表怎么排序") == "general"
    settings = replace(
        Settings(), tavily_api_key="active-key", tavily_api_keys=("active-key",)
    )
    provider = TavilyProvider(settings)
    calls = []

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse(
            200,
            {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com/result",
                        "content": "Relevant snippet",
                        "raw_content": "# Result\n\nComplete **Markdown** material: 150.80.",
                        "score": 0.8,
                    }
                ]
            },
        )

    provider.session.post = post
    question = "完整用户问题"
    hits = provider.search(question, 6)

    assert len(calls) == 1
    assert calls[0]["json"]["query"] == question
    assert calls[0]["headers"]["Authorization"] == "Bearer active-key"
    assert hits[0].url == "https://example.com/result"
    assert hits[0].snippet == "Relevant snippet"
    assert hits[0].content == "# Result\n\nComplete **Markdown** material: 150.80."
    assert calls[0]["json"]["include_raw_content"] == "markdown"
    assert provider.results_are_material is True

    calls.clear()
    provider.search("今天有哪些 AI 新闻", 6)
    assert calls[0]["json"]["topic"] == "news"
    assert "time_range" not in calls[0]["json"]


def test_tavily_falls_back_to_ranked_snippet_without_raw_markdown() -> None:
    settings = replace(Settings(), tavily_api_key="key")
    provider = TavilyProvider(settings)

    def post(url, **kwargs):
        del url, kwargs
        return FakeResponse(
            200,
            {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com/result",
                        "content": "Relevant snippet",
                        "raw_content": None,
                    }
                ]
            },
        )

    provider.session.post = post
    hits = provider.search("question", 1)

    assert hits[0].content == "Relevant snippet"
    assert hits[0].snippet == "Relevant snippet"


def test_tavily_bounds_raw_markdown_to_the_reader_material_limit() -> None:
    settings = replace(
        Settings(), tavily_api_key="key", max_evidence_chars=20
    )
    provider = TavilyProvider(settings)

    def post(url, **kwargs):
        del url, kwargs
        return FakeResponse(
            200,
            {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com/result",
                        "content": "Relevant snippet",
                        "raw_content": "# Result\n\n" + "material" * 10,
                    }
                ]
            },
        )

    provider.session.post = post
    hits = provider.search("question", 1)

    assert len(hits[0].content) == 20
    assert hits[0].content.startswith("# Result\n\n")


def test_tavily_limits_search_to_explicit_domain() -> None:
    settings = replace(Settings(), tavily_api_key="key")
    provider = TavilyProvider(settings)
    captured = {}

    def post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse(200, {"results": []})

    provider.session.post = post
    provider.search("请依据 python.org 查询最新版本", 6)

    assert captured["json"]["include_domains"] == ["python.org"]


def test_tavily_does_not_treat_filenames_as_domains() -> None:
    settings = replace(Settings(), tavily_api_key="key")
    provider = TavilyProvider(settings)
    captured = []

    def post(url, **kwargs):
        captured.append(kwargs["json"])
        return FakeResponse(200, {"results": []})

    provider.session.post = post
    for question in (
        "Node.js 当前 LTS 版本",
        "llama.cpp 最新版本",
        "给出 compose.yaml",
        "不要修改 uv.lock",
    ):
        provider.search(question, 6)

    assert all("include_domains" not in request for request in captured)


def test_tavily_retries_without_a_rejected_inferred_domain() -> None:
    settings = replace(Settings(), tavily_api_key="key")
    provider = TavilyProvider(settings)
    captured = []

    def post(url, **kwargs):
        del url
        captured.append(kwargs["json"].copy())
        if len(captured) == 1:
            return FakeResponse(
                400,
                {"detail": {"error": "All domains in include_domains are invalid"}},
            )
        return FakeResponse(200, {"results": []})

    provider.session.post = post
    provider.search("请核对 release.zz", 6)

    assert captured[0]["include_domains"] == ["release.zz"]
    assert "include_domains" not in captured[1]
    assert captured[1]["query"] == captured[0]["query"]
