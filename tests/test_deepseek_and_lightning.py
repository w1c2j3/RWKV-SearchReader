from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rwkv_search_reader.config import Settings, get_settings
from rwkv_search_reader.model_client import (
     OpenAICompatibleClient,
    _LIGHTNING_BATCH_URLS,
    _StopSequenceFilter,
    build_lightning_prompt,
)
from rwkv_search_reader.models import Evidence, ResearchRecord, SearchHit, SearchReaderError
from rwkv_search_reader.pipeline import _current_release_scope, _record_current_release_scopes
from rwkv_search_reader.reader import (
    RWKVReader,
    _copied_fact_supported,
    _normalized_full_dates,
    _parse_research_plan_output,
    _remove_unsupported_number_sentences,
    _source_relevance,
    _unsupported_summary_bindings,
    _yaml_example_errors,
)
from rwkv_search_reader.retrieval import Retriever, _balanced_hits, _focus_direct_url_content
from rwkv_search_reader.search import DeepSeekSearchProvider, _bound_deepseek_text


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | str, headers: dict | None = None):
        self.status_code = status_code
        if isinstance(payload, dict):
            self.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        else:
            self.content = payload.encode("utf-8")
        self.text = self.content.decode("utf-8")
        self.headers = headers or {"Content-Type": "application/json"}


class _FakeStreamResponse:
    def __init__(self, status_code: int, lines: list[bytes], content_type: str = "text/event-stream"):
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self._lines = lines
        self.text = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def iter_lines(self, **kwargs):
        del kwargs
        yield from self._lines


def test_deepseek_bounds_query_and_ori_question_to_128_chars() -> None:
    long_text = "Python 3.14 正式版发布日期、主要新特性、free-threaded 进展以及官方下载说明 " * 5
    bounded = _bound_deepseek_text(long_text, 128)
    assert 1 <= len(bounded) <= 128

    settings = replace(Settings(), search_provider="deepseek", deepseek_api_key="sk-test")
    provider = DeepSeekSearchProvider(settings)
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return _FakeResponse(
            200,
            {
                "results": [
                    {
                        "title": "Python Release",
                        "url": "https://www.python.org/downloads/",
                        "snippet": "Short snippet",
                        "summary": "Detailed summary with release date Oct. 25, 2026",
                        "content": "# Python Downloads\n\nPython 3.14 released Oct. 25, 2026.",
                        "rerank_score": 2.45,
                        "authority": 4,
                        "site_name": "Python.org",
                        "date": "2026-10-25T00:00:00+00:00",
                    }
                ]
            },
        )

    provider.session.post = post
    hits = provider.search_with_context(long_text, 5, ori_question=long_text)

    assert captured["url"] == "https://api.deepseek.com/v1/tools/search"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert len(captured["json"]["query"]) <= 128
    assert len(captured["json"]["ori_question"]) <= 128
    assert captured["json"]["need_content"] is True
    assert len(hits) == 1
    assert hits[0].authority == 4
    assert hits[0].site_name == "Python.org"
    assert hits[0].score == pytest.approx(2.45)
    assert hits[0].content_status == "fetched"


def test_deepseek_site_filter_retries_broadly_when_empty() -> None:
    settings = replace(Settings(), search_provider="deepseek", deepseek_api_key="sk-test")
    provider = DeepSeekSearchProvider(settings)
    queries: list[str] = []

    def post(url, **kwargs):
        del url
        queries.append(kwargs["json"]["query"])
        if len(queries) == 1:
            return _FakeResponse(200, {"results": []})
        return _FakeResponse(
            200,
            {
                "results": [
                    {
                        "title": "Mirror Doc",
                        "url": "https://docs.python.org/3/",
                        "snippet": "Docs",
                        "summary": "Python docs",
                        "content": "Python 3 documentation",
                        "rerank_score": 2.1,
                    }
                ]
            },
        )

    provider.session.post = post
    hits = provider.search("请依据 python.org 查询文档", 3)

    assert len(queries) == 2
    assert "site:python.org" in queries[0]
    assert "site:python.org" not in queries[1]
    assert len(hits) == 1


def test_deepseek_fetch_url_used_by_direct_url_retriever() -> None:
    settings = replace(Settings(), search_provider="deepseek", deepseek_api_key="sk-test")
    provider = DeepSeekSearchProvider(settings)
    calls: list[tuple[str, dict]] = []

    def post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        return _FakeResponse(
            200,
            {"title": "json — JSON encoder and decoder", "content": "# json\n\nSupports load and dump."},
        )

    provider.session.post = post
    urls, readable, chunks, warnings = Retriever(settings, provider).run_urls(
        ["https://docs.python.org/3.13/library/json.html"],
        question="json 模块支持读写吗？",
    )

    assert calls == [
        (
            "https://api.deepseek.com/v1/tools/fetch",
            {"url": "https://docs.python.org/3.13/library/json.html"},
        )
    ]
    assert urls == ["https://docs.python.org/3.13/library/json.html"]
    assert readable[0].title == "json — JSON encoder and decoder"
    assert "Supports load and dump" in chunks[0].text
    assert warnings == []


def test_balanced_hits_merges_same_page_fragment_sections() -> None:
    h1 = SearchHit(
        title="Python Versions",
        url="https://www.python.org/doc/versions/",
        content="Section A: Python 3.14 overview.",
        content_markdown="Section A: Python 3.14 overview.",
    )
    h2 = SearchHit(
        title="Python Versions #1",
        url="https://www.python.org/doc/versions/#1",
        content="Section B: Released on 7 October 2024.",
        content_markdown="Section B: Released on 7 October 2024.",
    )
    merged = _balanced_hits([[h1, h2]], limit=5, max_content_chars=5000)
    assert len(merged) == 1
    assert "Section A" in merged[0].content
    assert "Section B" in merged[0].content


def test_lightning_auto_detection_and_stop_stripping() -> None:
    base_url = "http://lightning-unit.test/v1"
    _LIGHTNING_BATCH_URLS.discard(base_url)
    client = OpenAICompatibleClient(
        base_url=base_url,
        api_key="secret",
        model="rwkv7-g1l-7.2b-20261031-ctx34304",
        timeout_seconds=10,
    )
    paths: list[tuple[str, dict]] = []

    def post(url, **kwargs):
        paths.append((url, kwargs["json"]))
        if url.endswith("/completions") and not url.endswith("/batch/completions"):
            return _FakeResponse(404, "Not Found")
        return _FakeResponse(
            200,
            {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "matched_stop": "<EOD>",
                        "message": {"role": "assistant", "content": '{"ok":true}<EOD>'},
                    }
                ]
            },
        )

    client.session.post = post
    out1 = client.completion(
        system="", user="只输出 JSON", max_tokens=32, temperature=0.0, stage="plan"
    )
    out2 = client.completion(
        system="", user="只输出 JSON", max_tokens=32, temperature=0.1, stage="plan"
    )

    assert out1 == '{"ok":true}'
    assert out2 == '{"ok":true}'
    assert paths[0][0].endswith("/v1/completions")
    assert paths[1][0].endswith("/v1/batch/completions")
    assert paths[1][1]["contents"] == [build_lightning_prompt("", "只输出 JSON")]
    assert paths[1][1]["top_k"] == 1
    assert paths[1][1]["temperature"] == 1.0
    # Second call goes directly to /batch/completions without another 404 probe.
    assert paths[2][0].endswith("/v1/batch/completions")
    assert paths[2][1]["top_p"] == 0.28
    _LIGHTNING_BATCH_URLS.discard(base_url)


def test_stop_sequence_filter_strips_split_eod_in_stream() -> None:
    filt = _StopSequenceFilter()
    chunks = ["Python 官方入口：", "https://docs.python.org [S1]<EO", "D>extra"]
    emitted = "".join(filt.feed(part) for part in chunks) + filt.finish("<EOD>")
    assert emitted == "Python 官方入口：https://docs.python.org [S1]"


def test_lightning_stream_strips_eod_before_emitting_deltas() -> None:
    client = OpenAICompatibleClient(
        base_url="http://lightning-stream.test/v1",
        api_key="secret",
        model="rwkv7-g1l",
        timeout_seconds=10,
    )
    stream_resp = _FakeStreamResponse(
        200,
        [
            b'data: {"choices":[{"delta":{"content":"Hello "},"finish_reason":null,"index":0}]}',
            b'data: {"choices":[{"delta":{"content":"world<EO"},"finish_reason":null,"index":0}]}',
            b'data: {"choices":[{"delta":{"content":"D>"},"finish_reason":null,"index":0}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0,"matched_stop":"<EOD>"}]}',
            b"data: [DONE]",
        ],
    )
    client.session.post = lambda *a, **kw: stream_resp
    deltas = list(
        client.stream_answer(
            user="hi", max_tokens=32, temperature=0.1, stage="finalize", protocol="lightning"
        )
    )
    assert "".join(deltas) == "Hello world"


def test_british_and_american_dates_reconcile_with_chinese_dates() -> None:
    material = "Python 3.13.0, documentation released on 7 October 2024. Patch on Oct. 25, 2026."
    assert _normalized_full_dates(material) == {"2024-10-07", "2026-10-25"}
    assert not _unsupported_summary_bindings(
        "Python 3.13.0 于 2024年10月7日发布，补丁日期为 2026-10-25。", material
    )
    assert _copied_fact_supported("2024年10月7日", material)


def test_multi_gpu_endpoints_round_robin_and_source_authority_boost(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_READER_SEARCH_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv(
        "SEARCH_READER_RWKV_BASE_URL",
        "http://127.0.0.1:18430/v1,http://127.0.0.1:18431/v1,http://127.0.0.1:18432/v1,http://127.0.0.1:18433/v1",
    )
    monkeypatch.setenv("SEARCH_READER_RWKV_MODEL", "rwkv7-g1l-7.2b-20261031-ctx34304")
    monkeypatch.setenv("SEARCH_READER_RWKV_PROTOCOL", "chat")  # normalized to pure continuation
    settings = get_settings()
    assert settings.rwkv_protocol == "completions"
    assert len(settings.rwkv_base_urls) == 4
    assert settings.readiness()["ready"] is True

    reader = RWKVReader(settings)
    picked = [reader._model_profile("summarize")[0] for _ in range(4)]
    assert set(picked) == set(settings.rwkv_base_urls)

    low_auth = Evidence("S1", "Python 教程", "https://blog.example/py", "claim", "Python 文档", score=2.2, authority=1)
    high_auth = Evidence("S2", "Python Docs", "https://docs.python.org/3/", "claim", "Python 文档", score=2.2, authority=4)
    assert _source_relevance("Python 文档", high_auth) > _source_relevance("Python 文档", low_auth)


def test_citation_tags_and_duration_equivalents_do_not_trigger_false_numeric_hallucination() -> None:
    material = "Default HTTP timeout is 1m30s across Python 3.14."
    answer = "Python 3.14 默认 HTTP 超时为 90 秒 [S8] [S18]。"
    assert not _unsupported_summary_bindings(answer, material)


def test_official_update_announcement_counts_as_online_release_scope() -> None:
    question = "《甲游戏》和《乙游戏》现在各自是什么版本？"
    records = [
        ResearchRecord("甲游戏", question, ["版本号"]),
        ResearchRecord("乙游戏", question, ["版本号"]),
    ]
    sources = [
        SearchHit("《甲游戏》官方网站-3.0版本「新章」更新公告", "https://a.example.com/30", source_id="S1"),
        SearchHit("《甲游戏》官方网站-3.1版本前瞻特别节目预告", "https://a.example.com/31", source_id="S2"),
        SearchHit("《乙游戏》官网-2.4版本现已上线", "https://b.example.com/24", source_id="S3"),
    ]
    scopes = _record_current_release_scopes(question, sources, records)
    assert scopes["甲游戏"] == ((3, 0), {"S1"})
    assert scopes["乙游戏"] == ((2, 4), {"S3"})
    assert _current_release_scope("《甲游戏》现在的版本是什么？", sources[:2]) == ((3, 0), {"S1"})


def test_single_subject_multi_query_caps_balanced_materials_to_fetch_pages() -> None:
    settings = replace(Settings(), search_provider="deepseek", deepseek_api_key="sk-test", search_max_results=8, fetch_pages=8)
    provider = DeepSeekSearchProvider(settings)

    def post(url, **kwargs):
        del url
        q = kwargs["json"]["query"]
        return _FakeResponse(
            200,
            {
                "results": [
                    {
                        "title": f"{q}-{i}",
                        "url": f"https://docs.python.org/{hash(q) & 0xfff}/{i}",
                        "snippet": "snippet",
                        "summary": "summary",
                        "content": f"content {q} {i}",
                        "rerank_score": 2.0,
                    }
                    for i in range(8)
                ]
            },
        )

    provider.session.post = post
    queries = ["Python 3.14 release date", "Python 3.14 whatsnew", "Python 3.14 download"]
    _, deduplicated, chunks, _ = Retriever(settings, provider).run_queries(
        queries,
        question="Python 3.14 发布日期与新特性",
        retrieval_point_by_query={q: "Python 3.14" for q in queries},
    )
    assert len(deduplicated) == 8
    assert len(chunks) == 8


def test_direct_url_plan_keeps_only_specified_urls_and_focuses_long_docs() -> None:
    q = "阅读 https://docs.python.org/3.13/library/json.html 和 https://docs.python.org/3.13/library/tomllib.html，对比读写与 load() 要求。"
    raw_plan = json.dumps(
        {
            "decision": "search",
            "scope_mode": "fixed",
            "goal": "对比 json 与 tomllib",
            "records": [{"subject": "json", "question": q, "fields": ["读写能力"]}],
            "queries": ["https://docs.python.org/3.13/library/json.html#ref"],
            "urls": [],
            "clarification": "",
        },
        ensure_ascii=False,
    )
    plan = _parse_research_plan_output(raw_plan, q, 4)
    assert plan.direct_urls == [
        "https://docs.python.org/3.13/library/json.html",
        "https://docs.python.org/3.13/library/tomllib.html",
    ]
    assert plan.queries == []
    assert plan.fallback is False

    long_doc = (
        "# json — JSON encoder and decoder\n\n"
        + (" Introductory prose about JSON syntax. " * 400)
        + "\n\njson.dump(obj, fp) writes JSON to a .write()-supporting file-like object.\n"
        + "json.load(fp) reads a text file or binary file containing a JSON document.\n"
        + (" Unrelated CLI options padding. " * 500)
    )
    focused = _focus_direct_url_content(long_doc, q, 6000)
    assert len(focused) <= 6000
    assert "json — JSON encoder" in focused
    assert "json.dump(obj, fp)" in focused
    assert "json.load(fp)" in focused


def test_remove_unsupported_number_sentences_preserves_yaml_indentation() -> None:
    material = "interval: 30s, timeout: 10s, retries: 3"
    answer = (
        "最小示例：\n"
        "```yaml\n"
        "services:\n"
        "  web:\n"
        "    image: nginx:alpine\n"
        "    healthcheck:\n"
        '      test: ["CMD", "curl", "-f", "http://localhost"]\n'
        "      interval: 30s\n"
        "      timeout: 10s\n"
        "      retries: 3\n"
        "```\n"
        "间隔为 30s。未证实默认值为 999。"
    )
    cleaned = _remove_unsupported_number_sentences(answer, material)
    assert "  web:\n    image: nginx:alpine" in cleaned
    assert "999" not in cleaned
    assert _yaml_example_errors(cleaned, "Docker Compose healthcheck 示例") == []
