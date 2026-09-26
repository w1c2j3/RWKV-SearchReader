from __future__ import annotations

import io
from dataclasses import replace
from datetime import datetime, timedelta

import requests

from rwkv_search_reader.config import Settings
from rwkv_search_reader.models import SearchHit
from rwkv_search_reader.retrieval import (
    Retriever,
    _balanced_hits,
    _filter_expired_news_hits,
    _search_queries,
    chunk_text,
    fetch_page,
    provider_materials,
    rank_chunks,
)
from rwkv_search_reader.search import SearchProvider


class StubProvider(SearchProvider):
    name = "stub"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.received: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.received.append(query)
        text = "Python 3.13.0 was released on October 7, 2024."
        return [
            SearchHit(
                title="Python 3.13 release",
                url="https://example.com/python-313",
                snippet=text,
                content=text,
                provider=self.name,
            )
        ][:max_results]


class MaterialProvider(SearchProvider):
    name = "material-stub"
    results_are_material = True

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.received: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.received.append(query)
        return [
            SearchHit(
                title=f"Result {index}",
                url=f"https://example.com/{index}",
                content=f"Provider-organized material {index}",
                score=1.0 / index,
            )
            for index in range(1, max_results + 1)
        ]

    def extract(self, hits: list[SearchHit], query: str) -> dict[str, str]:
        raise AssertionError("reader-ready provider results must not be extracted again")


def test_fetch_page_uses_detected_encoding_when_header_defaults_to_latin1(monkeypatch) -> None:
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "text/html"
    response.encoding = "ISO-8859-1"
    response.raw = io.BytesIO((
        "<html><body><p>据交易所记录，样例公司上市首日开盘价为1100元，随后交易继续进行。</p></body></html>"
    ).encode("utf-8"))
    monkeypatch.setattr("rwkv_search_reader.retrieval._assert_public_destination", lambda _url: None)
    monkeypatch.setattr("rwkv_search_reader.retrieval.requests.get", lambda *_args, **_kwargs: response)

    hit = fetch_page(SearchHit(title="样例报道", url="https://example.com/report"), Settings())

    assert "上市首日开盘价为1100元" in hit.content
    assert hit.content_status == "fetched"


def test_retriever_uses_complete_user_question_as_single_query() -> None:
    settings = replace(Settings(), fetch_pages=1)
    provider = StubProvider(settings)
    retriever = Retriever(settings, provider)
    question = "Python 3.13.0 的正式发布日期是什么？"

    queries, sources, chunks, warnings = retriever.run(question)

    assert provider.received == [question]
    assert queries == [question]
    assert len(sources) == 1
    assert chunks
    assert warnings == []


def test_explicit_weekly_news_window_excludes_dated_old_results() -> None:
    today = datetime.now().astimezone().date()
    recent = SearchHit(
        title="近期报道", url="https://example.com/recent", content="正文",
        published_date=(today - timedelta(days=2)).isoformat(),
    )
    old = SearchHit(
        title="旧报道", url="https://example.com/old", content="正文",
        published_date=(today - timedelta(days=20)).isoformat(),
    )
    unknown = SearchHit(title="未标日期", url="https://example.com/unknown", content="正文")

    assert _filter_expired_news_hits(
        [recent, old, unknown], "最近一周有哪些科技动态？"
    ) == [recent, unknown]
    assert _filter_expired_news_hits([recent, old], "科技动态有哪些？") == [recent, old]


def test_retriever_splits_only_explicit_multi_question_utterance() -> None:
    settings = replace(Settings(), search_max_results=2, fetch_pages=2)
    provider = StubProvider(settings)
    retriever = Retriever(settings, provider)
    question = "原神当前主题是什么？绝区零联动时间是什么？"

    queries, _, _, warnings = retriever.run(question)

    assert queries == ["原神当前主题是什么？", "绝区零联动时间是什么？"]
    assert provider.received == queries
    assert warnings == []


def test_search_query_split_keeps_trailing_instruction_with_single_question() -> None:
    question = "Python 3.13.0 的发布日期是什么？请以官方页面为准。"

    assert _search_queries(question) == [question]


def test_retriever_reads_user_url_directly_without_search(monkeypatch) -> None:
    settings = replace(Settings(), max_evidence_chars=30)
    provider = StubProvider(settings)

    def fetch(hit, settings):
        del settings
        hit.title = "Direct page"
        hit.content = "direct material " * 10
        return hit

    monkeypatch.setattr("rwkv_search_reader.retrieval.fetch_page", fetch)
    queries, sources, chunks, warnings = Retriever(settings, provider).run_urls(
        ["https://example.com/release"], question="页面说了什么？"
    )

    assert provider.received == []
    assert queries == ["https://example.com/release"]
    assert sources[0].source_id == "S1"
    assert len(chunks[0].text) == 30
    assert chunks[0].retrieval_point == "https://example.com/release"
    assert chunks[0].retrieval_query == "页面说了什么？"
    assert warnings == []


def test_direct_url_reader_retries_a_transient_empty_fetch(monkeypatch) -> None:
    settings = Settings()
    provider = StubProvider(settings)
    calls = []

    def fetch(hit, settings):
        del settings
        calls.append(hit.url)
        if len(calls) == 1:
            hit.error = "ReadTimeout"
        else:
            hit.content = "material after retry"
        return hit

    monkeypatch.setattr("rwkv_search_reader.retrieval.fetch_page", fetch)
    _, sources, _, warnings = Retriever(settings, provider).run_urls(
        ["https://example.com/release"], question="页面说了什么？"
    )

    assert calls == ["https://example.com/release"] * 2
    assert sources[0].content == "material after retry"
    assert sources[0].error == ""
    assert warnings == []


def test_reader_ready_results_keep_provider_order_without_extracting() -> None:
    settings = replace(Settings(), search_max_results=4, fetch_pages=3)
    provider = MaterialProvider(settings)

    queries, sources, chunks, warnings = Retriever(settings, provider).run("question")

    assert queries == ["question"]
    assert len(sources) == 4
    assert [chunk.source_id for chunk in chunks] == ["S1", "S2", "S3"]
    assert [chunk.text for chunk in chunks] == [
        "Provider-organized material 1",
        "Provider-organized material 2",
        "Provider-organized material 3",
    ]
    assert warnings == []


def test_named_title_search_discards_high_scoring_unrelated_pages() -> None:
    settings = replace(Settings(), search_max_results=3, fetch_pages=3)

    class MixedProvider(MaterialProvider):
        def search(self, query: str, max_results: int) -> list[SearchHit]:
            del query, max_results
            return [
                SearchHit(
                    "Unrelated finance article",
                    "https://example.com/finance",
                    content="Financial market changes unrelated to the product.",
                    score=0.99,
                ),
                SearchHit(
                    "《示例产品》版本公告",
                    "https://example.com/release",
                    content="《示例产品》发布了新版本。",
                    score=0.62,
                ),
            ]

    _, sources, chunks, warnings = Retriever(
        settings, MixedProvider(settings)
    ).run_queries(
        ["示例产品 最新版本"], question="《示例产品》当前版本是什么？"
    )

    assert [source.title for source in sources] == ["《示例产品》版本公告"]
    assert [chunk.title for chunk in chunks] == ["《示例产品》版本公告"]
    assert any("discarded 1" in warning for warning in warnings)


def test_planned_subject_query_discards_unrelated_provider_pages() -> None:
    settings = replace(Settings(), search_max_results=5, fetch_pages=5)

    class MixedProvider(MaterialProvider):
        def search(self, query: str, max_results: int) -> list[SearchHit]:
            del query, max_results
            return [
                SearchHit("甲产品版本公告", "https://example.com/a", content="甲产品已更新"),
                SearchHit("无关产品版本公告", "https://example.com/b", content="完全不相关"),
            ]

    _, sources, _, warnings = Retriever(settings, MixedProvider(settings)).run_queries(
        ["甲产品 最新版本"],
        question="这些产品现在各自是什么版本？",
        subject_by_query={"甲产品 最新版本": "甲产品"},
    )

    assert [source.title for source in sources] == ["甲产品版本公告"]
    assert any("unrelated to 甲产品" in warning for warning in warnings)


def test_announcement_query_prefers_announcement_pages_over_named_guides() -> None:
    settings = replace(Settings(), search_max_results=3, fetch_pages=3)

    class MixedProvider(MaterialProvider):
        def search(self, query: str, max_results: int) -> list[SearchHit]:
            del query, max_results
            return [
                SearchHit(
                    "《示例产品》道具位置攻略",
                    "https://example.com/guide",
                    content="《示例产品》的道具位置。",
                    score=0.95,
                ),
                SearchHit(
                    "《示例产品》版本更新公告",
                    "https://example.com/announcement",
                    content="《示例产品》的版本更新公告。",
                    score=0.7,
                ),
            ]

    _, sources, _, warnings = Retriever(settings, MixedProvider(settings)).run_queries(
        ["示例产品 最新 官方公告"],
        question="《示例产品》当前版本有什么变化？",
    )

    assert [source.url for source in sources] == ["https://example.com/announcement"]
    assert any("without an announcement title" in warning for warning in warnings)


def test_planned_queries_keep_up_to_ten_materials_per_object() -> None:
    settings = replace(Settings(), search_max_results=10, fetch_pages=10)

    class PerObjectProvider(MaterialProvider):
        def search(self, query: str, max_results: int) -> list[SearchHit]:
            self.received.append(query)
            prefix = "a" if query == "对象A" else "b"
            return [
                SearchHit(
                    title=f"{query} {index}",
                    url=f"https://example.com/{prefix}/{index}",
                    content=f"{query} material {index}",
                )
                for index in range(1, max_results + 1)
            ]

    provider = PerObjectProvider(settings)
    _, sources, chunks, _ = Retriever(settings, provider).run_queries(
        ["对象A", "对象B"], question="比较对象A和对象B"
    )

    assert len(sources) == 20
    assert len(chunks) == 20
    assert sum(chunk.title.startswith("对象A") for chunk in chunks) == 10
    assert sum(chunk.title.startswith("对象B") for chunk in chunks) == 10
    assert {source.retrieval_point for source in sources} == {"对象A", "对象B"}
    assert {chunk.retrieval_point for chunk in chunks} == {"对象A", "对象B"}
    assert {chunk.retrieval_query for chunk in chunks} == {"对象A", "对象B"}
    assert all(
        chunk.retrieval_point in chunk.title
        for chunk in chunks
    )


def test_supplemental_retrieval_excludes_old_urls_and_continues_source_ids() -> None:
    settings = replace(Settings(), search_max_results=4, fetch_pages=4)
    provider = MaterialProvider(settings)

    _, sources, chunks, _ = Retriever(settings, provider).run_queries(
        ["missing detail"],
        question="original question",
        excluded_urls={"https://example.com/1", "https://example.com/2"},
        start_index=11,
        retrieval_point_by_query={"missing detail": "stable point"},
    )

    assert [source.source_id for source in sources] == ["S11", "S12"]
    assert [chunk.source_id for chunk in chunks] == ["S11", "S12"]
    assert all(source.retrieval_point == "stable point" for source in sources)
    assert all(chunk.retrieval_point == "stable point" for chunk in chunks)
    assert all(chunk.retrieval_query == "missing detail" for chunk in chunks)


def test_balanced_hits_deduplicates_http_and_https_page_variants() -> None:
    groups = [
        [SearchHit("HTTPS", "https://www.example.com/news/item/")],
        [SearchHit("HTTP", "http://example.com/news/item")],
    ]

    selected = _balanced_hits(groups, limit=10)

    assert [hit.title for hit in selected] == ["HTTPS"]


def test_provider_materials_do_not_reorder_by_local_score() -> None:
    hits = [
        SearchHit(
            title="First",
            url="https://example.com/1",
            source_id="S8",
            content="A",
            score=0.1,
            published_date="2026-08-20",
        ),
        SearchHit(title="Second", url="https://example.com/2", content="B", score=0.9),
    ]

    chunks = provider_materials(hits)

    assert [chunk.title for chunk in chunks] == ["First", "Second"]
    assert [chunk.source_id for chunk in chunks] == ["S8", "S2"]
    assert chunks[0].published_date == "2026-08-20"


def test_chunk_text_respects_size_and_overlap() -> None:
    chunks = chunk_text("A" * 1700, size=700, overlap=100)
    assert len(chunks) == 3
    assert all(len(item) <= 700 for item in chunks)
    assert chunks[0][-100:] == chunks[1][:100]


def test_fallback_chunk_ranking_uses_question_terms() -> None:
    settings = replace(Settings(), chunk_chars=800, chunk_overlap_chars=80)
    hits = [
        SearchHit(
            title="无关页面",
            url="https://example.com/1",
            content="天气和旅行信息。" * 20,
        ),
        SearchHit(
            title="RWKV 推理速度",
            url="https://example.com/2",
            content="RWKV 推理速度和显存占用测试。" * 20,
        ),
    ]

    chunks = rank_chunks("RWKV 推理速度怎么样", hits, settings)

    assert chunks[0].source_id == "S2"
