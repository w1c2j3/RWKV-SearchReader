from __future__ import annotations

from dataclasses import replace

from rwkv_search_reader.config import Settings
from rwkv_search_reader.models import SearchHit
from rwkv_search_reader.retrieval import (
    Retriever,
    _balanced_hits,
    _search_queries,
    chunk_text,
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


def test_supplemental_retrieval_excludes_old_urls_and_continues_source_ids() -> None:
    settings = replace(Settings(), search_max_results=4, fetch_pages=4)
    provider = MaterialProvider(settings)

    _, sources, chunks, _ = Retriever(settings, provider).run_queries(
        ["missing detail"],
        question="original question",
        excluded_urls={"https://example.com/1", "https://example.com/2"},
        start_index=11,
    )

    assert [source.source_id for source in sources] == ["S11", "S12"]
    assert [chunk.source_id for chunk in chunks] == ["S11", "S12"]


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
