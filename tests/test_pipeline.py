from __future__ import annotations

from dataclasses import replace

import pytest

from rwkv_search_reader.config import Settings
from rwkv_search_reader.documents import MarkdownPassage
from rwkv_search_reader.models import (
    DocumentChunk,
    Evidence,
    EvidenceAudit,
    FieldEvidenceSelection,
    ResearchPlan,
    ResearchRecord,
    SearchHit,
    SearchReaderError,
)
from rwkv_search_reader.pipeline import (
    _EXPLICIT_WEB_RE,
    SearchReaderPipeline,
    _bind_queries_to_retrieval_points,
    _conversation_reply,
    _current_release_scope,
    _record_current_release_ids,
    _record_current_release_scopes,
    _require_online_scope_for_current_records,
    _newer_release_ambiguities,
    _flag_unresolved_newer_releases,
    _mask_unanchored_next_release_fields,
    _fallback_audit_query,
    _history_only_request,
    _rechecks_supported_fact,
    _sanitize_supplemental_query,
)
from rwkv_search_reader.tracing import append_trace, recent_traces


def test_pipeline_order_and_result_contract(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    calls = []
    hit = SearchHit(title="Source", url="https://example.com", snippet="Fact")
    chunk = DocumentChunk(
        source_id="S1", title=hit.title, url=hit.url, text="Fact", score=1
    )
    evidence = Evidence(
        source_id="S1", title=hit.title, url=hit.url, claim="Fact", quote="Fact"
    )

    def retrieve(question):
        calls.append("retrieve")
        return [question], [hit], [chunk], []

    def summarize(question, chunks, on_progress=None):
        calls.append("summarize")
        if on_progress:
            on_progress(
                "source_read",
                {"source_id": "S1", "completed": 1, "total": 1},
            )
        return [evidence], []

    def finalize(question, rows, on_progress=None, **kwargs):
        del kwargs
        calls.append("finalize")
        if on_progress:
            on_progress("finalizing", {"source_count": 1})
        return "Answer [S1]", []

    def audit(question, rows, queries):
        calls.append("audit")
        return EvidenceAudit(sufficient=True)

    pipeline.retriever.run = retrieve
    pipeline.reader.summarize = summarize
    pipeline.reader.audit = audit
    pipeline.reader.finalize = finalize

    progress = []
    result = pipeline.answer(
        "Question",
        on_progress=lambda stage, detail: progress.append((stage, detail)),
        session_id="session-test",
    )

    assert calls == ["retrieve", "summarize", "audit", "finalize"]
    assert result.retrieval_mode == "forced"
    assert result.reader_mode == "rwkv-reviewed-parallel-funnel-final-summary"
    assert result.session_id == "session-test"
    assert result.queries == ["Question"]
    assert result.answer == "Answer [S1]"
    assert result.warnings == []
    assert [stage for stage, _ in progress] == [
        "searching",
        "sources_found",
        "reading",
        "source_read",
        "auditing",
        "audit_completed",
        "finalizing",
        "completed",
    ]
    assert progress[1][1]["sources"][0] == {
        "source_id": "S1",
        "title": "Source",
            "url": "https://example.com",
            "score": 0.0,
            "published_date": "",
            "retrieval_point": "",
            "retrieval_query": "",
        }
    assert progress[0][1]["query"] == "Question"
    assert result.trace[-1]["stage"] == "completed"


def test_current_release_scope_uses_online_version_from_source_titles() -> None:
    sources = [
        SearchHit("《示例产品》1.2版本更新公告", "https://example.com/old", source_id="S1"),
        SearchHit("《示例产品》官网：1.3全新版本现已上线", "https://example.com/live", source_id="S2"),
        SearchHit("《示例产品》1.3版本更新公告|官方网站", "https://example.com/new", source_id="S3"),
        SearchHit("《示例产品》1.3版本攻略", "https://example.com/guide", source_id="S4"),
    ]

    assert _current_release_scope("《示例产品》现在的版本有哪些变化？", sources) == (
        (1, 3),
        {"S2", "S3"},
    )
    assert _current_release_scope("《示例产品》下一个版本是什么？", sources) == (
        None,
        set(),
    )


def test_current_release_summary_excludes_older_version_pages(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    hits = [
        SearchHit("《示例产品》1.2版本更新公告", "https://example.com/old", source_id="S1"),
        SearchHit("《示例产品》官网：1.3全新版本现已上线", "https://example.com/live", source_id="S2"),
        SearchHit("《示例产品》1.3版本更新公告|官方网站", "https://example.com/new", source_id="S3"),
        SearchHit("《示例产品》1.3版本攻略", "https://example.com/guide", source_id="S4"),
    ]
    chunks = [
        DocumentChunk(hit.source_id, hit.title, hit.url, hit.title)
        for hit in hits
    ]
    pipeline.retriever.run = lambda _question: (["当前版本"], hits, chunks, [])
    summarized = []

    def summarize(_question, selected_chunks, **_kwargs):
        summarized.append([chunk.source_id for chunk in selected_chunks])
        return (
            [
                Evidence(chunk.source_id, chunk.title, chunk.url, chunk.text, chunk.text)
                for chunk in selected_chunks
            ],
            [],
        )

    pipeline.reader.summarize = summarize
    pipeline.reader.audit = lambda *_args, **_kwargs: EvidenceAudit(sufficient=True)
    pipeline.reader.finalize = lambda *_args, **_kwargs: ("版本信息 [S2]", [])

    result = pipeline.answer("《示例产品》现在的版本有哪些变化？")

    assert summarized == [["S2", "S3"]]
    assert [item.source_id for item in result.evidence] == ["S2", "S3"]
    assert len(result.sources) == 4
    assert any(event["stage"] == "current_release_scoped" for event in result.trace)


def test_multi_record_current_release_ignores_preview_and_old_versions() -> None:
    question = "甲游戏和乙游戏现在各自是什么版本、主题？"
    records = [
        ResearchRecord("甲游戏", question, ["版本", "主题"]),
        ResearchRecord("乙游戏", question, ["版本", "主题"]),
    ]
    sources = [
        SearchHit("甲游戏官网：1.1版本上线", "https://a.example.com/main", source_id="S1",
                  retrieval_query="甲游戏 当前版本"),
        SearchHit("全新1.2版本上线", "https://a.example.com/news", source_id="S2",
                  retrieval_query="甲游戏 当前版本"),
        SearchHit("甲游戏1.2版本攻略", "https://guide.example.com/a", source_id="S3",
                  retrieval_query="甲游戏 当前版本"),
        SearchHit("乙游戏官网：3.4版本正式上线", "https://b.example.com/main", source_id="S4",
                  retrieval_query="乙游戏 当前版本"),
        SearchHit("乙游戏3.5版本即将上线", "https://b.example.com/preview", source_id="S5",
                  retrieval_query="乙游戏 当前版本"),
    ]

    scopes = _record_current_release_scopes(question, sources, records)

    assert scopes == {"甲游戏": ((1, 2), {"S2"}), "乙游戏": ((3, 4), {"S4"})}
    assert _record_current_release_ids(sources, records, scopes) == {"S2", "S4"}


def test_current_collection_does_not_promote_an_upcoming_release() -> None:
    question = "甲游戏和乙游戏现在各自是什么版本、主题？"
    records = [
        ResearchRecord("甲游戏", question, ["当前版本号", "版本主题"]),
        ResearchRecord("乙游戏", question, ["当前版本号", "版本主题"]),
    ]
    selections = [
        FieldEvidenceSelection("甲游戏", "当前版本号", "S1", "supported", "1.2版本", ["S1"]),
        FieldEvidenceSelection("乙游戏", "当前版本号", "S2", "supported", "2.0版本", ["S2"]),
        FieldEvidenceSelection("乙游戏", "版本主题", "S2", "supported", "远航", ["S2"]),
    ]

    guarded = _require_online_scope_for_current_records(
        question, records, {"甲游戏": ((1, 2), {"S1"})}, selections,
        representative_scope=True,
    )

    assert guarded[0].status == "supported"
    assert [(item.status, item.source_ids) for item in guarded[1:]] == [
        ("missing", []),
        ("missing", []),
    ]


def test_newer_release_on_publisher_host_marks_current_status_unresolved() -> None:
    records = [ResearchRecord("甲游戏", "现在是什么版本", ["当前版本号", "版本主题"])]
    sources = [
        SearchHit("甲游戏官网：1.2版本已上线", "https://a.example.com/main", source_id="S1"),
        SearchHit("甲游戏1.3版本活动汇总", "https://a.example.com/news", source_id="S2"),
        SearchHit("甲游戏1.4版本传闻", "https://other.example.com/rumor", source_id="S3"),
    ]
    scopes = {"甲游戏": ((1, 2), {"S1"})}
    ambiguities = _newer_release_ambiguities(sources, records, scopes)
    selections = [
        FieldEvidenceSelection("甲游戏", "当前版本号", "S1", "supported", "1.2版本", ["S1"]),
        FieldEvidenceSelection("甲游戏", "版本主题", "S1", "supported", "旧主题", ["S1"]),
    ]

    guarded = _flag_unresolved_newer_releases(selections, ambiguities)

    assert ambiguities == {"甲游戏": ["S2"]}
    assert [(item.status, item.source_ids) for item in guarded] == [
        ("conflict", ["S1", "S2"]),
        ("conflict", ["S1", "S2"]),
    ]


def test_pipeline_reads_an_explicit_url_before_search(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        research_plan_enabled=True,
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    url = "https://example.com/release"
    hit = SearchHit("Direct page", url, source_id="S1", content="Direct fact")
    chunk = DocumentChunk("S1", hit.title, hit.url, hit.content)
    evidence = Evidence("S1", hit.title, hit.url, "Direct fact", "Direct fact")
    calls = []
    pipeline.reader.plan_research = lambda question: ResearchPlan(
        decision="search",
        goal="读取指定网页",
        queries=[],
        direct_urls=[url],
    )

    def read_urls(urls, **kwargs):
        calls.append(("url", urls, kwargs))
        return urls, [hit], [chunk], []

    pipeline.retriever.run_urls = read_urls
    pipeline.retriever.run = lambda question: pytest.fail("search must not run")
    pipeline.reader.summarize = lambda *args, **kwargs: ([evidence], [])
    pipeline.reader.audit = lambda *args, **kwargs: EvidenceAudit(sufficient=True)
    pipeline.reader.finalize = lambda *args, **kwargs: ("Direct answer [S1]", [])

    result = pipeline.answer(f"请阅读 {url} 并总结")

    assert calls[0][0:2] == ("url", [url])
    assert result.retrieval_mode == "direct-url"
    assert result.search_provider == "direct-url"
    assert result.queries == [url]
    assert "url_reading" in [event["stage"] for event in result.trace]
    assert "searching" not in [event["stage"] for event in result.trace]


def test_model_planner_can_request_scope_clarification_without_search(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        research_plan_enabled=True,
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    pipeline.reader.plan_research = lambda question: ResearchPlan(
        decision="clarify",
        goal="确定缺失指代",
        clarification="你说的“这两个”具体指什么？",
    )
    pipeline.retriever.run = lambda question: pytest.fail("search must not run")
    pipeline.retriever.run_queries = lambda *args, **kwargs: pytest.fail(
        "search must not run"
    )
    progress = []

    result = pipeline.answer(
        "比较这两个哪个好？",
        on_progress=lambda stage, detail: progress.append((stage, detail)),
    )

    assert result.answer == "你说的“这两个”具体指什么？"
    assert result.retrieval_mode == "model-clarification"
    assert result.queries == []
    assert [stage for stage, _ in progress] == [
        "planning",
        "plan_completed",
        "clarifying",
        "completed",
    ]


def test_public_unspecified_set_discovers_objects_before_answer_retrieval(
    tmp_path,
) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        research_plan_enabled=True,
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    pipeline.reader.plan_research = lambda question: ResearchPlan(
        decision="search",
        scope_mode="discover",
        goal="按检索点收集资料",
        queries=["二游对象范围", "二游当前版本", "二游版本主题"],
    )
    first_hit = SearchHit(
        "二游版本盘点", "https://example.com/discovery", content="原神、绝区零"
    )
    scoped_hit = SearchHit(
        "原神版本公告", "https://example.com/genshin", content="原神当前版本公告"
    )
    retrieval_calls = []

    def retrieve(queries, **kwargs):
        retrieval_calls.append((queries, kwargs))
        hit = first_hit if len(retrieval_calls) == 1 else scoped_hit
        hit.source_id = f"S{len(retrieval_calls)}"
        return queries, [hit], [DocumentChunk(hit.source_id, hit.title, hit.url, hit.content)], []

    pipeline.retriever.run_queries = retrieve
    pipeline.reader.discover_research_scope = lambda question, sources, **_kwargs: ResearchPlan(
        decision="search",
        scope_mode="fixed",
        goal="逐对象核查",
        records=[ResearchRecord("原神", "当前版本和主题", ["版本", "主题"])],
        queries=["原神 当前版本 主题"],
        retrieval_points=["原神"],
    )
    pipeline.reader.summarize = lambda question, chunks, **kwargs: (
        [Evidence(chunk.source_id, chunk.title, chunk.url, chunk.text, chunk.text)
         for chunk in chunks],
        [],
    )
    pipeline.reader.audit = lambda *args, **kwargs: EvidenceAudit(sufficient=True)
    pipeline.reader.select_evidence = lambda *args, **kwargs: ([], [])
    final_call = {}

    def finalize(*args, **kwargs):
        final_call.update(kwargs)
        return "持续总结后的回答 [S1]", []

    pipeline.reader.finalize = finalize

    result = pipeline.answer("二游现在的版本是什么，各个主题是什么？")

    assert len(retrieval_calls) == 2
    expected_prefixes = [
        "二游对象范围",
        "二游当前版本",
        "二游版本主题",
    ]
    assert all(
        query.startswith(prefix) and "最新" in query
        for query, prefix in zip(retrieval_calls[0][0], expected_prefixes, strict=True)
    )
    assert result.queries[:3] == retrieval_calls[0][0]
    assert "原神" in result.queries[3]
    assert "最新" in result.queries[3]
    stages = [event["stage"] for event in result.trace]
    assert "scope_discovering" in stages
    assert "scope_discovered" in stages
    assert "scope_sources_found" in stages
    assert [record.subject for record in final_call["research_records"]] == ["原神"]
    assert final_call["evidence_selections"] == []
    assert [item.source_id for item in result.evidence] == ["S2"]


def test_unconfirmed_collection_members_stop_before_answer_synthesis(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        research_plan_enabled=True,
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    pipeline.reader.plan_research = lambda question: ResearchPlan(
        decision="search",
        scope_mode="discover",
        goal=question,
        queries=["类别成员"],
    )
    hit = SearchHit("类别报道", "https://example.com/category", source_id="S1", content="无具体成员")
    chunk = DocumentChunk("S1", hit.title, hit.url, hit.content)
    pipeline.retriever.run_queries = lambda queries, **_kwargs: (queries, [hit], [chunk], [])
    pipeline.reader.discover_research_scope = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(SearchReaderError("scope_discovery", "no concrete members"))
    )
    pipeline.reader.summarize = lambda *_args, **_kwargs: pytest.fail(
        "unconfirmed collection must not enter answer synthesis"
    )

    result = pipeline.answer("这一类对象各自的当前版本是什么？")

    assert "未能确认" in result.answer
    assert result.degraded
    assert result.evidence == []
    assert len(result.sources) == 1
    assert "scope_discovery_failed" in [event["stage"] for event in result.trace]


def test_collection_defers_field_model_calls_until_a_release_is_online(tmp_path) -> None:
    settings = replace(
        Settings(), rwkv_model="rwkv-test", research_plan_enabled=True,
        trace_enabled=False, trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    question = "甲游戏和乙游戏现在各自是什么版本？"
    pipeline.reader.plan_research = lambda _question: ResearchPlan(
        decision="search", scope_mode="discover", goal=question, queries=["类别成员"]
    )
    pipeline.reader.discover_research_scope = lambda *_args, **_kwargs: ResearchPlan(
        decision="search", scope_mode="fixed", goal=question,
        records=[
            ResearchRecord("甲游戏", question, ["当前版本号"]),
            ResearchRecord("乙游戏", question, ["当前版本号"]),
        ],
        queries=["甲游戏 当前版本", "乙游戏 当前版本"],
        retrieval_points=["甲游戏", "乙游戏"],
    )
    category = SearchHit("类别名单", "https://example.com/category", source_id="S1", content="甲游戏、乙游戏")
    live = SearchHit("甲游戏官网：1.2版本已上线", "https://a.example.com/main", source_id="S2",
                     content="甲游戏当前版本1.2", retrieval_query="甲游戏 当前版本")
    preview = SearchHit("乙游戏2.0版本即将上线", "https://b.example.com/preview", source_id="S3",
                        content="乙游戏即将更新", retrieval_query="乙游戏 当前版本")
    calls = []

    def retrieve(queries, **_kwargs):
        calls.append(queries)
        hits = [category] if len(calls) == 1 else [live, preview]
        return queries, hits, [DocumentChunk(hit.source_id, hit.title, hit.url, hit.content)
                               for hit in hits], []

    pipeline.retriever.run_queries = retrieve
    pipeline.reader.summarize = lambda _question, chunks, **_kwargs: (
        [Evidence(chunk.source_id, chunk.title, chunk.url, chunk.text, chunk.text,
                  retrieval_query=chunk.retrieval_query) for chunk in chunks], []
    )
    selected_records = []
    selected_sources = []

    def select(_question, records, rows, **_kwargs):
        selected_records.extend(record.subject for record in records)
        selected_sources.extend(item.source_id for item in rows)
        return [FieldEvidenceSelection("甲游戏", "当前版本号", "S2", "supported", "1.2版本", ["S2"])], []

    pipeline.reader.select_evidence = select
    pipeline.reader.audit = lambda *_args, **_kwargs: EvidenceAudit(sufficient=True)
    pipeline.reader.finalize = lambda *_args, **_kwargs: ("甲游戏 1.2版本 [S2]", [])

    result = pipeline.answer(question)

    assert selected_records == ["甲游戏"]
    assert selected_sources == ["S2"]
    assert [(item["subject"], item["status"]) for item in result.conversation_state["facts"]] == [
        ("甲游戏", "supported"), ("乙游戏", "missing")
    ]


def test_model_planner_queries_and_records_reach_audit_and_writer(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        research_plan_enabled=True,
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    records = [
        ResearchRecord("原神", "当前版本和福利", ["版本", "福利"]),
        ResearchRecord("绝区零", "当前版本和福利", ["版本", "福利"]),
    ]
    planned_queries = ["原神 当前版本 福利", "绝区零 当前版本 福利"]
    pipeline.reader.plan_research = lambda question: ResearchPlan(
        decision="search",
        goal="比较两个游戏",
        records=records,
        queries=planned_queries,
    )
    hit = SearchHit("Source", "https://example.com", content="Fact")
    chunk = DocumentChunk("S1", hit.title, hit.url, "Fact")
    received = {}

    def retrieve(queries, **kwargs):
        received["retrieval"] = (queries, kwargs)
        return queries, [hit], [chunk], []

    pipeline.retriever.run_queries = retrieve
    pipeline.retriever.run = lambda question: pytest.fail("planned search required")
    evidence = Evidence("S1", hit.title, hit.url, "Fact", "Fact")
    pipeline.reader.summarize = lambda *args, **kwargs: ([evidence], [])

    def audit(question, rows, queries, **kwargs):
        received["audit_records"] = kwargs["research_records"]
        return EvidenceAudit(sufficient=True)

    def finalize(question, rows, **kwargs):
        received["writer_records"] = kwargs["research_records"]
        received["evidence_sufficient"] = kwargs["evidence_sufficient"]
        return "比较结果 [S1]", []

    pipeline.reader.audit = audit
    pipeline.reader.select_evidence = lambda *args, **kwargs: (
        [
            FieldEvidenceSelection("原神", "版本", "S1", "supported", "Fact"),
            FieldEvidenceSelection("绝区零", "版本", "S1", "supported", "Fact"),
        ],
        [],
    )
    pipeline.reader.finalize = finalize

    result = pipeline.answer("原神和绝区零当前版本和福利分别是什么？")

    searched, options = received["retrieval"]
    assert all(
        query.startswith(prefix) and "最新" in query
        for query, prefix in zip(searched, planned_queries, strict=True)
    )
    assert options["question"] == "原神和绝区零当前版本和福利分别是什么？"
    assert options["start_index"] == 1
    assert options["retrieval_point_by_query"] == dict(
        zip(searched, planned_queries, strict=True)
    )
    assert received["audit_records"] == records
    assert received["writer_records"] == records
    assert received["evidence_sufficient"] is True
    assert len(received.get("writer_records", [])) == 2
    assert result.queries == searched
    assert result.conversation_state["records"][0]["subject"] == "原神"
    assert result.conversation_state["facts"][0]["source_ids"] == ["S1"]


def test_pipeline_persists_failed_runs_with_trace(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=True,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)

    def fail_retrieval(question):
        raise SearchReaderError("search", "provider unavailable")

    pipeline.retriever.run = fail_retrieval

    with pytest.raises(SearchReaderError):
        pipeline.answer("Question", session_id="failed-session")

    record = recent_traces(tmp_path, 1)[0]
    assert record["status"] == "error"
    assert record["session_id"] == "failed-session"
    assert record["error"]["stage"] == "search"
    assert record["trace"][-1]["stage"] == "failed"


def test_exact_greeting_skips_search_and_model_calls(tmp_path) -> None:
    settings = replace(
        Settings(),
        trace_enabled=True,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    pipeline.retriever.run = lambda question: pytest.fail("search must not run")
    pipeline.reader.summarize = lambda *args, **kwargs: pytest.fail(
        "reader must not run"
    )
    progress = []

    result = pipeline.answer(
        "你好！",
        on_progress=lambda stage, detail: progress.append((stage, detail)),
        session_id="greeting-session",
    )

    assert result.answer == "你好！有什么想了解的，直接问我就好。"
    assert result.retrieval_mode == "local-conversation"
    assert result.queries == []
    assert result.sources == []
    assert [stage for stage, _ in progress] == ["responding", "completed"]
    saved = recent_traces(tmp_path, 1)[0]
    assert saved["trace"][-1]["stage"] == "completed"


def test_greeting_with_real_question_does_not_match_small_talk_rule() -> None:
    assert _conversation_reply("你好，请帮我查询 Python 最新版本") is None


def test_history_only_request_is_conservative() -> None:
    assert _history_only_request("总结你现在知道的东西")
    assert _history_only_request("回顾一下我们刚才聊了什么")
    assert _history_only_request("整理前面对话的要点")
    assert not _history_only_request("总结本周科技新闻")
    assert not _history_only_request("前面提到的品牌分别是什么？")


def test_audit_stop_guard_rejects_answer_seeded_confirmation_search() -> None:
    evidence = [
        Evidence(
            source_id="S1",
            title="Source 1",
            url="https://example.com/1",
            claim="Python 3.14 于 2025 年 10 月 7 日发布。",
            quote="Material 1",
        ),
        Evidence(
            source_id="S2",
            title="Source 2",
            url="https://example.com/2",
            claim="正式发布日期是 2025-10-07。",
            quote="Material 2",
        ),
    ]

    assert _rechecks_supported_fact(
        "Python 3.14 正式发布日期 2025年10月7日 官方确认",
        "Python 3.14 正式发布于什么时候？",
        evidence,
    )
    assert _rechecks_supported_fact(
        "Python 3.14 正式发布日期 2025-10-07",
        "Python 3.14 正式发布于什么时候？",
        evidence,
    )
    assert not _rechecks_supported_fact(
        "Python 3.14 正式发布日期 官方确认",
        "请给出 Python 3.14 正式发布日期的官方来源",
        evidence,
    )
    assert not _rechecks_supported_fact(
        "Python 3.14 正式发布日期 2025年10月7日 官方确认",
        "Python 3.14 正式发布于什么时候？",
        evidence[:1],
    )
    assert not _rechecks_supported_fact(
        "Python 3.14 2025年10月7日的发行价格",
        "Python 3.14 什么时候发布，发行价格是多少？",
        evidence,
    )


def test_supplemental_query_removes_evidence_answer_seed_and_adds_current_month() -> (
    None
):
    sanitized = _sanitize_supplemental_query(
        "原神 5.0 版本 更新时间 福利",
        "原神当前版本和福利是什么？",
    )

    assert "5.0" not in sanitized
    assert "原神" in sanitized
    assert "最新" in sanitized
    assert "年 月" not in sanitized
    assert str(__import__("time").localtime().tm_year) in sanitized


def test_supplemental_query_preserves_version_named_by_user() -> None:
    sanitized = _sanitize_supplemental_query(
        "Python 3.14 正式发布日期",
        "Python 3.14 正式发布于什么时候？",
    )

    assert "3.14" in sanitized


def test_next_release_supplemental_query_uses_public_fields_and_current_month() -> None:
    question = "某产品下一个版本什么时候更新，主题是什么？"
    date_query = _sanitize_supplemental_query(
        "某产品 下一个版本更新时间 update_time 官方资料", question
    )
    theme_query = _sanitize_supplemental_query(
        "某产品 下一个版本主题 theme 官方资料", question
    )

    assert "update_time" not in date_query
    assert " theme " not in f" {theme_query} "
    assert "更新时间 更新时间" not in date_query
    assert "版本主题 版本主题" not in theme_query
    assert "最新" in date_query and "最新" in theme_query
    assert __import__("re").search(r"\d{4}年\d{1,2}月", date_query)


def test_next_release_state_drops_old_theme_without_future_date() -> None:
    selections = [
        FieldEvidenceSelection("样例产品", "update_time", "", "missing"),
        FieldEvidenceSelection("样例产品", "theme", "S1", "supported", "旧版本主题", ["S1"]),
    ]

    masked, changed = _mask_unanchored_next_release_fields(
        "样例产品下一个版本什么时候更新，主题是什么？", selections
    )

    assert changed
    assert masked[1].status == "missing"
    assert masked[1].source_ids == []


def test_current_release_supplemental_query_uses_public_field_names() -> None:
    query = _sanitize_supplemental_query(
        "样例产品 version_number version_name release_date 官方公告",
        "样例产品当前版本号、版本名称和上线日期是什么？",
    )

    assert "version_number" not in query
    assert "version_name" not in query
    assert "release_date" not in query
    assert "版本号" in query
    assert "版本名称" in query
    assert "发布日期" in query


def test_supplemental_queries_bind_back_to_their_original_search_points() -> None:
    points = [
        "米哈游 原神 当前版本 版本主题 卡池 抽数",
        "米哈游 崩坏：星穹铁道 当前版本 版本主题 卡池 抽数",
        "米哈游 绝区零 当前版本 版本主题 卡池 抽数",
    ]
    queries = [
        "原神 版本 主题 卡池 抽数 官方公告",
        "崩坏：星穹铁道 版本 主题 卡池 抽数 官方公告",
        "绝区零 版本 主题 卡池 抽数 官方公告",
        "鸣潮 当前版本",
    ]

    bindings = _bind_queries_to_retrieval_points(queries, points)

    assert [bindings[query] for query in queries[:3]] == points
    assert bindings[queries[3]] == queries[3]


def test_pipeline_summarizes_conversation_without_search(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    pipeline.retriever.run = lambda question: pytest.fail("search must not run")
    pipeline.reader.contextualize = lambda *args: pytest.fail(
        "contextualizer must not run"
    )
    received = []

    def answer_from_history(question, history):
        received.append((question, history))
        return "此前主要讨论了口粮茶，并确认你偏好回甘明显的茶。"

    pipeline.reader.answer_from_history = answer_from_history
    progress = []
    history = [
        {"question": "口粮茶有哪些？", "answer": "可以按茶类和品牌选择。"},
        {"question": "我喜欢回甘好的", "answer": "可以关注铁观音和生普。"},
    ]

    result = pipeline.answer(
        "总结你现在知道的东西",
        conversation_history=history,
        on_progress=lambda stage, detail: progress.append((stage, detail)),
    )

    assert received == [("总结你现在知道的东西", history)]
    assert result.queries == []
    assert result.sources == []
    assert result.retrieval_mode == "conversation-history"
    assert result.reader_mode == "rwkv-conversation-summary"
    assert [stage for stage, _ in progress] == ["history_reading", "completed"]


def test_pipeline_reads_attached_markdown_without_web_search(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    pipeline.retriever.run = lambda question: pytest.fail("web search must not run")
    passage = MarkdownPassage(
        "doc_pipeline123",
        "project.md",
        0,
        "部署方案",
        "项目使用 SQLite 保存对话，并通过临时隧道提供公网访问。",
    )

    def summarize(question, chunks, on_progress=None):
        assert chunks[0].url.startswith("document://")
        return [
            Evidence(
                source_id="S1",
                title=chunks[0].title,
                url=chunks[0].url,
                claim=chunks[0].text,
                quote=chunks[0].text,
            )
        ], []

    pipeline.reader.summarize = summarize
    pipeline.reader.audit = lambda *args: EvidenceAudit(sufficient=True)
    pipeline.reader.finalize = lambda *args, **kwargs: ("项目使用本地对话库 [S1]。", [])

    result = pipeline.answer("总结这份文档", document_passages=[passage])

    assert result.queries == []
    assert result.retrieval_mode == "document"
    assert result.search_provider == "local-markdown"
    assert result.sources[0].title == "project.md · 部署方案"
    assert result.sources[0].url.startswith("document://")


def test_document_question_that_mentions_web_policy_does_not_request_web() -> None:
    assert _EXPLICIT_WEB_RE.search("文档默认会联网吗？") is None
    assert _EXPLICIT_WEB_RE.search("请联网查一下最新变化") is not None


def test_pipeline_uses_audit_query_to_add_new_sources(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
        review_max_rounds=2,
    )
    pipeline = SearchReaderPipeline(settings)
    first_hit = SearchHit(
        source_id="S1", title="First", url="https://example.com/1", content="A"
    )
    second_hit = SearchHit(
        source_id="S2", title="Second", url="https://example.com/2", content="B"
    )
    first_chunk = DocumentChunk("S1", "First", first_hit.url, "A")
    second_chunk = DocumentChunk("S2", "Second", second_hit.url, "B")

    pipeline.retriever.run = lambda question: (
        [question],
        [first_hit],
        [first_chunk],
        [],
    )
    supplemental_calls = []

    def supplement(queries, **kwargs):
        supplemental_calls.append((queries, kwargs))
        return queries, [second_hit], [second_chunk], []

    pipeline.retriever.run_queries = supplement

    def summarize(question, chunks, on_progress=None):
        del question, on_progress
        return [
            Evidence(
                source_id=chunk.source_id,
                title=chunk.title,
                url=chunk.url,
                claim=chunk.text,
                quote=chunk.text,
            )
            for chunk in chunks
        ], []

    pipeline.reader.summarize = summarize
    audits = iter(
        [
            EvidenceAudit(sufficient=False, queries=["missing fact query"]),
            EvidenceAudit(sufficient=True),
        ]
    )
    pipeline.reader.audit = lambda *args: next(audits)
    pipeline.reader.finalize = lambda question, rows, on_progress=None, **kwargs: (
        f"Answer from {len(rows)} sources [S1][S2]",
        [],
    )

    result = pipeline.answer("Question")

    assert result.answer == "Answer from 2 sources [S1][S2]"
    assert [source.source_id for source in result.sources] == ["S1", "S2"]
    assert result.queries == ["Question", "missing fact query"]
    assert supplemental_calls[0][1]["excluded_urls"] == {first_hit.url}
    assert supplemental_calls[0][1]["start_index"] == 2
    assert supplemental_calls[0][1]["retrieval_point_by_query"] == {
        "missing fact query": "Question"
    }
    assert [event["stage"] for event in result.trace].count("auditing") == 2


def test_invalid_audit_without_fields_gets_one_bounded_fallback_search(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
        review_max_rounds=2,
        review_max_queries=1,
    )
    pipeline = SearchReaderPipeline(settings)
    first = SearchHit("First", "https://example.com/1", content="first fact")
    second = SearchHit("Second", "https://example.com/2", content="second fact")
    first_chunk = DocumentChunk("S1", first.title, first.url, first.content)
    second_chunk = DocumentChunk("S2", second.title, second.url, second.content)
    pipeline.retriever.run = lambda question: (
        [question], [first], [first_chunk], []
    )
    supplemental = []

    def run_supplement(queries, **kwargs):
        supplemental.append(queries)
        return queries, [second], [second_chunk], []

    pipeline.retriever.run_queries = run_supplement
    pipeline.reader.summarize = lambda _question, chunks, **_kwargs: (
        [Evidence(chunk.source_id, chunk.title, chunk.url, chunk.text, chunk.text)
         for chunk in chunks],
        [],
    )

    def invalid_audit(*_args, **_kwargs):
        raise SearchReaderError("audit", "invalid output")

    pipeline.reader.audit = invalid_audit
    pipeline.reader.finalize = lambda *_args, **_kwargs: ("Answer [S2]", [])

    result = pipeline.answer("Question")

    assert supplemental == [["Question 官方资料"]]
    assert result.queries == ["Question", "Question 官方资料"]
    assert [event["stage"] for event in result.trace].count("supplemental_searching") == 1


def test_fallback_audit_query_avoids_repeating_an_exact_search():
    assert _fallback_audit_query("Question", ["Question"]) == "Question 官方资料"


def test_duplicate_sanitized_query_does_not_claim_evidence_is_sufficient(tmp_path) -> None:
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
        review_max_rounds=1,
    )
    pipeline = SearchReaderPipeline(settings)
    question = "某产品下一个版本什么时候更新？"
    raw_query = "某产品 update_time 官方资料"
    already_tried = _sanitize_supplemental_query(raw_query, question)
    hit = SearchHit("公告", "https://example.com/release", content="未公布下次更新")
    chunk = DocumentChunk("S1", hit.title, hit.url, hit.content)
    pipeline.retriever.run = lambda _question: ([already_tried], [hit], [chunk], [])
    pipeline.reader.summarize = lambda _question, _chunks, **_kwargs: (
        [Evidence("S1", hit.title, hit.url, hit.content, hit.content)],
        [],
    )
    pipeline.reader.audit = lambda *_args, **_kwargs: EvidenceAudit(
        sufficient=False, queries=[raw_query]
    )
    pipeline.reader.finalize = lambda *_args, **_kwargs: ("下次更新时间尚未公布。", [])

    result = pipeline.answer(question)

    audit_events = [
        event for event in result.trace if event["stage"] == "audit_completed"
    ]
    assert len(audit_events) == 1
    assert audit_events[0]["detail"]["sufficient"] is False
    assert audit_events[0]["detail"]["queries"] == []
    assert any("without a new query" in warning for warning in result.warnings)
    assert [event for event in result.trace if event["stage"] == "auditing"][0][
        "detail"
    ]["tried_queries"] == [already_tried]


def test_pipeline_resolves_followup_from_same_session_before_search(tmp_path) -> None:
    append_trace(
        tmp_path,
        {
            "status": "ok",
            "session_id": "continuous-chat",
            "question": "RWKV 最新版本是什么？",
            "answer": "最新版本是 RWKV-7-G1。",
        },
    )
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    contextualized = []

    def contextualize(question, history):
        contextualized.append((question, history))
        return "RWKV-7-G1 是什么时候发布的？"

    pipeline.reader.contextualize = contextualize
    hit = SearchHit(source_id="S1", title="Release", url="https://example.com/1")
    chunk = DocumentChunk("S1", hit.title, hit.url, "release fact")

    def retrieve(question):
        assert question == "RWKV-7-G1 是什么时候发布的？"
        return [question], [hit], [chunk], []

    pipeline.retriever.run = retrieve
    pipeline.reader.summarize = lambda *args, **kwargs: (
        [Evidence("S1", hit.title, hit.url, "release fact", "release fact")],
        [],
    )
    pipeline.reader.audit = lambda *args: EvidenceAudit(sufficient=True)
    pipeline.reader.finalize = lambda question, rows, on_progress=None, **kwargs: (
        "发布日期回答 [S1]",
        [],
    )

    result = pipeline.answer("它是什么时候发布的？", session_id="continuous-chat")

    assert contextualized[0][0] == "它是什么时候发布的？"
    assert contextualized[0][1][0]["question"] == "RWKV 最新版本是什么？"
    assert result.question == "它是什么时候发布的？"
    assert result.queries == ["RWKV-7-G1 是什么时候发布的？"]
    assert [event["stage"] for event in result.trace][:2] == [
        "contextualizing",
        "context_resolved",
    ]


def test_pipeline_resolves_every_new_turn_but_preserves_an_independent_topic(
    tmp_path,
) -> None:
    append_trace(
        tmp_path,
        {
            "status": "ok",
            "session_id": "new-topic-chat",
            "question": "小米 SU7 Ultra 的价格是多少？",
            "answer": "52.99 万元。",
        },
    )
    settings = replace(
        Settings(),
        rwkv_model="rwkv-test",
        trace_enabled=False,
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    received = []

    def contextualize(question, history):
        received.append((question, history))
        return question

    pipeline.reader.contextualize = contextualize
    hit = SearchHit(source_id="S1", title="Care", url="https://example.com/care")
    chunk = DocumentChunk("S1", hit.title, hit.url, "洗车与售后信息")

    def retrieve(question):
        assert question == "Python 3.14 是什么时候发布的？"
        return [question], [hit], [chunk], []

    pipeline.retriever.run = retrieve
    pipeline.reader.summarize = lambda *args, **kwargs: (
        [Evidence("S1", hit.title, hit.url, "洗车与售后信息", "洗车与售后信息")],
        [],
    )
    pipeline.reader.audit = lambda *args: EvidenceAudit(sufficient=True)
    pipeline.reader.finalize = lambda *args, **kwargs: ("新问题回答 [S1]", [])

    result = pipeline.answer(
        "Python 3.14 是什么时候发布的？",
        session_id="new-topic-chat",
    )

    assert result.answer == "新问题回答 [S1]"
    assert received[0][0] == "Python 3.14 是什么时候发布的？"
    assert "contextualizing" in [event["stage"] for event in result.trace]
