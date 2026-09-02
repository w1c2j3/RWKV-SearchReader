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


def test_public_unspecified_set_runs_planned_points_then_funnel_summary(
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
    retrieval_calls = []

    def retrieve(queries, **kwargs):
        retrieval_calls.append((queries, kwargs))
        first_hit.source_id = "S1"
        return (
            queries,
            [first_hit],
            [DocumentChunk("S1", first_hit.title, first_hit.url, first_hit.content)],
            [],
        )

    pipeline.retriever.run_queries = retrieve
    evidence = [Evidence("S1", first_hit.title, first_hit.url, "资料", "资料")]
    pipeline.reader.summarize = lambda question, chunks, **kwargs: (evidence, [])
    pipeline.reader.audit = lambda *args, **kwargs: EvidenceAudit(sufficient=True)
    pipeline.reader.select_evidence = lambda *args, **kwargs: pytest.fail(
        "unnamed collections must not use atomic field binding"
    )
    final_call = {}

    def finalize(*args, **kwargs):
        final_call.update(kwargs)
        return "持续总结后的回答 [S1]", []

    pipeline.reader.finalize = finalize

    result = pipeline.answer("二游现在的版本是什么，各个主题是什么？")

    assert len(retrieval_calls) == 1
    assert retrieval_calls[0][0] == [
        "二游对象范围",
        "二游当前版本",
        "二游版本主题",
    ]
    assert result.queries == [
        "二游对象范围",
        "二游当前版本",
        "二游版本主题",
    ]
    stages = [event["stage"] for event in result.trace]
    assert "search_points_collected" in stages
    assert "scope_discovering" not in stages
    assert final_call["research_records"] == []
    assert final_call["evidence_selections"] == []


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

    assert received["retrieval"] == (
        planned_queries,
        {
            "question": "原神和绝区零当前版本和福利分别是什么？",
            "start_index": 1,
            "retrieval_point_by_query": {
                "原神 当前版本 福利": "原神 当前版本 福利",
                "绝区零 当前版本 福利": "绝区零 当前版本 福利",
            },
        },
    )
    assert received["audit_records"] == records
    assert received["writer_records"] == records
    assert received["evidence_sufficient"] is True
    assert len(received.get("writer_records", [])) == 2
    assert result.queries == planned_queries
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
