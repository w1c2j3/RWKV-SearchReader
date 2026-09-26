from __future__ import annotations

import re
import threading
import time
from dataclasses import replace

import pytest

from rwkv_search_reader.config import Settings
from rwkv_search_reader.models import (
    DocumentChunk,
    Evidence,
    FieldEvidenceSelection,
    ResearchRecord,
    SearchHit,
    SearchReaderError,
)
from rwkv_search_reader.reader import (
    RWKVReader,
    _attach_missing_citations,
    _atomic_fallback_answer,
    _bind_atomic_citations,
    _bind_retrieval_point_citations,
    _bind_subject_citations,
    _chunk_material_nodes_by_retrieval_point,
    _clean_final_answer,
    _coalesce_adjacent_identical_citations,
    _contains_prompt_leakage,
    _context_retry_tokens,
    _copied_fact_supported,
    _document_answer_points,
    _eligible_evidence,
    _field_binding_applicable,
    _field_qualifier_supported,
    _field_source_snippets,
    _freshness_review_queries,
    _final_evidence_order,
    _final_validation_material,
    _is_explicit_benefit_fact,
    _latest_effective_material_timestamp,
    _MaterialNode,
    _merge_duplicate_atomic_fields,
    _missing_field_queries,
    _narrow_selected_fact,
    _parse_audit_output,
    _parse_research_plan_output,
    _parse_source_summary,
    _price_fact_bound_to_field,
    _preserve_news_window,
    _relevant_markdown_tables,
    _remove_copied_temporal_qualifiers,
    _requested_date_difference_material,
    _structured_field_candidates,
    _structured_price_candidates,
    _unsupported_summary_bindings,
    _unsupported_summary_numbers,
    _valid_discovered_subject,
    _valid_field_value_shape,
    _verified_source_relation,
    _without_foreign_points,
    _yaml_example_errors,
    citation_warnings,
)


def _evidence(count: int = 1) -> list[Evidence]:
    return [
        Evidence(
            source_id=f"S{index}",
            title=f"Source {index}",
            url=f"https://example.com/{index}",
            claim=f"Fact {index}",
            quote=f"Material {index}",
        )
        for index in range(1, count + 1)
    ]


def test_citation_audit_accepts_known_sources() -> None:
    assert citation_warnings("Answer [S1][S2]", _evidence(2)) == []


def test_source_ranking_uses_original_page_text_instead_of_model_claim() -> None:
    from rwkv_search_reader.reader import _source_relevance

    unrelated = Evidence(
        "S1",
        "道具位置攻略",
        "https://example.com/guide",
        "版本更新公告有核心玩法调整。",
        "这篇攻略只列出道具位置。",
        score=0.95,
    )
    announcement = Evidence(
        "S2",
        "版本更新公告",
        "https://example.com/news",
        "版本更新公告有核心玩法调整。",
        "版本更新公告列出了核心玩法调整。",
        score=0.4,
    )

    assert _source_relevance("版本更新 核心玩法调整", announcement) > (
        _source_relevance("版本更新 核心玩法调整", unrelated)
    )


def test_relevant_document_table_survives_a_lossy_model_summary() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: (
        "L2 的两次失败必须发生在同一演练窗口。"
    )
    material = """| 级别 | 触发条件 | 首次响应时限 | 升级对象 |
| --- | --- | --- | --- |
| L1 | P95 延迟高于 120 ms | 20 分钟 | 当值工程师 |
| L2 | 重复率高于 0.20% | 8 分钟 | 现场负责人 |"""

    evidence = reader._summarize_one(
        "如果重复率升到 0.21%，应触发哪个报警级别，多久响应，升级给谁？",
        DocumentChunk(
            "S1",
            "运行手册 / 报警分级",
            "document://manual#chunk-2",
            material,
        ),
    )

    assert _relevant_markdown_tables("重复率报警", material) in evidence.claim
    assert "| L2 | 重复率高于 0.20% | 8 分钟 | 现场负责人 |" in evidence.claim


def test_citation_audit_reports_missing_and_unknown_sources() -> None:
    assert "no [S1]-style" in citation_warnings("Answer", _evidence())[0]
    assert "S9" in citation_warnings("Answer [S9]", _evidence())[0]


def test_research_plan_parser_keeps_records_and_queries() -> None:
    plan = _parse_research_plan_output(
        """```json
        {"decision":"search","goal":"比较两个游戏",
         "records":[
           {"subject":"原神","question":"当前版本和福利","fields":["版本","福利"]},
           {"subject":"绝区零","question":"当前版本和福利","fields":["版本","福利"]}],
         "queries":["原神 2026年8月 当前版本 福利", "绝区零 2026年8月 当前版本 福利"],
         "clarification":""}
        ```""",
        "fallback",
        4,
    )

    assert plan.decision == "search"
    assert [record.subject for record in plan.records] == ["原神", "绝区零"]
    assert plan.records[0].fields == ["版本", "福利"]
    assert plan.queries == [
        "原神 2026年8月 当前版本 福利",
        "绝区零 2026年8月 当前版本 福利",
    ]


def test_research_plan_parser_supports_material_scope_discovery() -> None:
    plan = _parse_research_plan_output(
        '{"decision":"search","scope_mode":"discover","goal":"查询公开集合",'
        '"records":[],"queries":["对象范围","当前版本","版本主题","当前卡池"],'
        '"clarification":""}',
        "各个二游版本和主题是什么？",
        4,
    )

    assert plan.decision == "search"
    assert plan.scope_mode == "discover"
    assert plan.records == []
    assert plan.queries == ["对象范围", "当前版本", "版本主题", "当前卡池"]


def test_single_quoted_title_is_the_atomic_record_subject() -> None:
    plan = _parse_research_plan_output(
        '{"decision":"search","scope_mode":"fixed","records":['
        '{"subject":"游戏更新内容","question":"当前版本变化",'
        '"fields":["版本号","更新日期"]}],"queries":[]}',
        "《样例产品》当前版本有哪些变化？",
        4,
    )

    assert [record.subject for record in plan.records] == ["《样例产品》"]


def test_multiple_fields_for_one_quoted_title_share_its_subject() -> None:
    plan = _parse_research_plan_output(
        '{"decision":"search","scope_mode":"fixed","records":['
        '{"subject":"样例产品下次更新时间","fields":["update_time"]},'
        '{"subject":"样例产品下次主题","fields":["theme"]}],"queries":[]}',
        "关于《样例产品》，下一个版本什么时候更新，主题是什么？",
        4,
    )

    assert [record.subject for record in plan.records] == [
        "《样例产品》",
        "《样例产品》",
    ]


def test_collapsed_collection_gets_a_separate_rwkv_search_design_call() -> None:
    reader = RWKVReader(replace(Settings(), research_plan_max_queries=4))
    calls = []

    def complete(user, **kwargs):
        calls.append((user, kwargs))
        if kwargs["stage"] == "plan":
            return (
                '{"decision":"search","scope_mode":"fixed","goal":"逐项回答",'
                '"records":[{"subject":"米哈游游戏","question":"版本主题卡池抽数",'
                '"fields":["版本","主题","卡池","抽数"]}],'
                '"queries":["米哈游游戏 版本主题卡池抽数"],"clarification":""}'
            )
        return (
            '{"points":['
            '["米哈游游戏范围","米哈游当前运营游戏列表"],'
            '["当前版本主题","米哈游各游戏当前版本主题"],'
            '["当前卡池","米哈游各游戏当前卡池"],'
            '["可获得抽数","米哈游各游戏版本可获得抽数"]]}'
        )

    reader._complete = complete
    plan = reader.plan_research(
        "米哈游的游戏当前都分别是什么版本，主题、卡池和抽数是什么？"
    )

    assert [kwargs["stage"] for _, kwargs in calls] == ["plan", "search_design"]
    assert plan.scope_mode == "discover"
    assert plan.records == []
    assert len(plan.queries) == 1
    assert "运营游戏列表" in plan.queries[0]
    assert plan.discovery_fields == ["版本", "主题", "卡池", "抽数"]
    assert plan.retrieval_points == ["候选对象范围"]


def test_search_design_accepts_a_bare_list_of_point_query_pairs() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *_args, **_kwargs: (
        '>[["游戏范围","二次元游戏代表作品列表"],'
        '["版本公告","二次元游戏最新版本公告"]]'
    )

    queries, points = reader.design_search_points(
        "主要二次元游戏各自是什么版本？"
    )

    assert points == ["游戏范围", "版本公告"]
    assert all("二次元游戏" in query for query in queries)


def test_each_game_collection_is_discovered_even_if_planner_mistakes_date_for_subject():
    reader = RWKVReader(replace(Settings(), research_plan_max_queries=4))
    calls = []

    def complete(user, **kwargs):
        calls.append(kwargs["stage"])
        if kwargs["stage"] == "plan":
            return (
                '{"decision":"search","scope_mode":"fixed","records":['
                '{"subject":"当前日期：2026-09-25",'
                '"question":"现在主要二次元游戏各自是什么版本？每个版本的主题是什么？",'
                '"fields":["游戏名称","版本号","版本主题"]}],"urls":[]}'
            )
        return (
            '{"points":[["二次元游戏版本","二次元游戏当前版本和版本主题"]]}'
        )

    reader._complete = complete
    plan = reader.plan_research(
        "现在主要二次元游戏各自是什么版本？每个版本的主题是什么？"
    )

    assert plan.scope_mode == "discover"
    assert plan.records == []
    assert calls == ["plan", "search_design"]
    assert all("2026-09-25" not in query for query in plan.queries)
    assert all("二次元游戏" in query for query in plan.queries)


def test_model_expanded_collection_keeps_queries_but_uses_funnel_path() -> None:
    reader = RWKVReader(replace(Settings(), research_plan_max_queries=4))
    calls = []

    def complete(user, **kwargs):
        calls.append(kwargs["stage"])
        return (
            '{"decision":"search","scope_mode":"fixed","goal":"逐项回答",'
            '"records":['
            '{"subject":"原神","question":"版本主题","fields":["版本","主题"]},'
            '{"subject":"绝区零","question":"版本主题","fields":["版本","主题"]}],'
            '"queries":["原神 当前版本主题","绝区零 当前版本主题"],'
            '"clarification":""}'
        )

    reader._complete = complete
    plan = reader.plan_research("米哈游的游戏当前都分别是什么版本和主题？")

    assert calls == ["plan"]
    assert plan.scope_mode == "discover"
    assert plan.records == []
    assert plan.retrieval_points == ["原神", "绝区零"]
    assert len(plan.queries) == 2
    assert plan.queries[0].startswith("原神 当前版本主题")


def test_game_scope_rejects_events_and_field_values_as_subjects() -> None:
    question = "二游现在的版本是什么，各个主题是什么？"

    assert _valid_discovered_subject(question, "绝区零")
    assert not _valid_discovered_subject(question, "2026动漫游戏节")
    assert not _valid_discovered_subject(question, "夏日版本")
    assert not _valid_discovered_subject(question, "周年庆活动")
    assert not _valid_discovered_subject(question, "新产品")
    assert not _valid_discovered_subject(question, "头部产品")
    assert not _valid_discovered_subject(question, "老牌垂直二游厂商")


def test_research_plan_parser_requires_a_real_clarification_question() -> None:
    plan = _parse_research_plan_output(
        '{"decision":"clarify","goal":"逐个回答","records":[],"queries":[],"clarification":"你说的各个二游具体包括哪些？"}',
        "各个二游版本是什么？",
        4,
    )

    assert plan.decision == "clarify"
    assert plan.clarification == "你说的各个二游具体包括哪些？"


def test_planner_cannot_refuse_a_public_set_for_lack_of_named_objects() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: (
        '{"decision":"clarify","goal":"确定对象","records":[],"queries":[],'
        '"clarification":"请先列出具体游戏。"}'
    )

    plan = reader.plan_research("各个二游现在的版本是什么？")

    assert plan.decision == "search"
    assert plan.scope_mode == "discover"
    assert plan.queries[0].startswith("各个二游现在的版本是什么？")


def test_research_plan_parser_drops_prompt_placeholders() -> None:
    plan = _parse_research_plan_output(
        '{"decision":"search","scope_mode":"fixed","goal":"查版本",'
        '"records":[{"subject":"明确对象","question":"该记录需要回答的问题",'
        '"fields":["字段"]}],"queries":["原神 当前版本"],"clarification":""}',
        "原神当前版本是什么？",
        4,
    )

    assert plan.records == []
    assert plan.queries == ["原神 当前版本"]


def test_research_plan_prompt_puts_question_before_requirements() -> None:
    reader = RWKVReader(replace(Settings(), research_plan_max_queries=4))
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        captured.update(kwargs)
        return '{"decision":"search","goal":"查日期","records":[],"queries":["Python 3.14 发布日期"],"clarification":""}'

    reader._complete = complete
    plan = reader.plan_research("UNIQUE PLAN QUESTION")

    assert captured["user"].index("UNIQUE PLAN QUESTION") < captured["user"].index(
        "列出待检索的对象和字段"
    )
    assert "判定示例" not in captured["user"]
    assert '"subject":"明确对象"' not in captured["user"]
    assert captured["stage"] == "plan"
    assert plan.queries == ["Python 3.14 发布日期"]


def test_planner_designs_queries_in_a_separate_second_call() -> None:
    reader = RWKVReader(replace(Settings(), research_plan_max_queries=4))
    calls: list[tuple[str, str]] = []

    def complete(user, **kwargs):
        calls.append((kwargs["stage"], user))
        if kwargs["stage"] == "plan":
            return (
                '{"decision":"search","scope_mode":"fixed","goal":"查 LTS",'
                '"records":[{"subject":"Node.js","question":"当前 LTS 信息",'
                '"fields":["LTS 主版本","代号","最新补丁版本"]}],'
                '"urls":[],"clarification":""}'
            )
        return (
            '{"points":[["LTS 主版本","Node.js LTS 主版本"],'
            '["代号","Node.js LTS 代号"],'
            '["最新补丁版本","Node.js LTS 最新补丁版本"]]}'
        )

    reader._complete = complete
    plan = reader.plan_research("Node.js 当前 LTS 信息是什么？")

    assert [stage for stage, _prompt in calls] == [
        "plan",
        "search_design",
        "search_design",
        "search_design",
    ]
    assert calls[1][1].index('"subject": "Node.js"') < calls[1][1].index(
        "问题：Node.js"
    )
    current = time.localtime()
    suffix = f"{current.tm_year}年{current.tm_mon:02d}月 最新 官方公告"
    assert plan.queries == [
        f"Node.js LTS 主版本 {suffix}",
        f"Node.js LTS 代号 {suffix}",
        f"Node.js LTS 最新补丁版本 {suffix}",
    ]
    assert plan.retrieval_points == ["LTS 主版本", "代号", "最新补丁版本"]


def test_research_plan_parser_reads_field_names_without_model_filled_values() -> None:
    plan = _parse_research_plan_output(
        '{"decision":"search","scope_mode":"fixed","records":['
        '{"subject":"Node.js","question":"LTS 信息","fields":['
        '{"field":"主版本","value":"20"},{"field":"代号","value":"Y"}]}],'
        '"urls":[],"clarification":""}',
        "Node.js LTS 信息",
        4,
    )

    assert plan.records[0].fields == ["主版本", "代号"]
    assert plan.fallback is True


def test_current_version_plan_keeps_the_model_queries_unchanged() -> None:
    reader = RWKVReader(replace(Settings(), research_plan_max_queries=4))
    reader._complete = lambda *args, **kwargs: (
        '{"decision":"search","goal":"查版本","records":[],"queries":["原神 当前版本 福利"],"clarification":""}'
    )

    plan = reader.plan_research("原神当前版本和福利是什么？")

    assert plan.queries == ["原神 当前版本 福利"]


def test_research_plan_prompt_delegates_urls_and_domain_constraints() -> None:
    reader = RWKVReader(replace(Settings(), research_plan_max_queries=4))
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return (
            '{"decision":"search","goal":"查 LTS","records":[],'
            '"queries":["Node.js LTS nodejs.org"],'
            '"urls":["https://nodejs.org/en/about/previous-releases"],'
            '"clarification":""}'
        )

    reader._complete = complete

    plan = reader.plan_research(
        "请使用 https://nodejs.org/en/about/previous-releases 核对 Node.js 当前 LTS。"
    )

    assert "用户原文已有的网址放进 urls" in captured["user"]
    assert plan.queries[0].startswith("Node.js LTS nodejs.org")
    assert plan.direct_urls == ["https://nodejs.org/en/about/previous-releases"]


def test_direct_url_sufficiency_is_decided_before_search_queries() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    captured: list[str] = []

    def complete(user, **kwargs):
        captured.append(user)
        assert kwargs["stage"] == "search_design"
        return "NO"

    reader._complete = complete
    queries, points = reader.design_record_queries(
        "请阅读这个页面并给出发布日期",
        [ResearchRecord("Python 3.13.0", "发布日期", ["发布日期"])],
        ["https://www.python.org/downloads/release/python-3130/"],
    )

    assert queries == []
    assert points == []
    assert len(captured) == 1
    assert captured[0].index("已有网址") < captured[0].index("问题：")


def test_search_design_repairs_a_missing_record_branch() -> None:
    reader = RWKVReader(
        replace(Settings(), rwkv_model="test", research_plan_max_queries=4)
    )
    outputs = iter(
        [
            (
                '{"points":[["售后地点","小米 SU7 Ultra 售后地点"],'
                '["售后网点","小米 SU7 Ultra 售后网点"],'
                '["售后网络","小米 SU7 Ultra 售后服务网络"]]}'
            ),
            '{"points":[["售后地点","小米 SU7 Ultra 售后地点"]]}',
            '{"points":[["售后网点","小米 SU7 Ultra 售后网点"]]}',
        ]
    )
    reader._complete = lambda *args, **kwargs: next(outputs)

    queries, points = reader.design_record_queries(
        "关于小米 SU7 Ultra，售后在哪里，多久洗一次车？",
        [
            ResearchRecord(
                "小米 SU7 Ultra",
                "一般售后在什么地方？",
                ["售后地点", "售后网点"],
            ),
            ResearchRecord(
                "小米 SU7 Ultra",
                "多久洗一次车比较好？",
                ["洗车频率", "洗车建议"],
            ),
        ],
        [],
    )

    assert "洗车频率" in points
    assert any("多久洗一次车" in query for query in queries)
    assert len(queries) <= 4


def test_scope_discovery_uses_first_pass_material_for_concrete_records() -> None:
    reader = RWKVReader(replace(Settings(), research_plan_max_queries=4))
    captured = {}

    def complete(user, **kwargs):
        captured.setdefault("calls", []).append((user, kwargs))
        if kwargs["stage"] == "scope_candidate":
            return '{"subjects":["原神"]}'
        return (
            '{"decision":"search","scope_mode":"fixed","goal":"逐个回答",'
            '"records":[{"subject":"原神","question":"不相关的活动",'
            '"fields":["活动"]}],'
            '"queries":["原神 不相关的活动"],"clarification":""}'
        )

    reader._complete = complete
    plan = reader.discover_research_scope(
        "UNIQUE DISCOVERY QUESTION：二游当前版本和主题是什么？",
        [
            SearchHit(
                "原神版本新闻",
                "https://example.com/genshin",
                source_id="S1",
                content="原神发布了新版本及其主题活动。",
            )
        ],
        required_fields=["版本", "主题"],
    )

    candidate_prompt, candidate_kwargs = captured["calls"][0]
    assert candidate_prompt.index("原神发布了新版本") < candidate_prompt.index(
        "UNIQUE DISCOVERY QUESTION"
    )
    assert candidate_kwargs["stage"] == "scope_candidate"
    assert len(captured["calls"]) == 1
    assert plan.scope_mode == "fixed"
    assert [record.subject for record in plan.records] == ["原神"]
    assert plan.records[0].fields == ["版本", "主题"]
    assert len(plan.queries) == 1
    assert plan.queries[0].startswith("原神 版本 主题")
    assert "最新 官方公告" in plan.queries[0]
    assert "UNIQUE DISCOVERY QUESTION" in plan.records[0].question


def test_scope_discovery_rejects_an_event_seen_only_in_unrelated_pages() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *_args, **_kwargs: '{"subjects":["转盘抽奖"]}'

    with pytest.raises(SearchReaderError, match="independently identified"):
        reader.discover_research_scope(
            "主要二次元游戏各自是什么版本？",
            [
                SearchHit(
                    "动漫乐园节庆活动",
                    "https://example.com/festival",
                    source_id="S1",
                    content="乐园今天举行转盘抽奖活动。",
                )
            ],
        )


def test_scope_discovery_normalizes_a_publisher_prefixed_title() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *_args, **_kwargs: (
        '{"subjects":["新产品","库洛《鸣潮》"]}'
    )

    plan = reader.discover_research_scope(
        "主要二次元游戏各自的当前版本是什么？",
        [SearchHit(
            "二次元游戏市场与版本观察",
            "https://example.com/market",
            source_id="S1",
            content="新产品陆续发布，库洛《鸣潮》继续更新。",
        )],
        required_fields=["游戏名称", "当前版本号", "版本主题"],
    )

    assert [record.subject for record in plan.records] == ["《鸣潮》"]


def test_multi_subject_field_extraction_keeps_source_identity() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    records = [
        ResearchRecord("原神", "当前版本主题", ["版本主题"]),
        ResearchRecord("崩坏：星穹铁道", "当前版本主题", ["版本主题"]),
    ]
    shared_quote = "页面同时提到原神和崩坏：星穹铁道。"
    evidence = [
        Evidence("S1", "原神版本公告", "https://example.com/a", "资料", shared_quote),
        Evidence("S2", "崩坏：星穹铁道版本公告", "https://example.com/b", "资料", shared_quote),
        Evidence("S3", "最新版本公告", "https://example.com/c", "资料", shared_quote,
                 retrieval_query="原神 当前版本主题"),
        Evidence("S4", "最新版本公告", "https://example.com/d", "资料", shared_quote,
                 retrieval_query="崩坏：星穹铁道 当前版本主题"),
    ]
    checked = []

    def extract(record, field, item, relevance):
        checked.append((record.subject, item.source_id))
        return None

    reader._extract_one_field = extract
    reader._resolve_record_fields_together = lambda question, record, *_args: [
        FieldEvidenceSelection(record.subject, "版本主题", "", "missing")
    ]

    reader.select_evidence("原神和崩坏：星穹铁道各自的版本主题是什么？", records, evidence)

    assert set(checked) == {
        ("原神", "S1"),
        ("原神", "S3"),
        ("崩坏：星穹铁道", "S2"),
        ("崩坏：星穹铁道", "S4"),
    }


def test_online_release_title_supplies_version_and_name_without_model_guess() -> None:
    record = ResearchRecord("《样例产品》", "《样例产品》当前版本和版本主题", ["当前版本号", "版本主题"])
    evidence = Evidence(
        "S1",
        "《样例产品》官网：1.3全新版本「山海之约」现已上线",
        "https://example.com/release",
        "页面摘要",
        "页面原文",
    )

    version = _structured_field_candidates(record, "当前版本号", evidence, 1.0)
    theme = _structured_field_candidates(record, "版本主题", evidence, 1.0)

    assert [(item.fact, item.source_id) for item in version] == [("1.3版本", "S1")]
    assert [(item.fact, item.source_id) for item in theme] == [("山海之约", "S1")]
    preview = Evidence(
        "S2", "《样例产品》1.4版本「远航」即将上线", "https://example.com/preview",
        "页面摘要", "页面原文",
    )
    assert _structured_field_candidates(record, "当前版本号", preview, 1.0) == []
    dated = Evidence(
        "S3", "《样例产品》1.3版本更新公告", "https://example.com/update",
        "页面摘要", "【更新开始时间】2020/01/02 06:00。",
    )
    date_facts = _structured_field_candidates(record, "更新日期", dated, 1.0)
    assert [(item.fact, item.source_id) for item in date_facts] == [
        ("2020/01/02 06:00", "S3")
    ]


def test_online_release_title_accepts_planner_field_keys() -> None:
    record = ResearchRecord(
        "《样例产品》",
        "《样例产品》当前版本的版本号、版本名称和上线日期",
        ["version_number", "version_name", "release_date"],
    )
    evidence = Evidence(
        "S1",
        "《样例产品》官网：1.3版本「山海之约」更新公告",
        "https://example.com/release",
        "页面摘要",
        "【更新开始时间】2020/01/02 06:00。",
    )

    assert [
        _structured_field_candidates(record, field, evidence, 1.0)[0].fact
        for field in record.fields
    ] == ["1.3版本", "山海之约", "2020/01/02 06:00"]


def test_single_latin_product_plan_drops_copied_field_description() -> None:
    plan = _parse_research_plan_output(
        '{"decision":"search","scope_mode":"fixed","records":'
        '[{"subject":"Orion 最新稳定版本号和发布日期",'
        '"question":"Orion 最新稳定版本号和发布日期是什么？",'
        '"fields":["version","release_date"]}],'
        '"queries":["Orion 最新稳定版本号和发布日期 official release"]}',
        "Orion 最新稳定版本号和发布日期是什么？",
        4,
    )

    assert plan.records[0].subject == "Orion"
    assert plan.queries == ["Orion official release"]


def test_news_window_remains_in_every_search_branch() -> None:
    queries = _preserve_news_window(
        "最近一周有哪些科技动态？",
        ["机器人 官方公告", "AI 最近一周 发布", "芯片 新闻"],
    )

    assert queries == [
        "机器人 官方公告 最近一周",
        "AI 最近一周 发布",
        "芯片 新闻 最近一周",
    ]


def test_recent_news_summary_ignores_sidebar_headlines() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    prompts: list[str] = []

    def complete(user: str, **_kwargs: object) -> str:
        prompts.append(user)
        return "样例机构发布新公告。"

    reader._complete = complete
    reader._summarize_one(
        "最近一周有哪些科技动态？",
        DocumentChunk(
            "S1", "样例机构发布新公告", "https://example.com/news",
            "样例机构发布新公告。侧栏推荐：另一家公司上市。",
        ),
    )

    assert "只总结网页主标题对应的事件" in prompts[0]
    assert "忽略导航、侧栏、相关推荐" in prompts[0]


def test_compose_example_requires_service_nesting() -> None:
    question = "Docker Compose 怎样配置 healthcheck？给一个示例。"
    malformed = """```yaml
services:
web:
image: nginx
healthcheck:
test: ["CMD", "true"]
```"""
    valid = """```yaml
services:
  web:
    image: nginx
    healthcheck:
      test: ["CMD", "true"]
```"""

    assert _yaml_example_errors(malformed, question)
    assert _yaml_example_errors(valid, question) == []


def test_compose_yaml_validator_reads_indented_explanatory_blocks() -> None:
    question = "Docker Compose 怎样配置 healthcheck？给一个示例。"
    answer = """示例：
```yaml
services:
  web:
    image: nginx
    healthcheck:
      test: ["CMD", "true"]
```
字段：
  ```yaml
  healthcheck:
    timeout: 10s
  ```"""

    assert _yaml_example_errors(answer, question) == []
    assert _yaml_example_errors(answer.replace("timeout: 10s", "timeout: ["), question)


def test_final_reader_repairs_invalid_compose_nesting() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    bad = '```yaml\nservices:\nweb:\nimage: busybox\nhealthcheck:\ntest: ["CMD", "true"]\n```'
    good = '```yaml\nservices:\n  web:\n    image: busybox\n    healthcheck:\n      test: ["CMD", "true"]\n```'
    prompts: list[str] = []

    def complete(user: str, **_kwargs: object) -> str:
        prompts.append(user)
        return good if "格式问题：" in user else bad

    reader._complete = complete
    answer, warnings = reader.finalize(
        "Docker Compose 怎样配置 healthcheck？给一个示例。",
        [Evidence("S1", "Compose 服务示例", "https://example.com/docs", good, good)],
    )

    assert _yaml_example_errors(answer, "Docker Compose 怎样配置 healthcheck？给一个示例。") == []
    assert any("格式问题：" in prompt for prompt in prompts)
    assert "invalid YAML example was regenerated" in warnings


def test_missing_field_queries_rotate_to_less_searched_subjects() -> None:
    selections = [
        FieldEvidenceSelection("甲产品", "版本主题", "", "missing"),
        FieldEvidenceSelection("乙产品", "版本主题", "", "missing"),
    ]
    queries = _missing_field_queries(
        "甲产品和乙产品现在各自的版本主题是什么？",
        selections,
        ["甲产品 版本主题 官方 更新公告 最新"],
        1,
    )

    assert len(queries) == 1
    assert queries[0].startswith("乙产品 版本主题")


def test_freshness_review_can_search_the_publisher_host() -> None:
    evidence = [Evidence(
        "S1", "《样例产品》官方网站：1.2版本已上线",
        "https://official.example.com/news", "版本摘要", "1.2版本已上线",
    )]
    selected = [FieldEvidenceSelection(
        "《样例产品》", "当前版本号", "S1", "supported", "1.2版本", ["S1"]
    )]

    queries = _freshness_review_queries(
        "《样例产品》现在的版本是什么？", evidence, [], selected, 1
    )

    assert queries == [
        "site:official.example.com 《样例产品》 最新版本 上线 官方公告"
    ]


def test_source_summary_cannot_copy_current_claim_from_question() -> None:
    assert (
        _remove_copied_temporal_qualifiers(
            "绝区零当前版本为3.1，当前福利包括签到。",
            "绝区零3.1版本公告，福利包括签到。",
        )
        == "绝区零版本为3.1，福利包括签到。"
    )


def test_source_summary_removes_material_wrapper() -> None:
    assert _parse_source_summary("<材料>相关事实。</材料>") == "相关事实。"


def test_source_summary_removes_generic_intro_and_duplicate_lines() -> None:
    value = """根据提供的网页内容，以下是总结：
- 版本3.1已经上线。
- 活动正在开放。
- 版本3.1已经上线。"""

    assert _parse_source_summary(value) == "- 版本3.1已经上线。- 活动正在开放。"


def test_final_answer_removes_material_preface_and_empty_source_footer() -> None:
    value = "根据提供的材料，《绝区零》为3.1版本 [S1]。材料来源：、、 [S1]。"

    assert _clean_final_answer(value) == "《绝区零》为3.1版本 [S1]。"


def test_final_answer_removes_runaway_material_number_line() -> None:
    value = "《绝区零》为3.1版本 [S1]。\n- 版本信息来源于材料1、2、3、4、5、6"

    assert _clean_final_answer(value) == "《绝区零》为3.1版本 [S1]。"


def test_final_answer_removes_repeated_list_generation_and_empty_source_footer() -> (
    None
):
    value = """最近动态：
1. 机器人产业正在落地 [S1]。
2. 新型抽血机器人获批 [S2]。
3. 机器人产业正在落地 [S3]。
4. 机器人产业正在落地 [S1][S3]。
**来源**：、、、、 [S1][S2][S3]"""

    assert _clean_final_answer(value) == (
        "最近动态：\n1. 机器人产业正在落地 [S1]。\n2. 新型抽血机器人获批 [S2]。"
    )


def test_final_answer_deduplicates_single_line_delimited_generation() -> None:
    value = (
        "- 新增代理人 **维琳娜**（风·异常）、**诺姆**（火·击破）、"
        "**维琳娜**（风·异常）、**诺姆**（火·击破）、**维琳娜**（风·异常）"
    )

    assert _clean_final_answer(value) == (
        "- 新增代理人 **维琳娜**（风·异常）、**诺姆**（火·击破）"
    )


def test_context_retry_budget_reserves_space_below_model_limit() -> None:
    error = (
        "maximum context length is 16384 tokens. However, you requested 4090 "
        "output tokens and your prompt contains at least 12295 input tokens"
    )

    assert _context_retry_tokens(error, 4090) == 2045


def test_final_reader_retries_with_smaller_output_budget_on_context_rejection() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test", rwkv_max_tokens=4090))
    requested: list[int] = []

    def complete(user, *, max_tokens, stage):
        del user, stage
        requested.append(max_tokens)
        if len(requested) == 1:
            raise SearchReaderError(
                "finalize",
                "maximum context length is 16384 tokens; requested 4090 output "
                "tokens; prompt contains at least 12295 input tokens",
            )
        return "可用答案 [S1]"

    reader._complete = complete
    answer, warnings = reader.finalize("问题", _evidence(1))

    assert requested == [4090, 2045]
    assert answer == "可用答案 [S1]"
    assert warnings == ["final output budget reduced to 2045 tokens to fit context"]


def test_missing_citation_binding_does_not_modify_fenced_code() -> None:
    rows = _evidence(1)
    rows[0].claim = "healthcheck test uses curl"
    answer = "示例：\n```yaml [S1]\ntest: curl [S1]\n```\n`test` 执行检查。"

    bound = _attach_missing_citations(answer, rows)

    assert "```yaml\ntest: curl\n```" in bound
    assert bound.endswith("`test` 执行检查 [S1]。")


def test_selected_fact_is_narrowed_to_the_requested_field() -> None:
    fact = "原神版本为7.0版本「无神怜爱的雪国」，更新时间为2026年8月12日，停服维护补偿为原石x300。"

    assert (
        _narrow_selected_fact("版本号", fact) == "原神版本为7.0版本「无神怜爱的雪国」"
    )
    assert _narrow_selected_fact("福利", fact) == "停服维护补偿为原石x300"
    assert (
        _narrow_selected_fact(
            "主题", "二游现在的版本是3.1版本，主题包括商城上新、限时活动。"
        )
        == "商城上新、限时活动"
    )
    assert _is_explicit_benefit_fact("停服维护补偿为原石x300")
    assert not _is_explicit_benefit_fact("新增角色、区域、玩法和剧情")


def test_source_summaries_run_concurrently_and_keep_input_order() -> None:
    settings = replace(Settings(), rwkv_model="test", rwkv_summary_workers=4)
    reader = RWKVReader(settings)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def summarize_one(question, chunk):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return Evidence(
            source_id=chunk.source_id,
            title=chunk.title,
            url=chunk.url,
            claim=f"summary {chunk.source_id}",
            quote=chunk.text,
        )

    reader._summarize_one = summarize_one
    chunks = [
        DocumentChunk(
            source_id=f"S{index}",
            title=f"Source {index}",
            url=f"https://example.com/{index}",
            text=f"Material {index}",
        )
        for index in range(1, 5)
    ]

    rows, warnings = reader.summarize("question", chunks)

    assert maximum_active >= 2
    assert [row.source_id for row in rows] == ["S1", "S2", "S3", "S4"]
    assert warnings == []


def test_summary_drops_only_the_sentence_with_an_invented_number() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: (
        "发行价为150.80元。上市首日价格为999元。公司于8月19日上市。"
    )
    chunk = DocumentChunk(
        "S1",
        "公司上市资料",
        "https://example.com/1",
        "公司于8月19日上市，发行价为150.8元。",
    )

    evidence = reader._summarize_one("问题", chunk)

    assert "150.80" in evidence.claim
    assert "8月19日" in evidence.claim
    assert "999" not in evidence.claim


def test_failed_summary_keeps_the_full_bounded_page_material() -> None:
    settings = replace(Settings(), rwkv_model="test", max_evidence_chars=3000)
    reader = RWKVReader(settings)
    reader._complete = lambda *args, **kwargs: "网页声称价格为999元。"
    material = "正文" * 900 + "末尾关键事实"

    rows, warnings = reader.summarize(
        "问题", [DocumentChunk("S1", "标题", "https://example.com/1", material)]
    )

    assert rows[0].claim.endswith("末尾关键事实")
    assert len(rows[0].claim) > 1200
    assert warnings


def test_audit_material_is_bounded_before_the_model_call() -> None:
    settings = replace(Settings(), rwkv_model="test", rwkv_reduce_chunk_chars=6000)
    reader = RWKVReader(settings)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "ENOUGH"

    reader._complete = complete
    rows = [
        Evidence(
            f"S{index}",
            f"来源{index}",
            f"https://example.com/{index}",
            "材料" * 2000,
            "原文",
        )
        for index in range(1, 31)
    ]

    reader.audit("问题", rows, ["查询"])

    assert len(captured["user"]) < 8000


def test_audit_cannot_mark_missing_atomic_fields_as_enough() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: "ENOUGH"
    missing = [FieldEvidenceSelection("对象", "发布日期", "", "missing", "未找到", [])]

    audit = reader.audit(
        "对象发布日期是什么？",
        _evidence(1),
        ["对象 发布日期"],
        research_records=[ResearchRecord("对象", "发布日期", ["发布日期"])],
        field_selections=missing,
    )

    assert audit.sufficient is False
    assert audit.queries == ["对象 发布日期 官方资料"]


def test_audit_reformats_one_invalid_model_result_before_using_its_search() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    prompts: list[str] = []
    outputs = iter(["还需要再找资料", "SEARCH: 对象 官方公告"])

    def complete(prompt: str, **_kwargs: object) -> str:
        prompts.append(prompt)
        return next(outputs)

    reader._complete = complete
    audit = reader.audit("对象最近有什么变化？", _evidence(1), ["对象变化"])

    assert audit.queries == ["对象 官方公告"]
    assert len(prompts) == 2
    assert "对象最近有什么变化" in prompts[1]
    assert "还需要再找资料" in prompts[1]


def test_missing_recent_field_uses_new_public_search_terms_each_round() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *_args, **_kwargs: "ENOUGH"
    missing = [
        FieldEvidenceSelection(
            "某产品下一个版本", "update_time", "", "missing", "未找到", []
        )
    ]
    question = "某产品下一个版本什么时候更新？"

    first = reader.audit(question, _evidence(1), [question], field_selections=missing)
    second = reader.audit(
        question,
        _evidence(1),
        [question, f"{first.queries[0]} 2099年1月 最新"],
        field_selections=missing,
    )

    assert first.queries == ["某产品下一个版本 更新时间 官方 更新公告 最新"]
    assert second.queries == ["某产品下一个版本 更新时间 官方 新闻公告 预告"]
    assert "update_time" not in " ".join(first.queries + second.queries)


def test_supported_atomic_fields_stop_redundant_audit_search() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: "SEARCH: 对象 发布日期 官方资料"
    supported = [
        FieldEvidenceSelection(
            "对象", "发布日期", "S1", "supported", "2026-09-02", ["S1"]
        )
    ]

    audit = reader.audit(
        "对象发布日期是什么？",
        _evidence(1),
        ["对象 发布日期"],
        research_records=[ResearchRecord("对象", "发布日期", ["发布日期"])],
        field_selections=supported,
    )

    assert audit.sufficient is True
    assert audit.queries == []


def test_latest_release_page_requires_freshness_evidence() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: "ENOUGH"
    supported = [
        FieldEvidenceSelection(
            "Node.js", "最新补丁版本", "S1", "supported", "24.19.0", ["S1"]
        )
    ]
    evidence = [
        Evidence(
            "S1",
            "Node.js 24.19.0 (LTS)",
            "https://nodejs.org/en/blog/release/v24.19.0",
            "Node.js 24.19.0 (LTS) 于 2026-08-03 发布。",
            "Node.js 24.19.0 (LTS)",
        )
    ]

    audit = reader.audit(
        "Node.js 当前 LTS 最新补丁版本是什么？",
        evidence,
        ["Node.js 当前 LTS 最新补丁版本"],
        research_records=[ResearchRecord("Node.js", "查 LTS", ["最新补丁版本"])],
        field_selections=supported,
    )

    assert audit.sufficient is False
    assert audit.queries == [
        "Node.js 最新补丁版本 官方 releases index 版本列表 最新"
    ]
    assert "24.19.0" not in audit.queries[0]


def test_lts_field_rejects_upcoming_lts_row() -> None:
    record = ResearchRecord(
        "Node.js", "当前 LTS 主版本和最新补丁版本", ["LTS 主版本"]
    )
    table = (
        "| 26 (Upcoming LTS) | 26.8.1 |\n"
        "| 24 (LTS) | Krypton | 24.20.0 |"
    )

    assert not _field_qualifier_supported(record, "LTS 主版本", "26", table)
    assert _field_qualifier_supported(record, "LTS 主版本", "24", table)
    assert not _field_qualifier_supported(record, "最新补丁版本", "26.8.1", table)
    assert _field_qualifier_supported(record, "最新补丁版本", "24.20.0", table)
    assert not _field_qualifier_supported(
        record,
        "LTS 主版本",
        "26",
        "Node.js 26 will eventually get LTS treatment.\n"
        "| Node.js 26 | May 5, 2026 | Current |",
    )


def test_lts_field_reader_gets_compact_record_rows_before_the_summary() -> None:
    record = ResearchRecord(
        "Node.js", "当前 LTS 主版本和最新补丁版本", ["最新补丁版本"]
    )
    material = (
        "Long policy text without a record.\n"
        "* Version 26.8.1 (Current)\n"
        "* Version 24.20.0 'Krypton' (LTS)\n"
        "* Version 24.19.0 'Krypton' (LTS)"
    )

    assert _field_source_snippets(record, "最新补丁版本", material) == (
        "* Version 24.19.0 'Krypton' (LTS)\n"
        "* Version 24.20.0 'Krypton' (LTS)"
    )


def test_explicit_latest_release_evidence_passes_freshness_review() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: "SEARCH: 再查一次"
    supported = [
        FieldEvidenceSelection(
            "Node.js", "最新补丁版本", "S1", "supported", "24.20.0", ["S1"]
        )
    ]
    evidence = [
        Evidence(
            "S1",
            "Node.js releases",
            "https://nodejs.org/en/blog/release",
            "最新 LTS 版本为 Node.js 24.20.0。",
            "Latest LTS: v24.20.0",
        )
    ]

    audit = reader.audit(
        "Node.js 当前 LTS 最新补丁版本是什么？",
        evidence,
        [
            "Node.js 当前 LTS 最新补丁版本",
            "Node.js 最新补丁版本 官方 releases index 版本列表 最新",
        ],
        research_records=[ResearchRecord("Node.js", "查 LTS", ["最新补丁版本"])],
        field_selections=supported,
    )

    assert audit.sufficient is True
    assert audit.queries == []


def test_final_summary_retries_when_it_invents_a_numeric_value() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    prompts: list[str] = []

    def complete(user, **kwargs):
        prompts.append(user)
        if "待修正回答" in user:
            return "材料中的对象字段分别为100和200，无法确认唯一值 [S1][S2]。"
        return "对象字段是999。"

    reader._complete = complete
    rows = [
        Evidence("S1", "记录A", "https://a", "对象字段是100。", "对象字段是100。"),
        Evidence("S2", "记录B", "https://b", "对象字段是200。", "对象字段是200。"),
    ]
    selections = [
        FieldEvidenceSelection(
            "对象", "字段", "", "conflict", "材料存在冲突。", ["S1", "S2"]
        )
    ]

    answer, warnings = reader.finalize(
        "对象字段是什么？",
        rows,
        research_records=[ResearchRecord("对象", "问题", ["字段"])],
        evidence_selections=selections,
    )

    assert "999" not in answer
    assert "无法确认" in answer
    assert "[S1][S2]" in answer
    assert len(prompts) == 2
    assert warnings == [
        "final answer was regenerated after introducing unsupported numbers"
    ]


def test_field_selection_maps_one_webpage_per_call_then_resolves_one_record() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test", rwkv_summary_workers=4))
    rows = [
        Evidence(
            "S1",
            "宇树科技发行价",
            "https://example.com/issue",
            "PAGE_ONE 宇树科技发行价为150.8元/股。",
            "PAGE_ONE 宇树科技发行价为150.8元/股。",
            score=0.9,
        ),
        Evidence(
            "S2",
            "宇树科技上市首日行情",
            "https://example.com/open",
            "PAGE_TWO 宇树科技上市首日开盘价为1100元。",
            "PAGE_TWO 宇树科技上市首日开盘价为1100元。",
            score=0.8,
        ),
    ]
    extraction_prompts: list[str] = []

    def complete(user, **kwargs):
        if kwargs["stage"] == "evidence_extract":
            extraction_prompts.append(user)
            assert ("PAGE_ONE" in user) != ("PAGE_TWO" in user)
            if "发行价" in user.rsplit("\n\n", 1)[-1] and "PAGE_ONE" in user:
                return "宇树科技发行价为150.8元/股。"
            if "开盘价" in user.rsplit("\n\n", 1)[-1] and "PAGE_TWO" in user:
                return "宇树科技上市首日开盘价为1100元。"
            return "NONE"
        assert kwargs["stage"] == "evidence_resolve"
        assert '"发行价": ["宇树科技发行价为150.8元/股。"]' in user
        assert '"上市首日开盘价": ["宇树科技上市首日开盘价为1100元。"]' in user
        return (
            '{"facts":['
            '{"field":"发行价","fact":"150.8元/股","source_ids":["S1","S2"]},'
            '{"field":"上市首日开盘价","fact":"1100元","source_ids":["S1","S2"]}'
            "]}"
        )

    reader._complete = complete
    selections, warnings = reader.select_evidence(
        "宇树科技发行价和上市首日开盘价分别是多少？",
        [ResearchRecord("宇树科技", "价格", ["发行价", "上市首日开盘价"])],
        rows,
    )

    assert len(extraction_prompts) == 4
    assert warnings == []
    assert [(item.field, item.fact, item.source_ids) for item in selections] == [
        ("发行价", "150.8元/股", ["S1"]),
        ("上市首日开盘价", "1100元", ["S2"]),
    ]


def test_current_patch_is_selected_within_the_resolved_main_version() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test", rwkv_summary_workers=8))
    rows = [
        Evidence(
            "S1",
            "Node.js v24 Krypton",
            "https://example.com/v24",
            "SOURCE_ONE Node.js 的 LTS 主版本是 v24，代号 Krypton，补丁版本为 24.18.0。",
            "SOURCE_ONE Node.js 的 LTS 主版本是 v24，代号 Krypton，补丁版本为 24.18.0。",
        ),
        Evidence(
            "S2",
            "Node.js 24.15.0",
            "https://example.com/old",
            "SOURCE_TWO Node.js v24 的补丁版本为 24.15.0。",
            "SOURCE_TWO Node.js v24 的补丁版本为 24.15.0。",
        ),
        Evidence(
            "S3",
            "Node.js 24.20.0",
            "https://example.com/new",
            "SOURCE_THREE Node.js v24 的补丁版本为 24.20.0。",
            "SOURCE_THREE Node.js v24 的补丁版本为 24.20.0。",
        ),
    ]

    def complete(user, **kwargs):
        if kwargs["stage"] == "evidence_extract":
            request = user.rsplit("\n\n", 1)[-1]
            if "LTS 主版本" in request and "SOURCE_ONE" in user:
                return "v24"
            if "代号" in request and "SOURCE_ONE" in user:
                return "Krypton"
            if "最新补丁版本" in request:
                if "SOURCE_ONE" in user:
                    return "24.18.0"
                if "SOURCE_TWO" in user:
                    return "24.15.0"
                if "SOURCE_THREE" in user:
                    return "24.20.0"
            return "NONE"
        assert kwargs["stage"] == "evidence_resolve"
        assert '"fields"' in user
        return (
            '{"facts":['
            '{"field":"LTS 主版本","fact":"v24","source_ids":["S1"]},'
            '{"field":"代号","fact":"Krypton","source_ids":["S1"]},'
            '{"field":"最新补丁版本","fact":"24.15.0","source_ids":["S2"]}'
            "]}"
        )

    reader._complete = complete
    selections, warnings = reader.select_evidence(
        "Node.js 当前 LTS 主版本、代号和最新补丁版本分别是什么？",
        [
            ResearchRecord(
                "Node.js",
                "当前 LTS 信息",
                ["LTS 主版本", "代号", "最新补丁版本"],
            )
        ],
        rows,
    )

    assert warnings == []
    assert [(item.field, item.fact, item.source_ids) for item in selections] == [
        ("LTS 主版本", "v24", ["S1"]),
        ("代号", "Krypton", ["S1"]),
        ("最新补丁版本", "24.20.0", ["S3"]),
    ]


def test_repeated_audit_round_reuses_page_field_extractions() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test", rwkv_summary_workers=4))
    rows = [
        Evidence(
            "S1",
            "对象版本",
            "https://example.com/1",
            "对象版本为2.0。",
            "对象版本为2.0。",
        )
    ]
    calls: list[str] = []

    def complete(user, **kwargs):
        calls.append(kwargs["stage"])
        if kwargs["stage"] == "evidence_extract":
            return "对象版本为2.0。"
        return '{"facts":[{"field":"版本","fact":"2.0","source_ids":["S1"]}]}'

    reader._complete = complete
    records = [ResearchRecord("对象", "当前版本", ["版本"])]
    reader.select_evidence("对象当前版本是什么？", records, rows)
    reader.select_evidence("对象当前版本是什么？", records, rows)

    assert calls.count("evidence_extract") == 1
    assert calls.count("evidence_resolve") == 2


def test_next_version_cannot_fall_back_to_a_version_before_prior_state() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    rows = [
        Evidence(
            "S1",
            "对象6.9版本公告",
            "https://example.com/old",
            "对象版本为6.9。",
            "对象版本为6.9。",
        )
    ]

    def complete(user, **kwargs):
        if kwargs["stage"] == "evidence_extract":
            return "对象版本为6.9。"
        return '{"facts":[{"field":"版本","fact":"6.9","source_ids":["S1"]}]}'

    reader._complete = complete
    selections, _ = reader.select_evidence(
        "对象的下一个版本是什么？",
        [ResearchRecord("对象", "下一个版本", ["版本"])],
        rows,
        prior_facts=[
            {
                "subject": "对象",
                "field": "版本",
                "status": "supported",
                "fact": "7.0版本",
            }
        ],
    )

    assert selections[0].status == "missing"
    assert "晚于" in selections[0].fact


def test_summary_prompt_contains_only_material_and_one_short_request() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        captured["max_tokens"] = kwargs["max_tokens"]
        return "Relevant fact."

    reader._complete = complete
    chunk = DocumentChunk(
        source_id="S1",
        title="Source",
        url="https://example.com",
        text="UNIQUE MATERIAL",
    )

    evidence = reader._summarize_one("UNIQUE QUESTION", chunk)

    prompt = captured["user"]
    assert prompt == (
        "网页标题：Source\n网页地址：https://example.com\n网页材料：\n"
        "UNIQUE MATERIAL\n\n"
        "总结以上网页内容。只复述同一条记录明确对应的值，"
        "不把不同表格行、版本或对象的信息拼在一起；"
        "网页有表格时保留相关的完整行。\n网页摘要："
    )
    assert "UNIQUE QUESTION" not in prompt
    assert "网页标题：Source" in prompt
    assert "网页地址：https://example.com" in prompt
    assert captured["max_tokens"] == 1024
    assert evidence.claim == "Relevant fact."


def test_page_summary_uses_its_search_point_and_complete_user_question() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "原神相关事实。"

    reader._complete = complete
    chunk = DocumentChunk(
        source_id="S1",
        title="综合网页",
        url="https://example.com",
        text="网页同时提到了多个游戏。",
        retrieval_point="原神",
        retrieval_query="原神 当前版本 卡池 抽数",
    )

    evidence = reader._summarize_one("完整用户问题", chunk)

    assert captured["user"] == (
        "网页标题：综合网页\n网页地址：https://example.com\n网页材料：\n"
        "网页同时提到了多个游戏。\n\n"
        "总结以上网页中与检索点“原神”有关、可用于回答“完整用户问题”的内容。"
        "只复述同一条记录明确对应的值，不把不同表格行、版本或对象的信息拼在一起；"
        "网页有表格时保留相关的完整行。\n"
        "网页摘要："
    )
    assert "完整用户问题" in captured["user"]
    assert evidence.retrieval_point == "原神"


def test_markdown_summary_uses_a_minimal_document_specific_prompt() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "文档策略：无相关信息。"

    reader._complete = complete
    chunk = DocumentChunk(
        source_id="S1",
        title="notes.md · 文档策略",
        url="document://doc_test123#chunk-1",
        text="Markdown 按标题和段落切块，默认只使用文档。",
    )

    evidence = reader._summarize_one("总结文档策略", chunk)

    assert captured["user"] == (
        "notes.md · 文档策略\n"
        "Markdown 按标题和段落切块，默认只使用文档。\n\n"
        "问题：总结文档策略\n"
        "概括这一段本身记录的信息，保留名称、数字和限定条件。"
        "这段属于用户要求总结的文档，不要因为问题没有点名具体字段而判为无关。"
        "只有段落没有实际信息时才写“无相关信息”。\n段落摘要："
    )
    assert evidence.claim == "文档策略：无相关信息。"


def test_source_summary_only_cleans_wrappers_and_exact_duplicates() -> None:
    assert _parse_source_summary("摘要：事实一。事实一。事实二。") == "事实一。事实二。"


def test_source_summary_removes_repeated_search_prompt_boilerplate() -> None:
    value = (
        "根据提供的网页摘要，与检索词“对象 当前版本”相关的内容如下：\n"
        "- 当前版本为4.5版本。"
    )

    assert _parse_source_summary(value) == "- 当前版本为4.5版本。"


def test_foreign_branch_clauses_are_removed_without_losing_current_branch() -> None:
    summary = (
        "原神当前为7.0版本。崩坏：星穹铁道当前为4.5版本。原神主题为无神怜爱的雪国。"
    )

    cleaned = _without_foreign_points(summary, {"崩坏：星穹铁道"})

    assert cleaned == "原神当前为7.0版本。原神主题为无神怜爱的雪国。"


def test_material_date_ranking_ignores_future_dates() -> None:
    old = _latest_effective_material_timestamp("2024年3月27日更新")
    current = _latest_effective_material_timestamp("2026年8月26日发布")
    current_with_future = _latest_effective_material_timestamp(
        "2026年8月26日发布，2099年9月23日预告"
    )

    assert old < current
    assert current_with_future == current


def test_audit_output_is_either_enough_or_targeted_search_queries() -> None:
    enough = _parse_audit_output("ENOUGH", 3)
    missing = _parse_audit_output(
        "SEARCH: RWKV 最新模型发布日期 官方\nSEARCH: RWKV latest release GitHub", 3
    )

    assert enough.sufficient is True
    assert enough.queries == []
    assert missing.sufficient is False
    assert missing.queries == [
        "RWKV 最新模型发布日期 官方",
        "RWKV latest release GitHub",
    ]


def test_audit_prompt_stops_rechecking_an_already_supported_answer() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "ENOUGH"

    reader._complete = complete
    audit = reader.audit(
        "Python 3.14 正式发布于什么时候？",
        [
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
                claim="Python 3.14 正式发布日期为 2025 年 10 月 7 日。",
                quote="Material 2",
            ),
        ],
        ["Python 3.14 正式发布于什么时候？"],
    )

    assert audit.sufficient is True
    assert "Python 3.14 于 2025 年 10 月 7 日发布" in captured["user"]
    assert "足够只输出 ENOUGH" in captured["user"]
    assert "无限增加来源" not in captured["user"]


def test_contextualizer_puts_history_before_current_message_and_instruction() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        captured["stage"] = kwargs["stage"]
        return "A"

    reader._complete = complete
    question = reader.contextualize(
        "它是什么时候发布的？",
        [
            {
                "question": "RWKV 最新版本？",
                "answer": "RWKV-7-G1。",
                "state": {
                    "facts": [
                        {
                            "subject": "RWKV-7-G1",
                            "field": "版本",
                            "status": "supported",
                            "fact": "RWKV-7-G1",
                        }
                    ]
                },
            }
        ],
    )

    prompt = captured["user"]
    assert prompt.index("RWKV-7-G1") < prompt.index("它是什么时候发布的？")
    assert prompt.index("它是什么时候发布的？") < prompt.index("只输出 A 或 B")
    assert captured["stage"] == "contextualize"
    assert question == "关于RWKV-7-G1，它是什么时候发布的？"


def test_contextualizer_falls_back_to_previous_question_if_pronoun_remains() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    reader._complete = lambda *args, **kwargs: "无法判断"

    question = reader.contextualize(
        "它是什么时候发布的？",
        [{"question": "RWKV 最新版本是什么？", "answer": "RWKV-7-G1。"}],
    )

    assert question == "RWKV 最新版本是什么时候发布的？"


def test_contextualizer_uses_previous_record_subject_without_copying_broad_question():
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: "A"

    question = reader.contextualize(
        "那下一个版本呢？什么时候更新，主题是什么？",
        [
            {
                "question": "《绝区零》当前版本发生了哪些主要变化？",
                "answer": "此前答案内容不作为新事实。",
                "state": {
                    "records": [
                        {"subject": "《绝区零》当前版本发生的主要变化"}
                    ],
                    "facts": [],
                },
            }
        ],
    )

    assert question.startswith("关于《绝区零》，")
    assert "下一个版本" in question
    assert "哪些主要变化" not in question


def test_search_design_drops_model_invented_date_and_version_numbers():
    reader = RWKVReader(replace(Settings(), research_plan_max_queries=4))
    prompts = []

    def complete(user, **kwargs):
        prompts.append(user)
        return '{"points":[["直播时间","2024-10-15"],["版本","v1.0"]]}'

    reader._complete = complete
    question = "《明日方舟：终末地》最近一次前瞻直播是什么时候？这是哪个版本或活动的前瞻？"
    record = ResearchRecord(
        "《明日方舟：终末地》",
        question,
        ["直播时间", "版本"],
    )

    queries, points = reader.design_record_queries(question, [record], [])

    assert len(queries) == len(points) == 2
    assert all("2024-10-15" not in query and "v1.0" not in query for query in queries)
    assert all("《明日方舟：终末地》" in query for query in queries)
    assert all("当前日期：" in prompt for prompt in prompts)
    assert all("不得猜测日期或版本号" in prompt for prompt in prompts)


def test_next_version_date_selection_rejects_already_past_releases():
    record = ResearchRecord(
        "《绝区零》下一个版本更新时间",
        "《绝区零》下一个版本什么时候更新？",
        ["update_time"],
    )

    assert not _field_qualifier_supported(
        record,
        "update_time",
        "2026年2月6日",
        "2.6版本更新于2026年2月6日。",
    )
    assert _field_qualifier_supported(
        record,
        "update_time",
        "2099年2月6日",
        "2.6版本预计于2099年2月6日更新。",
    )


def test_finalizer_withholds_next_version_theme_when_no_future_date_is_supported():
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: pytest.fail(
        "unanchored page summaries must not override the missing future date"
    )
    subject = "绝区零下一个版本"
    evidence = [
        Evidence(
            "S1",
            "旧版本公告",
            "https://example.com/release",
            "2.6版本于2026年2月6日更新，主题为旧梦的安可曲。",
            "2.6版本于2026年2月6日更新，主题为旧梦的安可曲。",
        )
    ]
    selections = [
        FieldEvidenceSelection(
            subject, "update_time", "", "missing", "材料未能确认未来更新时间。"
        ),
        FieldEvidenceSelection(
            subject, "theme", "S1", "supported", "旧梦的安可曲"
        ),
    ]

    answer, warnings = reader.finalize(
        "关于《绝区零》，那下一个版本呢？什么时候更新，主题是什么？",
        evidence,
        research_records=[
            ResearchRecord(subject, "下一个版本", ["update_time", "theme"])
        ],
        evidence_selections=selections,
    )

    assert "没有确认晚于当前日期" in answer
    assert "旧梦的安可曲" not in answer
    assert "next-release details withheld" in warnings[0]


def test_contextualizer_turns_document_pronoun_into_a_standalone_question() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: "A"

    question = reader.contextualize(
        "它使用的数据集有多少个有效窗口？预热窗口是否包含在内？",
        [
            {
                "question": "当前管线版本是什么？这个版本修复了什么？",
                "answer": "当前版本是 2.7.3。",
            }
        ],
    )

    assert question == "当前管线版本使用的数据集有多少个有效窗口？预热窗口是否包含在内？"


def test_contextualizer_rejects_rewrite_that_drops_the_current_question() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: "无法判断"

    question = reader.contextualize(
        "它的售后在哪里，以及多久洗一次车？",
        [
            {
                "question": "小米 SU7 Ultra 的正式售价是多少？",
                "answer": "52.99 万元。",
                "state": {
                    "facts": [
                        {
                            "subject": "小米 SU7 Ultra",
                            "field": "正式售价",
                            "status": "supported",
                            "fact": "52.99 万元",
                        }
                    ]
                },
            }
        ],
    )

    assert question == "关于小米 SU7 Ultra，它的售后在哪里，以及多久洗一次车？"


def test_contextualizer_resolves_an_omitted_subject_when_model_marks_related() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: "A"

    question = reader.contextualize(
        "一般售后在什么地方？以及多久洗一次车比较好？",
        [
            {
                "question": "小米 SU7 Ultra 的价格是多少？",
                "answer": "52.99 万元。",
                "state": {
                    "facts": [
                        {
                            "subject": "小米 SU7 Ultra",
                            "field": "价格",
                            "status": "supported",
                            "fact": "52.99 万元",
                        }
                    ]
                },
            }
        ],
    )

    assert question.startswith("关于小米 SU7 Ultra，一般售后在什么地方")
    assert "多久洗一次车" in question


def test_contextualizer_keeps_an_independent_new_topic_exactly() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: "B"

    question = reader.contextualize(
        "Python 3.14 是什么时候发布的？",
        [{"question": "小米 SU7 Ultra 的价格是多少？", "answer": "52.99 万元。"}],
    )

    assert question == "Python 3.14 是什么时候发布的？"


def test_contextualizer_does_not_prepend_a_broad_document_question_to_complete_followup() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: "A"

    question = reader.contextualize(
        "当前管线版本是什么？这个版本修复了什么？",
        [
            {
                "question": "总结这四份文档，分别说明它们负责的信息。",
                "answer": "四份文档摘要。",
            }
        ],
    )

    assert question == "当前管线版本是什么？这个版本修复了什么？"


def test_final_cleaner_removes_a_repeated_semicolon_list_item() -> None:
    answer = (
        "**局限性**：生成指令可能含有噪声 [S1]；依赖大型模型 [S2]；"
        "生成指令可能含有噪声 [S1]。"
    )

    cleaned = _clean_final_answer(answer)

    assert cleaned.count("生成指令可能含有噪声") == 1
    assert "依赖大型模型 [S2]" in cleaned


def test_internal_prompt_markers_are_detected_as_leakage() -> None:
    assert _contains_prompt_leakage("当前检索点：售后地点\n检索要求：查询网点")
    assert _contains_prompt_leakage("只回答这些检索点：版本、日期")
    assert _contains_prompt_leakage("以下按问项组织，每个问项仅说明一次")
    assert _contains_prompt_leakage("这段属于用户要求总结的文档")
    assert not _contains_prompt_leakage("售后地点主要位于省会城市 [S1]。")


def test_per_document_reader_rejects_copied_summary_instructions() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: (
        "这段属于用户要求总结的文档，只有段落没有实际信息时才写无相关信息。"
    )

    with pytest.raises(SearchReaderError, match="copied summary instructions"):
        reader._summarize_one(
            "总结这份文档",
            DocumentChunk(
                "S1",
                "项目说明",
                "document://brief#chunk-1",
                "项目代号为 Cobalt Finch。",
            ),
        )


def test_contextualizer_merges_scope_acceptance_with_previous_user_task() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: pytest.fail(
        "deterministic scope acceptance should not call the model"
    )

    question = reader.contextualize(
        "都需要",
        [
            {
                "question": "二游现在的版本是什么，各个主题是什么。",
                "answer": "请问您想了解哪些二游？",
            }
        ],
    )

    assert question.startswith("二游现在的版本是什么，各个主题是什么")
    assert "无需再次询问具体名单" in question
    assert "请问您想了解哪些二游" not in question


def test_summary_numbers_must_exist_in_the_source_material() -> None:
    material = "公司于2026年8月19日上市，发行价150.80元。"

    assert (
        _unsupported_summary_numbers("2026年8月19日上市，发行价150.8元。", material)
        == set()
    )
    assert _unsupported_summary_numbers(
        "2024年8月19日上市，最新价1100元。", material
    ) == {"1.1E+3", "2024"}


def test_summary_numbers_ignore_markdown_latex_and_unicode_formatting() -> None:
    material = "发行价为 **１５０．８０** 元，开盘价为 $1{,}100$ 元。"

    assert (
        _unsupported_summary_numbers("发行价150.8元，开盘价1\\,100元。", material)
        == set()
    )


def test_summary_numbers_normalize_english_month_names() -> None:
    assert (
        _unsupported_summary_numbers(
            "正式发布日期为2024年10月7日。",
            "Release date: Oct. 7, 2024",
        )
        == set()
    )


def test_summary_dates_accept_korean_and_iso_equivalence() -> None:
    material = "발행일 2020년 4월 2일"

    assert _unsupported_summary_bindings("发布日期：2020-04-02", material) == set()


def test_final_answer_closes_an_unbalanced_code_fence() -> None:
    assert _clean_final_answer("示例：\n```js\nconst value = 1;").endswith("\n```")


def test_summary_bindings_reject_unseen_semver_and_mixed_dates() -> None:
    material = (
        "v26 - May 05, 2026 Aug 26, 2026 Current\n"
        "v24 Krypton May 06, 2025 Aug 26, 2026 LTS"
    )

    assert (
        _unsupported_summary_bindings(
            "v24.20.0 发布于2026年8月26日。", material + " v24.20.0"
        )
        == set()
    )
    assert _unsupported_summary_bindings("v26.9.0 发布于2025年8月26日。", material) == {
        "version:26.9.0",
        "date:2025-08-26",
    }
    assert (
        _unsupported_summary_numbers(
            "正式发布日期为2026年9月2日。",
            "Published September 2, 2026",
        )
        == set()
    )


def test_summary_numbers_compare_scaled_units_and_percentages() -> None:
    material = "发行4044.64万股，占比5.74%，估值为1.5亿美元。"

    assert (
        _unsupported_summary_numbers(
            "发行40,446,400股，占比5.74％，估值150 million美元。",
            material,
        )
        == set()
    )

    assert _unsupported_summary_numbers("普通数值0.0574。", material) == {"0.0574"}


def test_summary_numbers_compare_scientific_and_fraction_latex() -> None:
    material = r"参数为 $1.5 \times 10^{8}$，比例为 $\frac{3}{4}$。"

    assert _unsupported_summary_numbers("参数150000000，比例0.75。", material) == set()


def test_extreme_scientific_markup_does_not_overflow_number_validation() -> None:
    material = r"页面脚本包含 $1 \times 10^{9999999}$。"

    assert _unsupported_summary_numbers(material, material) == set()


def test_summary_numbers_still_reject_changed_scaled_values() -> None:
    material = "发行4044.64万股，占比5.74%。"

    assert _unsupported_summary_numbers("发行40,446,401股，占比5.75%。", material) == {
        "40446401",
        "5.75",
    }


def test_copied_fact_support_normalizes_decimal_zeros_but_not_word_order() -> None:
    assert _copied_fact_supported("发行价150.8元", "发行价为150.80元/股")
    assert not _copied_fact_supported(
        "2025年8月19日上市",
        "公司在2025年完成融资，随后于8月19日上市",
    )


def test_semantic_version_numbers_support_their_major_version() -> None:
    assert _unsupported_summary_numbers("7.0版本", "版本7.0.0_47144228") == set()


def test_summary_list_numbers_are_not_treated_as_factual_numbers() -> None:
    assert (
        _unsupported_summary_numbers("1. 事实一\n6. 事实六", "事实一 事实六") == set()
    )


def test_final_validation_accepts_identifiers_copied_from_question() -> None:
    material = _final_validation_material("09:12 首次报警。", "概括 Indigo 042 事件。")

    assert (
        _unsupported_summary_bindings("Indigo 042 于 09:12 报警。", material) == set()
    )


def test_final_validation_accepts_requested_date_difference() -> None:
    material = _final_validation_material(
        "草案日期为 2026-09-12，最终日期为 2026-09-18。",
        "最终日期比草案晚了几天？",
    )

    assert _unsupported_summary_bindings("最终日期晚了 6 天。", material) == set()
    assert "2026-09-12 至 2026-09-18 相差 6 天" in (
        _requested_date_difference_material(
            "草案日期为 2026-09-12，最终日期为 2026-09-18。",
            "最终日期比草案晚了几天？",
        )
    )


def test_evidence_filter_drops_only_explicitly_irrelevant_summaries() -> None:
    rows = _evidence(4)
    rows[0].published_date = "Fri, 21 Aug 2026 07:00:00 GMT"
    rows[1].published_date = "Thu, 20 Aug 2026 07:00:00 GMT"
    rows[1].claim = "无相关信息"
    rows[2].published_date = "Fri, 07 Aug 2026 07:00:00 GMT"
    rows[3].published_date = ""

    eligible = _eligible_evidence("概括本周动态", rows)

    assert [row.source_id for row in eligible] == ["S1", "S3", "S4"]


def test_relevance_order_places_strongest_material_nearest_question() -> None:
    rows = _evidence(2)
    rows[0].title = "无关主题"
    rows[0].claim = "无关内容"
    rows[0].score = 0.1
    rows[1].title = "Python 最新版本"
    rows[1].claim = "Python 发布了新版本"
    rows[1].score = 0.9

    ordered = _final_evidence_order("Python 最新版本", rows)

    assert [row.source_id for row in ordered] == ["S1", "S2"]


def test_recency_order_keeps_every_source_and_places_newest_nearest_question() -> None:
    rows = _evidence(3)
    rows[0].published_date = "Fri, 21 Aug 2026 07:00:00 GMT"
    rows[1].published_date = ""
    rows[2].published_date = "Mon, 20 Jul 2026 01:40:53 GMT"

    ordered = _final_evidence_order("今天有什么新闻", rows)

    assert [row.source_id for row in ordered] == ["S2", "S3", "S1"]


def test_evidence_order_keeps_search_point_materials_adjacent() -> None:
    rows = _evidence(4)
    rows[0].retrieval_point = "原神"
    rows[1].retrieval_point = "崩坏：星穹铁道"
    rows[2].retrieval_point = "原神"
    rows[3].retrieval_point = "崩坏：星穹铁道"

    ordered = _final_evidence_order("当前版本", rows)

    assert [row.retrieval_point for row in ordered] == [
        "原神",
        "原神",
        "崩坏：星穹铁道",
        "崩坏：星穹铁道",
    ]


def test_funnel_groups_interleaved_materials_within_each_search_point_first() -> None:
    nodes = [
        _MaterialNode(("S1",), "原神材料一", "原神"),
        _MaterialNode(("S2",), "崩铁材料一", "崩坏：星穹铁道"),
        _MaterialNode(("S3",), "原神材料二", "原神"),
        _MaterialNode(("S4",), "崩铁材料二", "崩坏：星穹铁道"),
    ]

    groups = _chunk_material_nodes_by_retrieval_point(nodes, max_chars=10_000, fan_in=4)

    assert [[node.source_ids[0] for node in group] for group in groups] == [
        ["S1", "S3"],
        ["S2", "S4"],
    ]


def test_funnel_never_crosses_search_points_when_point_summaries_need_compression() -> (
    None
):
    settings = replace(
        Settings(),
        rwkv_model="test",
        rwkv_reduce_fan_in=4,
        rwkv_reduce_chunk_chars=6000,
        rwkv_final_material_chars=1000,
    )
    reader = RWKVReader(settings)
    rows = _evidence(6)
    points = ["原神", "崩坏：星穹铁道", "绝区零"]
    for index, row in enumerate(rows):
        row.retrieval_point = points[index % len(points)]
        row.claim = "详细材料" * 100
    prompt_points: list[set[str]] = []

    def complete(user, **kwargs):
        assert kwargs["stage"] in {"reduce", "branch_finalize"}
        assert "\n问题：" not in user
        assert "检索要求：" in user
        found = set(re.findall(r"检索点：([^\\]+)\\n", user))
        prompt_points.append(found)
        source_ids = list(dict.fromkeys(re.findall(r'"(S\d+)"', user)))
        body = "压缩材料" * (
            20 if "压缩以上" in user or kwargs["stage"] == "branch_finalize" else 120
        )
        return body + "".join(f"[{source_id}]" for source_id in source_ids)

    reader._complete = complete

    material, warnings, group_count = reader._reduce_evidence("问题", rows)

    assert warnings == []
    assert group_count == 3
    assert len(material) <= settings.rwkv_final_material_chars
    assert prompt_points
    assert all(len(found) == 1 for found in prompt_points)
    assert all(f"检索点：{point}" in material for point in points)


def test_citation_binding_prefers_exact_version_and_date_support() -> None:
    rows = _evidence(2)
    rows[0].source_id = "S1"
    rows[0].claim = "RWKV-7 最新模型是 RWKV7-G1，发布日期为 2025 年 3 月 18 日。"
    rows[1].source_id = "S4"
    rows[1].claim = "RWKV-7-World-0.1B 于 2025 年 1 月发布。"
    answer = "RWKV-7 的最新模型是 RWKV-7-G1，发布日期为 2025 年 3 月 18 日。"

    bound = _attach_missing_citations(answer, rows)

    assert (
        bound == "RWKV-7 的最新模型是 RWKV-7-G1，发布日期为 2025 年 3 月 18 日 [S1]。"
    )


def test_citation_binding_skips_a_missing_field_notice() -> None:
    rows = _evidence(1)
    rows[0].title = "原神 更新时间"
    rows[0].claim = "2026年8月12日更新"

    bound = _attach_missing_citations(
        "绝区零的更新时间未在材料中明确写出，无法核实。",
        rows,
    )

    assert bound == "绝区零的更新时间未在材料中明确写出，无法核实。"


def test_subject_citations_use_all_sources_selected_for_that_subject() -> None:
    selections = [
        FieldEvidenceSelection("原神", "版本", "S1", "supported", "7.0版本"),
        FieldEvidenceSelection("原神", "更新时间", "S12", "supported", "8月12日"),
        FieldEvidenceSelection("绝区零", "版本", "S2", "supported", "3.1版本"),
    ]

    bound = _bind_subject_citations(
        "原神为7.0版本，8月12日更新 [S12]。绝区零为3.1版本 [S1]。",
        selections,
    )

    assert bound == "原神为7.0版本，8月12日更新 [S1][S12]。绝区零为3.1版本 [S2]。"


def test_atomic_citations_match_field_values_instead_of_every_subject_source() -> None:
    selections = [
        FieldEvidenceSelection(
            "宇树科技", "发行价", "S2", "supported", "150.8元/股", ["S2"]
        ),
        FieldEvidenceSelection(
            "宇树科技", "上市首日开盘价", "S4", "supported", "1100元", ["S4"]
        ),
    ]

    bound = _bind_atomic_citations(
        "宇树科技发行价为150.8元/股。上市首日开盘价为1100元。",
        selections,
    )

    assert bound == "宇树科技发行价为150.8元/股 [S2]。上市首日开盘价为1100元 [S4]。"


def test_atomic_citations_inherit_subject_and_use_one_primary_field_source() -> None:
    selections = [
        FieldEvidenceSelection(
            "原神", "当前版本", "S7", "supported", "无神怜爱的雪国", ["S7"]
        ),
        FieldEvidenceSelection(
            "原神",
            "版本主题",
            "S7",
            "supported",
            "无神怜爱的雪国",
            ["S7", "S10"],
        ),
        FieldEvidenceSelection(
            "原神", "上线时间", "S1", "supported", "2026-08-12", ["S1", "S22"]
        ),
        FieldEvidenceSelection(
            "崩坏：星穹铁道",
            "当前版本",
            "S8",
            "supported",
            "4.5版本「挥掷千星的筹码」",
            ["S8"],
        ),
        FieldEvidenceSelection(
            "崩坏：星穹铁道", "上线时间", "S11", "supported", "2026年8月", ["S11"]
        ),
    ]
    answer = (
        "原神：当前版本为无神怜爱的雪国，版本主题为无神怜爱的雪国，"
        "上线时间为2026-08-12。\n"
        "崩坏：星穹铁道：当前版本为4.5版本「挥掷千星的筹码」，"
        "上线时间为2026年8月。"
    )

    bound = _bind_atomic_citations(answer, selections)

    assert bound == (
        "原神：当前版本为无神怜爱的雪国 [S7]，"
        "版本主题为无神怜爱的雪国 [S7]，上线时间为2026-08-12 [S1]。\n"
        "崩坏：星穹铁道：当前版本为4.5版本「挥掷千星的筹码」 [S8]，"
        "上线时间为2026年8月 [S11]。"
    )


def test_atomic_citations_merge_identical_fields_into_one_source() -> None:
    selections = [
        FieldEvidenceSelection("原神", "当前版本", "S7", "supported", "无神怜爱的雪国"),
        FieldEvidenceSelection(
            "原神", "版本主题", "S7", "supported", "无神怜爱的雪国", ["S10"]
        ),
    ]

    bound = _bind_atomic_citations(
        "原神：当前版本及版本主题均为无神怜爱的雪国。", selections
    )

    assert bound == "原神：当前版本及版本主题均为无神怜爱的雪国 [S7]。"


def test_identical_adjacent_atomic_fields_are_merged_before_citation_binding() -> None:
    selections = [
        FieldEvidenceSelection("原神", "当前版本", "S7", "supported", "无神怜爱的雪国"),
        FieldEvidenceSelection("原神", "版本主题", "S7", "supported", "无神怜爱的雪国"),
        FieldEvidenceSelection("原神", "上线时间", "S1", "supported", "2026-08-12"),
    ]

    merged = _merge_duplicate_atomic_fields(
        "原神：当前版本为无神怜爱的雪国，版本主题为无神怜爱的雪国，"
        "上线时间为2026-08-12。",
        selections,
    )

    assert merged == (
        "原神：当前版本及版本主题均为无神怜爱的雪国，上线时间为2026-08-12。"
    )


def test_adjacent_facts_with_the_same_source_share_one_citation() -> None:
    compact = _coalesce_adjacent_identical_citations(
        "绝区零：当前版本为3.2版本 [S18]，"
        "版本主题为她与她的隐秘往事 [S18]，上线时间为8月28日 [S3]。"
    )

    assert compact == (
        "绝区零：当前版本为3.2版本，"
        "版本主题为她与她的隐秘往事 [S18]，上线时间为8月28日 [S3]。"
    )


def test_known_secondary_host_cannot_be_promoted_to_official() -> None:
    assert (
        _verified_source_relation("https://zh.wikipedia.org/wiki/Apex", "official")
        == "secondary"
    )
    assert (
        _verified_source_relation("https://www.ea.com/apex", "official") == "official"
    )


def test_final_answer_cleanup_preserves_natural_answer() -> None:
    raw = "回答：这是自然段答案。[s1][ S2 ]"

    assert _clean_final_answer(raw) == "这是自然段答案 [S1][S2]。"


def test_missing_citations_are_bound_without_rewriting_answer() -> None:
    rows = _evidence(2)
    rows[0].title = "量子计算突破"
    rows[0].claim = "研究团队发布了新的量子芯片。"
    rows[1].title = "AI 内存实验室"
    rows[1].claim = "美光宣布投资100亿美元建设AI内存实验室。"
    answer = "量子芯片取得新进展；美光将投资100亿美元建设AI内存实验室。"

    bound = _attach_missing_citations(answer, rows)

    assert (
        bound == "量子芯片取得新进展 [S1]；美光将投资100亿美元建设AI内存实验室 [S2]。"
    )


def test_one_clause_can_bind_two_sources_with_distinct_facts() -> None:
    rows = _evidence(2)
    rows[0].claim = "项目目标是建立一个连续多轮的资料阅读助手。"
    rows[1].claim = "最终回答最多生成 4090 token。"
    answer = "项目目标是建立一个连续多轮的资料阅读助手，最终回答最多生成 4090 token。"

    bound = _attach_missing_citations(answer, rows)

    assert (
        bound
        == "项目目标是建立一个连续多轮的资料阅读助手，最终回答最多生成 4090 token [S1][S2]。"
    )


def test_answer_lead_ending_in_colon_does_not_receive_a_fake_citation() -> None:
    answer = "以下是主要变化：\n性能得到提升。"
    rows = _evidence(1)
    rows[0].claim = "性能得到提升。"

    bound = _attach_missing_citations(answer, rows)

    assert bound == "以下是主要变化：\n性能得到提升 [S1]。"


def test_retrieval_point_citations_cannot_cross_plain_answer_branches() -> None:
    rows = _evidence(2)
    rows[0].source_id = "S1"
    rows[0].retrieval_point = "原神"
    rows[0].claim = "原神当前为7.0版本。"
    rows[1].source_id = "S10"
    rows[1].retrieval_point = "崩坏：星穹铁道"
    rows[1].claim = "崩坏：星穹铁道当前为4.5版本。"

    bound, rebound = _bind_retrieval_point_citations(
        "原神：当前为7.0版本 [S1]。\n崩坏：星穹铁道：当前为4.5版本 [S1][S10]。",
        rows,
    )

    assert rebound is True
    assert bound == (
        "原神：当前为7.0版本 [S1]。\n崩坏：星穹铁道：当前为4.5版本 [S10]。"
    )


def test_retrieval_point_heading_binds_following_lines_to_its_sources() -> None:
    rows = _evidence(2)
    rows[0].source_id = "S2"
    rows[0].retrieval_point = "原神"
    rows[0].claim = "当前版本主题为无神怜爱的雪国。"
    rows[1].source_id = "S8"
    rows[1].retrieval_point = "绝区零"
    rows[1].claim = "当前版本主题为昨日金墟。"

    bound, rebound = _bind_retrieval_point_citations(
        "### 原神\n当前版本主题为无神怜爱的雪国 [S8]。\n"
        "### 绝区零\n当前版本主题为昨日金墟 [S2]。",
        rows,
    )

    assert rebound is True
    assert "无神怜爱的雪国 [S2]" in bound
    assert "昨日金墟 [S8]" in bound


def test_model_calls_keep_full_local_analysis_record() -> None:
    settings = replace(Settings(), rwkv_model="test", rwkv_protocol="completions")
    reader = RWKVReader(settings)

    class Client:
        def completion(self, **kwargs):
            assert kwargs["user"] == "FULL INPUT"
            return "RAW OUTPUT"

    reader._client = lambda stage: Client()

    output = reader._complete("FULL INPUT", max_tokens=64, stage="summarize")
    record = reader.call_records()[0]

    assert output == "RAW OUTPUT"
    assert record["input"] == "FULL INPUT"
    assert record["output"] == "RAW OUTPUT"
    assert record["stage"] == "summarize"
    assert record["status"] == "ok"


def test_final_stages_use_the_larger_model_profile_and_4090_cap() -> None:
    settings = replace(
        Settings(),
        rwkv_base_url="http://reader.test/v1",
        rwkv_model="reader-7b",
        rwkv_final_base_url="http://final.test/v1",
        rwkv_final_api_key="final-key",
        rwkv_final_model="final-13b",
        rwkv_final_protocol="completions",
        rwkv_max_tokens=4090,
    )
    reader = RWKVReader(settings)

    assert reader._model_profile("summarize") == (
        "http://reader.test/v1",
        settings.rwkv_api_key,
        "reader-7b",
    )
    assert reader._model_profile("finalize") == (
        "http://final.test/v1",
        "final-key",
        "final-13b",
    )
    assert settings.rwkv_max_tokens == 4090


def test_contextualizer_receives_validated_previous_field_state() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "A"

    reader._complete = complete
    rewritten = reader.contextualize(
        "那下一个版本呢？",
        [
            {
                "question": "原神当前版本是什么？",
                "answer": "原神当前为7.0版本 [S1]。",
                "state": {
                    "facts": [
                        {
                            "subject": "原神",
                            "field": "版本",
                            "status": "supported",
                            "fact": "7.0版本",
                            "source_ids": ["S1"],
                        }
                    ]
                },
            }
        ],
    )

    assert "此前对象：原神" in captured["user"]
    assert '"fact": "7.0版本"' in rewritten
    assert rewritten.endswith("当前追问：那下一个版本呢？")


def test_model_call_budget_stops_unbounded_request_amplification() -> None:
    reader = RWKVReader(
        replace(Settings(), rwkv_model="test", rwkv_max_calls_per_request=2)
    )

    class Client:
        def completion(self, **kwargs):
            return "ok"

    reader._client = lambda stage: Client()
    reader._complete("a", max_tokens=8, stage="summarize")
    reader._complete("b", max_tokens=8, stage="summarize")

    with pytest.raises(SearchReaderError, match="call budget"):
        reader._complete("c", max_tokens=8, stage="summarize")


def test_funnel_reduces_twenty_source_summaries_to_five_lineage_pairs() -> None:
    settings = replace(
        Settings(),
        rwkv_model="test",
        rwkv_summary_workers=10,
        rwkv_reduce_chunk_chars=6000,
        rwkv_reduce_fan_in=4,
        rwkv_final_material_chars=9000,
    )
    reader = RWKVReader(settings)
    prompts: list[str] = []

    def complete(user, **kwargs):
        assert kwargs["stage"] == "reduce"
        prompts.append(user)
        source_ids = list(dict.fromkeys(re.findall(r'"(S\d+)"', user)))
        return "合并材料 " + "".join(f"[{source_id}]" for source_id in source_ids)

    reader._complete = complete

    material, warnings, group_count = reader._reduce_evidence("question", _evidence(20))

    assert group_count == 5
    assert warnings == []
    assert len(prompts) == 5
    assert all(len(re.findall(r'"S\d+"', prompt)) == 4 for prompt in prompts)
    assert all(f'"S{index}"' in material for index in range(1, 21))


def test_funnel_uses_another_parallel_level_when_final_material_is_too_large() -> None:
    settings = replace(
        Settings(),
        rwkv_model="test",
        rwkv_summary_workers=10,
        rwkv_reduce_chunk_chars=2400,
        rwkv_reduce_fan_in=4,
        rwkv_final_material_chars=1000,
    )
    reader = RWKVReader(settings)
    calls: list[str] = []

    def complete(user, **kwargs):
        assert kwargs["stage"] == "reduce"
        calls.append(user)
        source_ids = list(dict.fromkeys(re.findall(r'"(S\d+)"', user)))
        return ("归并信息" * 80) + "".join(f"[{source_id}]" for source_id in source_ids)

    reader._complete = complete
    material, warnings, group_count = reader._reduce_evidence("question", _evidence(20))

    assert warnings == []
    assert group_count < 5
    assert len(calls) > 5
    assert len(material) <= settings.rwkv_final_material_chars


def test_final_reader_returns_direct_natural_summary_without_list_renderer() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}
    rows = _evidence(2)
    rows[1].claim = "BOUND FACT"

    def complete(user, **kwargs):
        captured.setdefault("prompts", []).append(user)
        return "这是综合后的自然回答，不是逐来源清单。[S1][S2]"

    reader._complete = complete

    answer, warnings = reader.finalize("UNIQUE QUESTION", rows)

    prompt = captured["prompts"][0]
    assert '[["S2"], "Source 2\\nBOUND FACT"]' in prompt
    assert prompt.index("BOUND FACT") < prompt.index("UNIQUE QUESTION")
    assert prompt.index("UNIQUE QUESTION") < prompt.index("根据以上材料")
    assert '"facts"' not in prompt
    assert "逐字段证据选择" not in prompt
    assert "材料审核状态" not in prompt
    assert answer == "这是综合后的自然回答，不是逐来源清单 [S1][S2]。"
    assert warnings == []
    assert len(captured["prompts"]) == 1


def test_point_branches_are_aggregated_once_before_the_final_summary() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    rows = _evidence(2)
    rows[0].retrieval_point = "原神"
    rows[0].retrieval_query = "原神 当前版本"
    rows[0].claim = "原神事实"
    rows[1].retrieval_point = "绝区零"
    rows[1].retrieval_query = "绝区零 当前版本"
    rows[1].claim = "绝区零事实"
    calls: list[tuple[str, str]] = []

    def complete(user, **kwargs):
        calls.append((kwargs["stage"], user))
        if kwargs["stage"] == "branch_finalize":
            if "当前检索点：原神" in user:
                return "原神：原神事实 [S1]"
            return "绝区零：绝区零事实 [S2]"
        if kwargs["stage"] == "aggregate":
            return "原神：原神事实 [S1]\n绝区零：绝区零事实 [S2]"
        assert kwargs["stage"] == "finalize"
        assert "原神：原神事实 [S1]" in user
        assert "绝区零：绝区零事实 [S2]" in user
        return "原神与绝区零的最终总结 [S1][S2]"

    reader._complete = complete

    answer, warnings = reader.finalize("比较两个游戏", rows)

    assert answer == "原神与绝区零的最终总结 [S1][S2]"
    assert warnings == []
    assert [stage for stage, _ in calls].count("branch_finalize") == 2
    assert [stage for stage, _ in calls][-2:] == ["aggregate", "finalize"]
    for stage, prompt in calls:
        if stage == "branch_finalize":
            assert prompt.startswith('[["S')
            assert "问题：比较两个游戏" not in prompt
            assert prompt.index("事实") < prompt.index("检索要求：")
            assert prompt.rstrip().endswith(("原神：", "绝区零："))
        else:
            assert prompt.index("事实") < prompt.index("问题：比较两个游戏")
            cue = "汇总材料：" if stage == "aggregate" else "最终回答："
            assert prompt.rstrip().endswith(cue)


def test_document_branches_use_document_prompts_then_cross_document_summary() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    rows = [
        Evidence(
            "S1",
            "alpha.pdf · Method",
            "document://doc_alpha123#chunk-1",
            "Alpha 方法使用线性注意力。",
            "Alpha 方法使用线性注意力。",
            retrieval_point="alpha.pdf",
            retrieval_query="对比两篇论文",
        ),
        Evidence(
            "S2",
            "beta.pdf · Method",
            "document://doc_beta123#chunk-1",
            "Beta 方法使用多头注意力。",
            "Beta 方法使用多头注意力。",
            retrieval_point="beta.pdf",
            retrieval_query="对比两篇论文",
        ),
    ]
    calls: list[tuple[str, str]] = []

    def complete(user, **kwargs):
        stage = kwargs["stage"]
        calls.append((stage, user))
        if stage == "branch_finalize":
            assert "当前日期" not in user
            if "alpha.pdf" in user:
                return "alpha.pdf：Alpha 方法使用线性注意力 [S1]。"
            return "beta.pdf：Beta 方法使用多头注意力 [S2]。"
        if stage == "aggregate":
            return (
                "alpha.pdf：Alpha 使用线性注意力 [S1]。\n"
                "beta.pdf：Beta 使用多头注意力 [S2]。"
            )
        assert stage == "finalize"
        if "初稿：" not in user:
            assert "根据以上文档材料直接回答" in user
        return "Alpha 使用线性注意力，Beta 使用多头注意力 [S1][S2]。"

    reader._complete = complete
    answer, warnings = reader.finalize("对比两篇论文", rows)

    assert answer == "Alpha 使用线性注意力，Beta 使用多头注意力 [S1][S2]。"
    assert warnings == []
    assert [stage for stage, _ in calls] == [
        "branch_finalize",
        "branch_finalize",
        "finalize",
    ]


def test_document_branch_keeps_one_direct_anchor_for_each_explicit_question_item() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    question = "请总结论文：说明研究问题、方法流程、数据规模、主要结果和局限性。"
    rows = [
        Evidence(
            f"S{index}",
            f"paper.pdf · {label}",
            f"document://doc_paper#chunk-{index}",
            claim,
            claim,
            retrieval_point="paper.pdf",
            retrieval_query=question,
        )
        for index, (label, claim) in enumerate(
            [
                ("Problem", "研究问题是自动生成指令。"),
                ("Method", "方法流程包括生成、过滤和微调。"),
                ("Scale", "数据规模为 52,445 条指令和 82,000 个实例。"),
                ("Results", "主要结果是在评测中达到 51.6 分。"),
                ("Limitations", "局限性包括依赖大型模型并继承偏差。"),
            ],
            start=1,
        )
    ]
    captured = {}

    def complete(user, **kwargs):
        assert kwargs["stage"] == "branch_finalize"
        captured["user"] = user
        return "论文覆盖研究问题、方法、规模、结果和局限性 [S1][S2][S3][S4][S5]。"

    reader._complete = complete
    node = _MaterialNode(
        tuple(item.source_id for item in rows),
        "漏斗摘要只保留了研究问题 [S1]。",
        "paper.pdf",
    )
    reader._finalize_branch(
        question,
        node,
        {item.source_id: item for item in rows},
        3000,
    )

    assert _document_answer_points(question) == [
        "研究问题",
        "方法流程",
        "数据规模",
        "主要结果",
        "局限性",
    ]
    prompt = captured["user"]
    assert "问项锚点：数据规模" in prompt
    assert "52,445 条指令和 82,000 个实例" in prompt
    assert "问项锚点：局限性" in prompt
    assert prompt.index("问项锚点：数据规模") < prompt.index("问题：")


def test_broad_document_map_prompt_keeps_each_passage_in_scope() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "恢复演练必须在隔离环境执行。"

    reader._complete = complete
    evidence = reader._summarize_one(
        "总结这四份文档，分别说明它们负责的信息。",
        DocumentChunk(
            "S1",
            "operations.md · 恢复演练",
            "document://doc_ops#chunk-1",
            "恢复演练必须在隔离环境执行。",
            retrieval_point="operations.md",
            retrieval_query="总结这四份文档，分别说明它们负责的信息。",
        ),
    )

    assert "不要因为问题没有点名具体字段而判为无关" in captured["user"]
    assert evidence.claim == "恢复演练必须在隔离环境执行。"


def test_retrieval_point_aggregate_retries_with_smaller_context_budget() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    requested: list[int] = []

    def complete(user, *, max_tokens, stage):
        del user
        assert stage == "aggregate"
        requested.append(max_tokens)
        if len(requested) == 1:
            raise SearchReaderError(
                "aggregate",
                "maximum context length is 16384 tokens. However, you requested "
                "2048 output tokens and your prompt contains at least 14337 input tokens",
            )
        return "# 原神\n原神事实 [S1]\n# 绝区零\n绝区零事实 [S2]"

    reader._complete = complete
    material = (
        "[[" + '"S1"], "检索点：原神\\n原神事实 [S1]"]\n'
        "[[" + '"S2"], "检索点：绝区零\\n绝区零事实 [S2]"]'
    )
    aggregated, warnings = reader._aggregate_retrieval_points(
        "比较两个游戏", material, ["原神", "绝区零"]
    )

    assert requested == [2048, 1024]
    assert "原神事实 [S1]" in aggregated
    assert "绝区零事实 [S2]" in aggregated
    assert warnings == ["aggregate output budget reduced to 1024 tokens to fit context"]


def test_field_review_is_appended_without_replacing_final_summary_material() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    rows = _evidence(2)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "综合两份材料得到 Fact 1 和 Fact 2 [S1][S2]"

    reader._complete = complete
    answer, _ = reader.finalize(
        "question",
        rows,
        evidence_selections=[
            FieldEvidenceSelection("对象", "字段", "S2", "supported", "SELECTED ONLY")
        ],
        research_records=[ResearchRecord("对象", "question", ["字段"])],
    )

    assert '"S1"' in captured["user"]
    assert '"S2"' in captured["user"]
    assert "Fact 1" in captured["user"]
    assert "Fact 2" in captured["user"]
    assert "SELECTED ONLY" in captured["user"]
    assert captured["user"].index("Fact 2") < captured["user"].index("SELECTED ONLY")
    assert answer == "综合两份材料得到 Fact 1 和 Fact 2 [S1][S2]"


def test_supported_fields_do_not_discard_independent_web_material() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "Node.js 当前 LTS 主版本为24，代号为Krypton，最新补丁为24.20.0。"

    reader._complete = complete
    answer, _ = reader.finalize(
        "Node.js 当前 LTS 主版本、代号和最新补丁版本分别是什么？",
        [
            Evidence(
                "S1",
                "被淘汰的旧材料",
                "https://example.com/old",
                "Node.js 16 是旧 LTS。",
                "Node.js 16 是旧 LTS。",
            )
        ],
        evidence_sufficient=True,
        evidence_selections=[
            FieldEvidenceSelection("Node.js", "LTS 主版本", "S2", "supported", "24"),
            FieldEvidenceSelection("Node.js", "代号", "S3", "supported", "Krypton"),
            FieldEvidenceSelection(
                "Node.js", "最新补丁版本", "S4", "supported", "24.20.0"
            ),
        ],
        research_records=[
            ResearchRecord(
                "Node.js",
                "当前 LTS",
                ["LTS 主版本", "代号", "最新补丁版本"],
            )
        ],
    )

    assert "Node.js 16 是旧 LTS" in captured["user"]
    assert "候选提取，不代表官方身份或版本归属已核实" in captured["user"]
    assert '"fact": "24.20.0"' in captured["user"]
    assert answer == (
        "Node.js 当前 LTS 主版本为24 [S2]，代号为Krypton [S3]，"
        "最新补丁为24.20.0 [S4]。"
    )


def test_atomic_material_never_exceeds_the_configured_final_budget() -> None:
    settings = replace(Settings(), rwkv_model="test", rwkv_final_material_chars=9000)
    reader = RWKVReader(settings)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "材料不足。"

    reader._complete = complete
    records = []
    selections = []
    rows = []
    for record_index in range(4):
        subject = f"对象{record_index}-" + "甲" * 190
        fields = []
        for field_index in range(8):
            field = f"字段{field_index}-" + "乙" * 70
            fields.append(field)
            source_id = f"S{record_index * 8 + field_index + 1}"
            fact = "事实内容" + "丙" * 490
            selections.append(
                FieldEvidenceSelection(
                    subject, field, source_id, "supported", fact, [source_id]
                )
            )
            rows.append(
                Evidence(
                    source_id,
                    subject,
                    f"https://example.com/{source_id}",
                    fact,
                    fact,
                )
            )
        records.append(ResearchRecord(subject, "问题", fields))

    reader.finalize(
        "请查询这些对象的信息",
        rows,
        research_records=records,
        evidence_selections=selections,
    )

    material = captured["user"].split("\n\n问题：", 1)[0]
    assert len(material) <= settings.rwkv_final_material_chars


def test_final_summary_receives_all_page_summaries_after_field_review() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    rows = _evidence(3)
    rows[0].title = "原神版本活动"
    rows[0].quote = "原神在该版本开放了多个主题活动。"
    rows[0].claim = "原神新增主题活动与奖励说明。"
    rows[1].title = "原神版本公告"
    rows[1].quote = "原神7.0版本公告。"
    rows[1].claim = "原神当前为7.0版本。"
    rows[2].title = "无关游戏公告"
    rows[2].quote = "其他游戏的活动，页面底部推荐原神。"
    rows[2].claim = "其他游戏更新。"
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "原神为7.0版本，并开放了多个主题活动 [S1][S2]。"

    reader._complete = complete
    answer, _ = reader.finalize(
        "原神当前版本和主题是什么？",
        rows,
        evidence_selections=[
            FieldEvidenceSelection("原神", "版本", "S2", "supported", "7.0版本")
        ],
        research_records=[ResearchRecord("原神", "当前版本", ["版本"])],
    )

    assert "<来源摘要二元组>" not in captured["user"]
    assert "7.0版本" in captured["user"]
    assert "原神新增主题活动与奖励说明" in captured["user"]
    assert "其他游戏更新" in captured["user"]
    assert "逐字段证据选择" not in captured["user"]
    assert answer == "原神为7.0版本 [S2]，并开放了多个主题活动 [S1][S2]。"


def test_open_ended_question_keeps_the_summary_funnel() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    rows = _evidence(2)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "本周有两项动态 [S1][S2]。"

    reader._complete = complete
    reader.finalize(
        "最近一周有哪些科技动态？",
        rows,
        evidence_selections=[
            FieldEvidenceSelection("科技动态", "事件", "S2", "supported", "单条事件")
        ],
        research_records=[ResearchRecord("科技动态", "最近动态", ["事件"])],
    )

    assert "Fact 1" in captured["user"]
    assert "Fact 2" in captured["user"]
    assert "单条事件" not in captured["user"]


def test_atomic_field_binding_skips_procedural_and_open_ended_questions() -> None:
    records = [ResearchRecord("对象", "问题", ["字段"])]

    assert _field_binding_applicable("对象的版本号和发布日期是什么？", records)
    assert _field_binding_applicable("《示例产品》当前版本有哪些主要变化？", records)
    assert not _field_binding_applicable("最近有哪些科技动态？", records)
    assert not _field_binding_applicable("怎样配置 healthcheck？给出示例。", records)


def test_incomplete_current_release_uses_only_bound_field_facts() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *_args, **_kwargs: pytest.fail(
        "incomplete current release must not be regenerated from summaries"
    )
    answer, warnings = reader.finalize(
        "《示例产品》当前版本有哪些主要变化？",
        [Evidence("S1", "《示例产品》1.3版本公告", "https://example.com/a",
                  "未经核实的额外改动", "原文只有1.3版本号")],
        evidence_sufficient=False,
        evidence_selections=[
            FieldEvidenceSelection("《示例产品》", "版本号", "S1", "supported", "1.3", ["S1"]),
            FieldEvidenceSelection("《示例产品》", "主要变化", "", "missing"),
        ],
    )

    assert "1.3" in answer
    assert "主要变化：现有材料未能确认" in answer
    assert "未经核实的额外改动" not in answer
    assert warnings == [
        "answer limited to verified atomic fields because field evidence is incomplete"
    ]


def test_incomplete_price_fields_do_not_reenter_freeform_finalization() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    reader._complete = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("incomplete fields must not be regenerated from summaries")
    )

    answer, warnings = reader.finalize(
        "样例公司上市首日的开盘价和收盘价分别是多少？",
        _evidence(1),
        evidence_sufficient=False,
        evidence_selections=[
            FieldEvidenceSelection("样例公司", "开盘价", "S1", "supported", "10元", ["S1"]),
            FieldEvidenceSelection("样例公司", "收盘价", "", "missing", "材料未给出这个字段。"),
        ],
    )

    assert "开盘价：10元 [S1]" in answer
    assert "收盘价：现有材料未能确认。" in answer
    assert warnings == [
        "answer limited to verified atomic fields because field evidence is incomplete"
    ]


def test_complete_bounded_fields_use_selected_values_without_regeneration() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("bounded fields must not be regenerated from source summaries")
    )
    question = "样例公司上市日期和开盘价分别是多少？"
    answer, warnings = reader.finalize(
        question,
        _evidence(1),
        research_records=[ResearchRecord("样例公司", question, ["上市日期", "开盘价"])],
        evidence_sufficient=True,
        evidence_selections=[
            FieldEvidenceSelection("样例公司", "上市日期", "S1", "supported", "2026年8月19日上市", ["S1"]),
            FieldEvidenceSelection("样例公司", "开盘价", "S1", "supported", "100元", ["S1"]),
        ],
    )

    assert "上市日期：2026年8月19日上市 [S1]" in answer
    assert "开盘价：100元 [S1]" in answer
    assert warnings == ["answer rendered from verified atomic fields"]


def test_atomic_field_answer_removes_model_quote_marker() -> None:
    answer = _atomic_fallback_answer([
        FieldEvidenceSelection("样例公司", "发行价", "S1", "supported", ">\n发行价为100元", ["S1"])
    ])

    assert answer == "- 样例公司 · 发行价：发行价为100元 [S1]"


def test_audit_cannot_override_missing_field_status() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("missing field must not enter freeform finalization")
    )

    answer, _warnings = reader.finalize(
        "样例产品的价格是什么？",
        _evidence(1),
        evidence_sufficient=True,
        evidence_selections=[
            FieldEvidenceSelection("样例产品", "价格", "", "missing", "材料未给出这个字段。"),
        ],
    )

    assert "价格：现有材料未能确认。" in answer


def test_atomic_fallback_explains_unresolved_release_status() -> None:
    answer = _atomic_fallback_answer([
        FieldEvidenceSelection(
            "甲游戏", "当前版本号", "", "conflict",
            "同一发布方站点还有更高版本资料，其上线状态尚未核实。", ["S1", "S2"],
        ),
        FieldEvidenceSelection(
            "乙游戏", "当前版本号", "", "missing",
            "材料未确认该对象已有哪个版本上线。",
        ),
    ])

    assert "上线状态尚未核实。[S1][S2]" in answer
    assert "材料未确认该对象已有哪个版本上线。" in answer


def test_atomic_fallback_renders_planner_field_keys_for_readers() -> None:
    answer = _atomic_fallback_answer([
        FieldEvidenceSelection("样例产品", "version_number", "S1", "supported", "1.3版本", ["S1"]),
        FieldEvidenceSelection("样例产品", "release_date", "", "missing", "材料未给出这个字段。"),
    ])

    assert "版本号：1.3版本 [S1]" in answer
    assert "发布日期：现有材料未能确认。" in answer
    assert "version_number" not in answer


def test_empty_supported_field_is_reported_missing() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    record = ResearchRecord("样例产品", "样例产品的发布日期是什么？", ["date"])
    reader._extract_one_field = lambda *_args: None
    reader._resolve_record_fields_together = lambda *_args: [
        FieldEvidenceSelection("样例产品", "date", "S1", "supported", "", ["S1"])
    ]

    selections, _warnings = reader.select_evidence(
        "样例产品的发布日期是什么？", [record], _evidence(1)
    )

    assert selections[0].status == "missing"
    assert selections[0].source_ids == []


def test_field_value_shape_rejects_percent_as_price_and_partial_date() -> None:
    assert not _valid_field_value_shape("上市首日开盘价", "上市首日换手率85.28%")
    assert _valid_field_value_shape("上市首日开盘价", "开盘价1100元/股")
    assert not _valid_field_value_shape("上市日期", "将于8月19日上市")
    assert _valid_field_value_shape("上市日期", "2026年8月19日上市")
    assert not _valid_field_value_shape(
        "上市日期", "已受理 2026-03-20。上市委会议 2026-06-01 通过。"
    )
    assert not _valid_field_value_shape(
        "上市日期", "样例公司上市发行结果公告，2026-08-14；上市公告书提示性公告，2026-08-18"
    )
    assert not _valid_field_value_shape(
        "上市日期", "样例公司上市公告书提示性公告，2026-08-18"
    )


def test_price_fact_must_match_its_local_source_label() -> None:
    source = (
        "上市首日，发行价150.80元/股，股价开盘达到1100元。"
        "截至当日收盘，股价回落至845元。"
    )

    assert _price_fact_bound_to_field("上市首日开盘价", "1100元", source)
    assert _price_fact_bound_to_field("上市首日收盘价", "845元", source)
    assert not _price_fact_bound_to_field("上市首日收盘价", "1100元", source)


def test_price_change_is_not_the_labelled_price() -> None:
    source = (
        "样例公司上市首日开盘价为100元，收盘价为80元。"
        "与开盘价相比，收盘价较开盘价回落20元。"
    )

    assert _price_fact_bound_to_field("上市首日开盘价", "100元", source)
    assert _price_fact_bound_to_field("上市首日收盘价", "80元", source)
    assert not _price_fact_bound_to_field("上市首日开盘价", "20元", source)
    assert not _price_fact_bound_to_field("上市首日收盘价", "20元", source)


def test_original_price_clauses_survive_model_abstention() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    record = ResearchRecord(
        "样例公司", "样例公司上市首日开盘价和收盘价是多少？",
        ["上市首日开盘价", "上市首日收盘价"],
    )
    source = Evidence(
        "S1", "样例公司上市首日报价", "https://example.com/market",
        "", "样例公司上市首日，股价开盘达到1100元。截至当日收盘，股价回落至845元。",
    )
    reader._extract_one_field = lambda *_args: None
    reader._resolve_record_fields_together = lambda *_args: [
        FieldEvidenceSelection("样例公司", field, "", "missing", "材料未给出这个字段。")
        for field in record.fields
    ]

    selections, _warnings = reader.select_evidence(record.question, [record], [source])

    assert [item.fact for item in selections] == ["1100元", "845元"]
    assert all(item.status == "supported" and item.source_ids == ["S1"] for item in selections)
    assert [item.fact for item in _structured_price_candidates(record, "上市首日收盘价", source, 1.0)] == ["845元"]


def test_direct_price_candidate_stays_in_subject_paragraph_after_long_navigation() -> None:
    record = ResearchRecord("样例公司", "样例公司上市首日开盘价是多少？", ["上市首日开盘价"])
    source = Evidence(
        "S1", "样例公司上市报道", "https://example.com/market", "",
        "导航链接 " * 1800
        + "\n样例公司上市首日开盘价为100元。"
        + "\n其他公司上市首日开盘价为200元。",
    )

    candidates = _structured_price_candidates(record, "上市首日开盘价", source, 1.0)

    assert [item.fact for item in candidates] == ["100元"]


def test_original_price_source_joins_matching_selected_value() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    record = ResearchRecord("样例公司", "样例公司上市首日开盘价是多少？", ["上市首日开盘价"])
    source = Evidence(
        "S2", "样例公司上市报道", "https://example.com/report", "",
        "样例公司上市首日开盘价为100元。",
    )
    reader._extract_one_field = lambda *_args: None
    reader._resolve_record_fields_together = lambda *_args: [
        FieldEvidenceSelection("样例公司", "上市首日开盘价", "S1", "supported", "100元", ["S1"])
    ]

    selections, _warnings = reader.select_evidence(record.question, [record], [source])

    assert selections[0].status == "supported"
    assert selections[0].source_ids == ["S1", "S2"]


def test_price_extraction_rejects_unrelated_percentage_before_resolution() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *_args, **_kwargs: "上市首日换手率85.28%"
    record = ResearchRecord("样例公司", "上市首日开盘价是多少？", ["上市首日开盘价"])
    evidence = Evidence(
        "S1", "样例公司上市首日报道", "https://example.com/market",
        "上市首日换手率85.28%", "上市首日换手率85.28%",
    )

    assert reader._extract_one_field(record, "上市首日开盘价", evidence, 1.0) is None


def test_final_reader_uses_current_turn_as_answer_contract() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}
    rows = _evidence(2)

    def complete(user, **kwargs):
        captured.setdefault("prompts", []).append(user)
        return "结合偏好后的回答 [S1]"

    reader._complete = complete
    reader.finalize(
        "适合回甘偏好的口粮茶有哪些？",
        rows,
        conversation_history=[
            {"question": "口粮茶有哪些？", "answer": "可以按六大茶类选择。"}
        ],
        current_message="我喜欢回甘好的",
    )

    prompt = captured["prompts"][0]
    assert prompt.index("Fact 1") < prompt.index("我喜欢回甘好的")
    assert "适合回甘偏好的口粮茶有哪些？" not in prompt
    assert "可以按六大茶类选择" not in prompt


def test_final_reader_does_not_reanswer_previous_question_in_pronoun_followup() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "有效窗口为 48,600 个，不包含预热窗口 [S1]。"

    reader._complete = complete
    answer, _ = reader.finalize(
        "上一问：当前管线版本是什么；当前追问：它使用的数据集有多少个有效窗口？预热窗口是否包含在内？",
        [
            Evidence(
                "S1",
                "实验结果",
                "document://experiment#chunk-1",
                "River-17C 有 48,600 个有效窗口，不包含 1,200 个预热窗口。",
                "River-17C 有 48,600 个有效窗口，不包含 1,200 个预热窗口。",
            )
        ],
        current_message="它使用的数据集有多少个有效窗口？预热窗口是否包含在内？",
    )

    assert "问题：它使用的数据集有多少个有效窗口？预热窗口是否包含在内？" in captured["user"]
    assert "问题：上一问" not in captured["user"]
    assert answer.startswith("有效窗口为 48,600 个")


def test_conversation_answer_reads_history_before_request_without_citations() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        captured["stage"] = kwargs["stage"]
        return "根据此前对话内容，我们讨论了口粮茶，你偏好回甘明显的茶 [S8]。"

    reader._complete = complete
    answer = reader.answer_from_history(
        "总结你现在知道的东西",
        [
            {"question": "口粮茶有哪些？", "answer": "有红茶和乌龙茶 [S1]。"},
            {"question": "我喜欢回甘好的", "answer": "可以关注铁观音 [S8]。"},
        ],
    )

    prompt = captured["user"]
    assert captured["stage"] == "conversation"
    assert prompt.index("口粮茶有哪些？") < prompt.index("总结你现在知道的东西")
    assert "根据此前对话完成请求" in prompt
    assert "<此前对话消息>" not in prompt
    assert answer == "我们讨论了口粮茶，你偏好回甘明显的茶。"


def test_planned_field_query_keeps_its_subject():
    from rwkv_search_reader.reader import _bind_query_subjects
    from rwkv_search_reader.models import ResearchRecord
    records = [ResearchRecord("RWKV-7", "代号是什么", ["代号"])]
    assert _bind_query_subjects(["代号"], ["RWKV-7"], records) == ["RWKV-7 代号"]
    assert _bind_query_subjects(["RWKV-7 代号"], ["RWKV-7"], records) == ["RWKV-7 代号"]


def test_planned_query_collapses_repeated_explicit_title() -> None:
    from rwkv_search_reader.reader import _bind_query_subjects

    records = [ResearchRecord("《样例游戏》", "《样例游戏》下次更新", ["更新时间"])]

    assert _bind_query_subjects(
        ["《样例游戏》 《样例游戏》 更新时间"], ["更新时间"], records
    ) == ["《样例游戏》 更新时间"]


def test_query_subject_binding_does_not_mix_distinct_records():
    from rwkv_search_reader.reader import _bind_query_subjects
    from rwkv_search_reader.models import ResearchRecord
    records = [ResearchRecord("RWKV-7", "代号", ["代号"]), ResearchRecord("Python", "版本", ["版本"])]
    assert _bind_query_subjects(["代号", "版本"], ["RWKV-7", "Python"], records) == ["RWKV-7 代号", "Python 版本"]


def test_query_subject_binding_maps_field_names_back_to_their_own_record():
    from rwkv_search_reader.reader import _bind_query_subjects

    records = [
        ResearchRecord("绝区零下一个版本更新时间", "下一个版本什么时候更新", ["update_time"]),
        ResearchRecord("绝区零下一个版本主题", "下一个版本主题是什么", ["theme"]),
    ]

    assert _bind_query_subjects(
        ["下一个版本更新时间", "下一个版本主题"],
        ["update_time", "theme"],
        records,
    ) == [
        "绝区零 下一个版本更新时间",
        "绝区零 下一个版本主题",
    ]


def test_plan_research_repairs_field_only_search_before_execution():
    reader = RWKVReader(Settings())
    reader._complete = lambda *args, **kwargs: (
        '{"decision":"search","scope_mode":"fixed","goal":"查代号",'
        '"records":[{"subject":"RWKV-7","question":"代号是什么","fields":["代号"]}],'
        '"queries":["代号"],"retrieval_points":["RWKV-7"]}'
    )
    plan = reader.plan_research("RWKV-7 代号是什么？")
    assert plan.queries == ["RWKV-7 代号"]


def test_plan_keeps_records_when_model_search_design_is_invalid():
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))

    def complete(_prompt: str, *, stage: str, **_kwargs: object) -> str:
        if stage == "plan":
            return (
                '{"decision":"search","scope_mode":"fixed",'
                '"records":[{"subject":"某产品","question":"下一个版本什么时候更新，主题是什么",'
                '"fields":["update_time","theme"]}],"urls":[]}'
            )
        raise SearchReaderError(stage, "invalid search design")

    reader._complete = complete
    plan = reader.plan_research("某产品下一个版本什么时候更新，主题是什么？")

    assert len(plan.queries) == len(plan.retrieval_points) == 2
    assert all("某产品" in query and "官方公告" in query for query in plan.queries)
    assert any("更新时间" in query for query in plan.queries)
    assert any("版本主题" in query for query in plan.queries)
    assert all(re.search(r"\d{4}年\d{2}月", query) for query in plan.queries)
    assert all("update_time" not in query for query in plan.queries)


def test_fixed_plan_carries_an_explicit_title_into_generic_record_queries():
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))

    def complete(_prompt: str, *, stage: str, **_kwargs: object) -> str:
        if stage == "plan":
            return (
                '{"decision":"search","scope_mode":"fixed",'
                '"records":[{"subject":"游戏更新内容",'
                '"question":"《示例产品》当前版本有哪些变化？",'
                '"fields":["版本号","更新内容"]}],"urls":[]}'
            )
        raise SearchReaderError(stage, "invalid search design")

    reader._complete = complete
    plan = reader.plan_research("《示例产品》当前版本有哪些变化？")

    assert plan.records[0].subject.startswith("《示例产品》")
    assert plan.queries
    assert all("《示例产品》" in query for query in plan.queries)
