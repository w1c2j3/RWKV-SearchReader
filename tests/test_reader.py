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
    _bind_atomic_citations,
    _bind_retrieval_point_citations,
    _bind_subject_citations,
    _chunk_material_nodes_by_retrieval_point,
    _clean_final_answer,
    _context_retry_tokens,
    _copied_fact_supported,
    _eligible_evidence,
    _field_binding_applicable,
    _final_evidence_order,
    _is_explicit_benefit_fact,
    _latest_effective_material_timestamp,
    _MaterialNode,
    _narrow_selected_fact,
    _parse_audit_output,
    _parse_research_plan_output,
    _parse_source_summary,
    _remove_copied_temporal_qualifiers,
    _unsupported_summary_numbers,
    _valid_discovered_subject,
    _verified_source_relation,
    _without_foreign_points,
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
    assert len(plan.queries) == 4
    assert "运营游戏列表" in plan.queries[0]
    assert "卡池" in plan.queries[2]
    assert plan.retrieval_points == [
        "米哈游游戏范围",
        "当前版本主题",
        "当前卡池",
        "可获得抽数",
    ]


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
        "把问题转换为检索计划"
    )
    assert "判定示例" not in captured["user"]
    assert '"subject":"明确对象"' not in captured["user"]
    assert captured["stage"] == "plan"
    assert plan.queries == ["Python 3.14 发布日期"]


def test_current_version_plan_adds_time_and_announcement_terms() -> None:
    reader = RWKVReader(replace(Settings(), research_plan_max_queries=4))
    reader._complete = lambda *args, **kwargs: (
        '{"decision":"search","goal":"查版本","records":[],"queries":["原神 当前版本 福利"],"clarification":""}'
    )

    plan = reader.plan_research("原神当前版本和福利是什么？")

    assert "最新" in plan.queries[0]
    assert "更新公告" in plan.queries[0]
    assert "活动" in plan.queries[0]


def test_scope_discovery_uses_first_pass_material_for_concrete_records() -> None:
    reader = RWKVReader(replace(Settings(), research_plan_max_queries=4))
    captured = {}

    def complete(user, **kwargs):
        captured.setdefault("calls", []).append((user, kwargs))
        if kwargs["stage"] == "scope_candidate":
            return '{"subjects":["原神"]}'
        return (
            '{"decision":"search","scope_mode":"fixed","goal":"逐个回答",'
            '"records":[{"subject":"原神","question":"当前版本和主题",'
            '"fields":["版本","主题"]}],'
            '"queries":["原神 当前版本 主题"],"clarification":""}'
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
    )

    candidate_prompt, candidate_kwargs = captured["calls"][0]
    organizer_prompt, organizer_kwargs = captured["calls"][1]
    assert candidate_prompt.index("原神发布了新版本") < candidate_prompt.index(
        "UNIQUE DISCOVERY QUESTION"
    )
    assert candidate_kwargs["stage"] == "scope_candidate"
    assert organizer_prompt.index("原神") < organizer_prompt.index(
        "UNIQUE DISCOVERY QUESTION"
    )
    assert organizer_kwargs["stage"] == "scope_discovery"
    assert plan.scope_mode == "fixed"
    assert [record.subject for record in plan.records] == ["原神"]
    assert plan.records[0].fields == ["版本", "主题"]
    assert "更新公告" in plan.queries[0]


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
        "最近动态：\n"
        "1. 机器人产业正在落地 [S1]。\n"
        "2. 新型抽血机器人获批 [S2]。"
    )


def test_final_answer_deduplicates_single_line_delimited_generation() -> None:
    value = (
        "- 新增代理人 **维琳娜**（风·异常）、**诺姆**（火·击破）、"
        "**维琳娜**（风·异常）、**诺姆**（火·击破）、**维琳娜**（风·异常）"
    )

    assert _clean_final_answer(value) == (
        "- 新增代理人 **维琳娜**（风·异常）、**诺姆**（火·击破）、"
        "**维琳娜**（风·异常）"
    )


def test_context_retry_budget_reserves_space_below_model_limit() -> None:
    error = (
        "maximum context length is 16384 tokens. However, you requested 4090 "
        "output tokens and your prompt contains at least 12295 input tokens"
    )

    assert _context_retry_tokens(error, 4090) == 2045


def test_final_reader_retries_with_smaller_output_budget_on_context_rejection() -> (
    None
):
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
    settings = replace(
        Settings(), rwkv_model="test", rwkv_reduce_chunk_chars=6000
    )
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
    missing = [
        FieldEvidenceSelection("对象", "发布日期", "", "missing", "未找到", [])
    ]

    audit = reader.audit(
        "对象发布日期是什么？",
        _evidence(1),
        ["对象 发布日期"],
        research_records=[ResearchRecord("对象", "发布日期", ["发布日期"])],
        field_selections=missing,
    )

    assert audit.sufficient is False
    assert audit.queries == ["对象 发布日期 官方资料"]


def test_conflict_sources_cannot_endorse_an_invented_final_value() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: "对象字段是999。"
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
    assert warnings


def test_field_selection_maps_one_webpage_per_call_then_resolves_one_record() -> None:
    reader = RWKVReader(
        replace(Settings(), rwkv_model="test", rwkv_summary_workers=4)
    )
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
        assert '"field": "发行价"' in user
        assert '"field": "上市首日开盘价"' in user
        return (
            '{"facts":['
            '{"field":"发行价","fact":"150.8元/股","source_ids":["S1"]},'
            '{"field":"上市首日开盘价","fact":"1100元","source_ids":["S2"]}'
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


def test_repeated_audit_round_reuses_page_field_extractions() -> None:
    reader = RWKVReader(
        replace(Settings(), rwkv_model="test", rwkv_summary_workers=4)
    )
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
    assert prompt == "UNIQUE MATERIAL\n\n总结以上网页内容。\n网页摘要："
    assert "UNIQUE QUESTION" not in prompt
    assert "来源材料" not in prompt
    assert "URL" not in prompt
    assert captured["max_tokens"] == 1024
    assert evidence.claim == "Relevant fact."


def test_page_summary_uses_only_its_short_search_point_as_focus() -> None:
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
        "网页同时提到了多个游戏。\n\n"
        "总结以上网页中与检索词“原神 当前版本 卡池 抽数”有关的内容。\n"
        "网页摘要："
    )
    assert "完整用户问题" not in captured["user"]
    assert evidence.retrieval_point == "原神"


def test_markdown_summary_keeps_raw_fragment_when_model_marks_it_missing() -> None:
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
        "Markdown 按标题和段落切块，默认只使用文档。\n\n"
        "总结以上网页内容。\n网页摘要："
    )
    assert evidence.claim == "Markdown 按标题和段落切块，默认只使用文档。"


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
        "原神当前为7.0版本。"
        "崩坏：星穹铁道当前为4.5版本。"
        "原神主题为无神怜爱的雪国。"
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
        return "RWKV-7-G1 是什么时候发布的？"

    reader._complete = complete
    question = reader.contextualize(
        "它是什么时候发布的？",
        [{"question": "RWKV 最新版本？", "answer": "RWKV-7-G1。"}],
    )

    prompt = captured["user"]
    assert prompt.index("RWKV 最新版本？") < prompt.index("它是什么时候发布的？")
    assert prompt.index("它是什么时候发布的？") < prompt.index("把当前消息改写")
    assert captured["stage"] == "contextualize"
    assert question == "RWKV-7-G1 是什么时候发布的？"


def test_contextualizer_falls_back_to_previous_question_if_pronoun_remains() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    reader._complete = lambda *args, **kwargs: "它是什么时候发布的？"

    question = reader.contextualize(
        "它是什么时候发布的？",
        [{"question": "RWKV 最新版本是什么？", "answer": "RWKV-7-G1。"}],
    )

    assert question == "上一问：RWKV 最新版本是什么；当前追问：它是什么时候发布的？"


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

    assert _unsupported_summary_numbers("发行价150.8元，开盘价1\\,100元。", material) == set()


def test_summary_numbers_compare_scaled_units_and_percentages() -> None:
    material = "发行4044.64万股，占比5.74%，估值为1.5亿美元。"

    assert _unsupported_summary_numbers(
        "发行40,446,400股，占比5.74％，估值150 million美元。",
        material,
    ) == set()

    assert _unsupported_summary_numbers("普通数值0.0574。", material) == {"0.0574"}


def test_summary_numbers_compare_scientific_and_fraction_latex() -> None:
    material = r"参数为 $1.5 \times 10^{8}$，比例为 $\frac{3}{4}$。"

    assert _unsupported_summary_numbers("参数150000000，比例0.75。", material) == set()


def test_extreme_scientific_markup_does_not_overflow_number_validation() -> None:
    material = r"页面脚本包含 $1 \times 10^{9999999}$。"

    assert _unsupported_summary_numbers(material, material) == set()


def test_summary_numbers_still_reject_changed_scaled_values() -> None:
    material = "发行4044.64万股，占比5.74%。"

    assert _unsupported_summary_numbers(
        "发行40,446,401股，占比5.75%。", material
    ) == {"40446401", "5.75"}


def test_copied_fact_support_normalizes_decimal_zeros_but_not_word_order() -> None:
    assert _copied_fact_supported("发行价150.8元", "发行价为150.80元/股")
    assert not _copied_fact_supported(
        "2025年8月19日上市",
        "公司在2025年完成融资，随后于8月19日上市",
    )


def test_semantic_version_numbers_support_their_major_version() -> None:
    assert _unsupported_summary_numbers("7.0版本", "版本7.0.0_47144228") == set()


def test_summary_list_numbers_are_not_treated_as_factual_numbers() -> None:
    assert _unsupported_summary_numbers("1. 事实一\n6. 事实六", "事实一 事实六") == set()


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

    groups = _chunk_material_nodes_by_retrieval_point(
        nodes, max_chars=10_000, fan_in=4
    )

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
            20
            if "压缩以上" in user or kwargs["stage"] == "branch_finalize"
            else 120
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
        "原神：当前为7.0版本 [S1]。\n"
        "崩坏：星穹铁道：当前为4.5版本 [S1][S10]。",
        rows,
    )

    assert rebound is True
    assert bound == (
        "原神：当前为7.0版本 [S1]。\n"
        "崩坏：星穹铁道：当前为4.5版本 [S10]。"
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
        return "原神在7.0版本之后的下一个版本是什么？"

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

    assert "verified_state" in captured["user"]
    assert "7.0版本" in captured["user"]
    assert rewritten == "原神在7.0版本之后的下一个版本是什么？"


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

    material, warnings, group_count = reader._reduce_evidence(
        "question", _evidence(20)
    )

    assert group_count == 5
    assert warnings == []
    assert len(prompts) == 5
    assert all(len(re.findall(r'"S\d+"', prompt)) == 4 for prompt in prompts)
    assert all(f'"S{index}"' in material for index in range(1, 21))


def test_funnel_uses_another_parallel_level_when_final_material_is_too_large() -> (
    None
):
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
        return ("归并信息" * 80) + "".join(
            f"[{source_id}]" for source_id in source_ids
        )

    reader._complete = complete
    material, warnings, group_count = reader._reduce_evidence(
        "question", _evidence(20)
    )

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
    assert '[["S2"], "BOUND FACT"]' in prompt
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


def test_field_bound_final_uses_only_atomic_field_records() -> None:
    reader = RWKVReader(replace(Settings(), rwkv_model="test"))
    rows = _evidence(2)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "只使用已绑定字段 [S2]"

    reader._complete = complete
    answer, _ = reader.finalize(
        "question",
        rows,
        evidence_selections=[
            FieldEvidenceSelection(
                "对象", "字段", "S2", "supported", "SELECTED ONLY"
            )
        ],
        research_records=[ResearchRecord("对象", "question", ["字段"])],
    )

    assert '"S2"' in captured["user"]
    assert "Fact 1" not in captured["user"]
    assert "Fact 2" not in captured["user"]
    assert "SELECTED ONLY" in captured["user"]
    assert answer == "只使用已绑定字段 [S2]"


def test_atomic_material_never_exceeds_the_configured_final_budget() -> None:
    settings = replace(
        Settings(), rwkv_model="test", rwkv_final_material_chars=9000
    )
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


def test_field_bound_final_does_not_reintroduce_unselected_page_summaries() -> None:
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
    assert "原神新增主题活动与奖励说明" not in captured["user"]
    assert "其他游戏更新" not in captured["user"]
    assert "逐字段证据选择" not in captured["user"]
    assert answer == "原神为7.0版本 [S2]。"


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
    assert not _field_binding_applicable("最近有哪些科技动态？", records)
    assert not _field_binding_applicable("怎样配置 healthcheck？给出示例。", records)


def test_final_reader_receives_resolved_question_without_duplicate_history() -> None:
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
    assert prompt.index("Fact 1") < prompt.index("适合回甘偏好的口粮茶有哪些？")
    assert "可以按六大茶类选择" not in prompt
    assert "我喜欢回甘好的" not in prompt


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
