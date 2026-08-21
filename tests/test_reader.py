from __future__ import annotations

import threading
import time
from dataclasses import replace

from rwkv_search_reader.config import Settings
from rwkv_search_reader.models import DocumentChunk, Evidence
from rwkv_search_reader.reader import (
    RWKVReader,
    _answer_limit,
    _bind_facts_to_support,
    _eligible_evidence,
    _format_bound_answer,
    _grounding_issue,
    _latest_explicit_date,
    _parse_bound_facts,
    _parse_source_summary,
    _verified_source_relation,
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


def test_citation_audit_accepts_known_source() -> None:
    assert citation_warnings("Answer [S1]", _evidence()) == []


def test_citation_audit_reports_missing_and_unknown_sources() -> None:
    assert "no [S1]-style" in citation_warnings("Answer", _evidence())[0]
    assert "S9" in citation_warnings("Answer [S9]", _evidence())[0]


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


def test_summary_prompt_puts_material_before_question_and_requirements() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "summary"

    reader._complete = complete
    chunk = DocumentChunk(
        source_id="S1",
        title="Source",
        url="https://example.com",
        text="UNIQUE MATERIAL",
    )

    reader._summarize_one("UNIQUE QUESTION", chunk)

    prompt = captured["user"]
    assert prompt.index("UNIQUE MATERIAL") < prompt.index("UNIQUE QUESTION")
    assert prompt.index("UNIQUE QUESTION") < prompt.index("只根据这一条材料")


def test_stock_summary_prompt_requests_only_latest_price_and_time() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
        return "source_relation: secondary\n最近收盘价为687元。"

    reader._complete = complete
    chunk = DocumentChunk(
        source_id="S1",
        title="宇树科技上市次日行情",
        url="https://example.com/quote",
        text="首日高点1100元，8月20日收盘价687元。",
    )

    reader._summarize_one("宇树股价", chunk)

    prompt = captured["user"]
    assert "否则选最近日期的收盘价" in prompt
    assert "只允许出现一个股价数值" in prompt


def test_grounding_rejects_date_and_price_relation_missing_from_material() -> None:
    material = "8月19日开盘上涨，股价报1100元。"

    assert "unsupported date" in _grounding_issue(
        "8月20日收盘价1100元。", material
    )
    assert "closing-price" in _grounding_issue("当日收盘价1100元。", material)
    assert _grounding_issue("8月19日股价报1100元。", material) == ""
    assert _latest_explicit_date("机器人极限速度12.66m/s，8月19日发布。") == (
        2026,
        8,
        19,
    )


def test_model_calls_keep_full_local_analysis_record() -> None:
    settings = replace(Settings(), rwkv_model="test", rwkv_protocol="completions")
    reader = RWKVReader(settings)

    class Client:
        def completion(self, **kwargs):
            assert kwargs["user"] == "FULL INPUT"
            return "RAW OUTPUT"

    reader._client = lambda: Client()

    output = reader._complete("FULL INPUT", max_tokens=64, stage="summarize")
    record = reader.call_records()[0]

    assert output == "RAW OUTPUT"
    assert record["input"] == "FULL INPUT"
    assert record["output"] == "RAW OUTPUT"
    assert record["stage"] == "summarize"
    assert record["status"] == "ok"


def test_bound_fact_parser_keeps_only_valid_binary_source_pairs() -> None:
    output = (
        'prefix {"facts":[["S1","supported fact"],["S9","unknown"],'
        '["S2","second fact"],["S1","too","many"]]} suffix'
    )

    facts = _parse_bound_facts(output, {"S1", "S2"})

    assert facts == [("S1", "supported fact"), ("S2", "second fact")]


def test_bound_fact_parser_recovers_rwkv_misclosed_json_pair() -> None:
    output = '{"facts":[["S7","official fact"]}]'

    facts = _parse_bound_facts(output, {"S7"})

    assert facts == [("S7", "official fact")]


def test_bound_fact_parser_accepts_flat_single_pair_from_rwkv() -> None:
    output = '{"facts":["S7","official fact"]}'

    facts = _parse_bound_facts(output, {"S7"})

    assert facts == [("S7", "official fact")]


def test_bound_fact_parser_accepts_flat_multiple_pairs_from_rwkv() -> None:
    output = '{"facts":["S1","first fact","S2","second fact"]}'

    facts = _parse_bound_facts(output, {"S1", "S2"})

    assert facts == [("S1", "first fact"), ("S2", "second fact")]


def test_answer_limit_uses_requested_count_and_atomic_questions() -> None:
    assert _answer_limit("列出 3 条新闻") == 3
    assert _answer_limit("正式发布日期是什么？") == 1
    assert _answer_limit("主题是什么，角色是谁，时间是什么？") == 3
    assert _answer_limit("宇树股价") == 1
    assert _answer_limit("概括本周科技动态") == 3


def test_source_relation_is_bound_and_official_request_filters_secondary() -> None:
    relation, summary = _parse_source_summary(
        "source_relation: official\nsummary: first-party fact"
    )
    rows = _evidence(2)
    rows[0].source_relation = relation
    rows[0].claim = summary
    rows[1].source_relation = "secondary"

    eligible = _eligible_evidence("请依据官方页面回答", rows)

    assert relation == "official"
    assert summary == "first-party fact"
    assert [row.source_id for row in eligible] == ["S1"]


def test_current_stock_price_filters_ipo_and_concept_index_materials() -> None:
    rows = _evidence(3)
    rows[0].title = "宇树科技上市首日开盘"
    rows[0].claim = "发行价150.80元，上市首日开盘价1100元。"
    rows[1].title = "宇树科技上市次日行情"
    rows[1].claim = "2026年8月20日收盘价为687元。"
    rows[2].title = "宇树科技概念股票实时行情"
    rows[2].claim = "概念指数最新价976.11元。"

    eligible = _eligible_evidence("宇树股价", rows)

    assert [row.source_id for row in eligible] == ["S2"]


def test_current_stock_price_uses_evidence_time_before_final_model() -> None:
    rows = _evidence(2)
    rows[0].claim = "宇树科技当前股价为900元/股。"
    rows[0].quote = "8月19日，截至发稿股价约900元/股。"
    rows[0].published_date = "Wed, 19 Aug 2026 10:00:00 GMT"
    rows[1].claim = "宇树科技当前股价为687元/股。"
    rows[1].quote = "8月20日，截至收盘股价报687元/股。"
    rows[1].published_date = "Thu, 20 Aug 2026 18:00:00 GMT"

    eligible = _eligible_evidence("宇树股价", rows)

    assert [row.source_id for row in eligible] == ["S2"]


def test_known_secondary_host_cannot_be_promoted_to_official() -> None:
    assert (
        _verified_source_relation("https://zh.wikipedia.org/wiki/Apex", "official")
        == "secondary"
    )
    assert _verified_source_relation("https://www.ea.com/apex", "official") == "official"


def test_bound_answer_renderer_attaches_each_source_to_its_fact() -> None:
    answer = _format_bound_answer([("S3", "Fact A"), ("S8", "Fact B")])

    assert answer == "- Fact A [S3]\n- Fact B [S8]"


def test_bound_answer_renderer_removes_near_duplicate_facts() -> None:
    answer = _format_bound_answer(
        [
            ("S3", "当前赛季为诸神烙印，开始日期是2026年8月4日。"),
            ("S8", "《Apex》当前赛季为诸神烙印，开始日期是2026年8月4日。"),
        ]
    )

    assert answer == "当前赛季为诸神烙印，开始日期是2026年8月4日。 [S3]"


def test_bound_answer_renderer_drops_empty_fact_and_honors_limit() -> None:
    answer = _format_bound_answer(
        [
            ("S1", "无相关信息"),
            ("S4", "该来源明确支持的一个事实"),
            ("S2", "Fact A"),
            ("S3", "Fact B"),
        ],
        limit=1,
    )

    assert answer == "Fact A [S2]"


def test_fact_binding_repairs_unsupported_numeric_source() -> None:
    rows = _evidence(3)
    rows[0].claim = "公司于2026年8月19日上市。"
    rows[1].claim = "上市首日收盘价为825元，日期为2026年8月19日。"
    rows[2].claim = "发行价为150.80元。"

    facts = _bind_facts_to_support(
        [("S1", "公司股价为825元，数据时间是2026年8月19日。")], rows
    )

    assert facts == [("S2", "公司股价为825元，数据时间是2026年8月19日。")]


def test_final_prompt_keeps_source_and_fact_pairs_bound_before_requirements() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}
    rows = _evidence(2)
    rows[1].claim = "BOUND FACT"

    def complete(user, **kwargs):
        captured["user"] = user
        return '{"facts":[["S2","BOUND FACT"]]}'

    reader._complete = complete

    answer, warnings = reader.finalize("UNIQUE QUESTION", rows)

    prompt = captured["user"]
    assert '["S2", {' in prompt
    assert prompt.index("BOUND FACT") < prompt.index("UNIQUE QUESTION")
    assert prompt.index("UNIQUE QUESTION") < prompt.index("请把上面的来源摘要")
    assert "候选事实" not in prompt
    assert "不要输出思维链" in prompt
    assert answer == "BOUND FACT [S2]"
    assert warnings == []
