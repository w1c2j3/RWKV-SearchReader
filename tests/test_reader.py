from __future__ import annotations

import re
import threading
import time
from dataclasses import replace

from rwkv_search_reader.config import Settings
from rwkv_search_reader.models import DocumentChunk, Evidence
from rwkv_search_reader.reader import (
    RWKVReader,
    _attach_missing_citations,
    _clean_final_answer,
    _eligible_evidence,
    _final_evidence_order,
    _parse_audit_output,
    _parse_source_summary,
    _unsupported_summary_numbers,
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


def test_citation_audit_accepts_known_sources() -> None:
    assert citation_warnings("Answer [S1][S2]", _evidence(2)) == []


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


def test_summary_prompt_is_generic_and_puts_material_before_request() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}

    def complete(user, **kwargs):
        captured["user"] = user
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
    assert prompt.index("UNIQUE MATERIAL") < prompt.index("UNIQUE QUESTION")
    assert prompt.index("UNIQUE QUESTION") < prompt.index("阅读这一个网页")
    assert "股价" not in prompt
    assert "本周最重要" not in prompt
    assert evidence.claim == "Relevant fact."


def test_source_summary_only_cleans_wrappers_and_exact_duplicates() -> None:
    assert _parse_source_summary("摘要：事实一。事实一。事实二。") == "事实一。事实二。"


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


def test_summary_numbers_must_exist_in_the_source_material() -> None:
    material = "公司于2026年8月19日上市，发行价150.80元。"

    assert _unsupported_summary_numbers(
        "2026年8月19日上市，发行价150.8元。", material
    ) == set()
    assert _unsupported_summary_numbers(
        "2024年8月19日上市，最新价1100元。", material
    ) == {"1.1E+3", "2024"}


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


def test_citation_binding_prefers_exact_version_and_date_support() -> None:
    rows = _evidence(2)
    rows[0].source_id = "S1"
    rows[0].claim = "RWKV-7 最新模型是 RWKV7-G1，发布日期为 2025 年 3 月 18 日。"
    rows[1].source_id = "S4"
    rows[1].claim = "RWKV-7-World-0.1B 于 2025 年 1 月发布。"
    answer = "RWKV-7 的最新模型是 RWKV-7-G1，发布日期为 2025 年 3 月 18 日。"

    bound = _attach_missing_citations(answer, rows)

    assert bound == "RWKV-7 的最新模型是 RWKV-7-G1，发布日期为 2025 年 3 月 18 日 [S1]。"


def test_known_secondary_host_cannot_be_promoted_to_official() -> None:
    assert (
        _verified_source_relation("https://zh.wikipedia.org/wiki/Apex", "official")
        == "secondary"
    )
    assert _verified_source_relation("https://www.ea.com/apex", "official") == "official"


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

    assert bound == "量子芯片取得新进展 [S1]；美光将投资100亿美元建设AI内存实验室 [S2]。"


def test_answer_lead_ending_in_colon_does_not_receive_a_fake_citation() -> None:
    answer = "以下是主要变化：\n性能得到提升。"
    rows = _evidence(1)
    rows[0].claim = "性能得到提升。"

    bound = _attach_missing_citations(answer, rows)

    assert bound == "以下是主要变化：\n性能得到提升 [S1]。"


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


def test_evidence_groups_are_reduced_concurrently_and_keep_all_source_ids() -> None:
    settings = replace(Settings(), rwkv_model="test", rwkv_summary_workers=5)
    reader = RWKVReader(settings)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def complete(user: str, *, stage: str, **kwargs: object) -> str:
        del kwargs
        assert stage == "reduce"
        source_ids = list(dict.fromkeys(re.findall(r'"(S\d+)"', user)))
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return f"组摘要 [{source_ids[0]}]"

    reader._complete = complete

    material, warnings, group_count = reader._reduce_evidence(
        "question", _evidence(6)
    )

    assert maximum_active >= 2
    assert group_count == 3
    assert warnings == []
    assert all(f'"S{index}"' in material for index in range(1, 7))


def test_final_reader_returns_direct_natural_summary_without_list_renderer() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}
    rows = _evidence(2)
    rows[1].claim = "BOUND FACT"

    def complete(user, **kwargs):
        captured["user"] = user
        return "这是综合后的自然回答，不是逐来源清单。[S1][S2]"

    reader._complete = complete

    answer, warnings = reader.finalize("UNIQUE QUESTION", rows)

    prompt = captured["user"]
    assert '["S2", {' in prompt
    assert prompt.index("BOUND FACT") < prompt.index("UNIQUE QUESTION")
    assert prompt.index("UNIQUE QUESTION") < prompt.index("根据上面的材料")
    assert '"facts"' not in prompt
    assert "不输出 JSON" in prompt
    assert "组织 4–8 个互不重复的信息维度" in prompt
    assert answer == "这是综合后的自然回答，不是逐来源清单 [S1][S2]。"
    assert warnings == []


def test_final_reader_receives_prior_conversation_as_context_not_evidence() -> None:
    settings = replace(Settings(), rwkv_model="test")
    reader = RWKVReader(settings)
    captured = {}
    rows = _evidence(2)

    def complete(user, **kwargs):
        captured["user"] = user
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

    prompt = captured["user"]
    assert prompt.index("Fact 1") < prompt.index("口粮茶有哪些？")
    assert prompt.index("口粮茶有哪些？") < prompt.index("我喜欢回甘好的")
    assert prompt.index("我喜欢回甘好的") < prompt.index("此前对话用于保持连续语境")
    assert "不是本轮事实来源" in prompt


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
    assert "不要把当前请求里的泛称" in prompt
    assert "当前用户请求不属于待总结的历史内容" in prompt
    assert "不要按轮次复述" in prompt
    assert answer == "我们讨论了口粮茶，你偏好回甘明显的茶。"
