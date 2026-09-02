from __future__ import annotations

from rwkv_search_reader.rag_evaluation import (
    DEFAULT_DATASET,
    _grade_turn,
    _load_suite,
    _match_text,
)


def test_reference_suite_contains_twenty_questions() -> None:
    documents, cases = _load_suite(DEFAULT_DATASET)

    assert len(documents) >= 8
    assert sum(len(case["turns"]) for case in cases) == 20
    assert any(len(case["turns"]) > 1 for case in cases)


def test_rag_match_treats_field_separators_as_formatting() -> None:
    assert _match_text("index_version=5") == _match_text("index_version：5")


def test_rag_grade_checks_content_and_document_provenance() -> None:
    grade = _grade_turn(
        {
            "must_include_all": ["12,000条", "480次"],
            "must_include_any": ["Lumen", "Fjord"],
            "must_not_include": ["Rook最好"],
        },
        {
            "answer": "实验用了 12000 条工单和 480 次查询，Lumen 召回最高 [S1]。",
            "retrieval_mode": "document",
            "search_provider": "local-markdown",
            "sources": [{"source_id": "S1", "url": "document://doc#chunk-1"}],
            "evidence": [{"source_id": "S1"}],
        },
    )

    assert grade["content_pass"] is True
    assert grade["structural_pass"] is True
    assert grade["pass"] is True


def test_rag_grade_rejects_web_fallback_and_unknown_citation() -> None:
    grade = _grade_turn(
        {"must_include_all": ["7443"]},
        {
            "answer": "端口为 7443 [S9]。",
            "retrieval_mode": "forced",
            "search_provider": "tavily",
            "sources": [{"source_id": "S1", "url": "https://example.com"}],
            "evidence": [{"source_id": "S1"}],
        },
    )

    assert grade["content_pass"] is True
    assert grade["structural_pass"] is False
    assert grade["pass"] is False


def test_rag_grade_rejects_unclosed_code_fence() -> None:
    grade = _grade_turn(
        {"must_include_all": ["function add"]},
        {
            "answer": "```js\nfunction add(a, b) { return a + b; }",
            "retrieval_mode": "document",
            "search_provider": "local-markdown",
            "sources": [{"source_id": "S1", "url": "document://doc#chunk-1"}],
            "evidence": [{"source_id": "S1"}],
        },
    )

    assert grade["checks"]["balanced_code_fences"] is False
    assert grade["structural_pass"] is False
