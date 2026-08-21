from __future__ import annotations

import json
from dataclasses import replace

from rwkv_search_reader.config import Settings
from rwkv_search_reader.evaluation import _load_cases, build_summary


def test_load_cases_and_build_structural_summary(tmp_path) -> None:
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "a", "category": "release", "query": "Question A"},
                    {"case_id": "b", "category": "release", "query": "Question B"},
                ]
            }
        ),
        encoding="utf-8",
    )
    cases = _load_cases(dataset)
    records = {
        "a": {
            **cases[0],
            "status": "ok",
            "elapsed_seconds": 2.0,
            "result": {"answer": "Answer [S1]", "degraded": False},
        },
        "b": {
            **cases[1],
            "status": "error",
            "elapsed_seconds": 1.0,
            "error_stage": "search",
        },
    }
    settings = replace(Settings(), rwkv_model="rwkv-test")

    summary = build_summary(
        records=records,
        cases=cases,
        dataset=dataset,
        settings=settings,
    )

    assert summary["counts"]["ok"] == 1
    assert summary["counts"]["error"] == 1
    assert summary["counts"]["with_source_citations"] == 1
    assert summary["counts"]["with_only_valid_source_citations"] == 0
    assert summary["counts"]["with_invalid_source_citations"] == 1
    assert summary["counts"]["empty_answers"] == 0
    assert summary["counts"]["reasoning_markup_leaks"] == 0
    assert summary["answer_chars"]["max"] == len("Answer [S1]")
    assert summary["error_stages"] == {"search": 1}
