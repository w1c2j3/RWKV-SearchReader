from __future__ import annotations

import argparse
import json
import re
import statistics
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, get_settings
from .conversations import ConversationStore
from .pipeline import SearchReaderPipeline

DEFAULT_DATASET = PROJECT_ROOT / "evals" / "rag_reference_20_20260902.json"
_WRITE_LOCK = threading.Lock()
_CITATION_RE = re.compile(r"\[(S\d+)\]")


def _load_suite(path: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    cases = payload.get("cases")
    if not isinstance(documents, dict) or not isinstance(cases, list):
        raise TypeError("RAG dataset must contain documents{} and cases[]")
    clean_documents: dict[str, dict[str, str]] = {}
    for key, row in documents.items():
        if not isinstance(row, dict):
            continue
        filename = str(row.get("filename") or "").strip()
        content = str(row.get("content") or "").strip()
        if filename and content:
            clean_documents[str(key)] = {"filename": filename, "content": content}
    clean_cases: list[dict[str, Any]] = []
    for index, row in enumerate(cases, start=1):
        if not isinstance(row, dict):
            continue
        turns = row.get("turns")
        document_ids = [str(item) for item in row.get("documents") or []]
        if not isinstance(turns, list) or not document_ids:
            continue
        if any(item not in clean_documents for item in document_ids):
            raise ValueError(f"case {index} references an unknown document")
        clean_turns = [item for item in turns if isinstance(item, dict) and item.get("question")]
        if not clean_turns:
            continue
        clean_cases.append(
            {
                "case_id": str(row.get("case_id") or f"rag_{index:03d}"),
                "category": str(row.get("category") or "unknown"),
                "documents": document_ids,
                "turns": clean_turns,
            }
        )
    return clean_documents, clean_cases


def _match_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace(",", "").replace("，", "")
    return re.sub(r"[\s`*_:：=|]+", "", text)


def _grade_turn(expected: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = str(result.get("answer") or "")
    normalized = _match_text(answer)
    required = [_match_text(item) for item in expected.get("must_include_all") or []]
    alternatives = [_match_text(item) for item in expected.get("must_include_any") or []]
    forbidden = [_match_text(item) for item in expected.get("must_not_include") or []]
    references = set(_CITATION_RE.findall(answer))
    evidence_ids = {
        str(item.get("source_id"))
        for item in result.get("evidence") or []
        if isinstance(item, dict) and item.get("source_id")
    }
    sources = [item for item in result.get("sources") or [] if isinstance(item, dict)]
    exact_checks = {
        "must_include_all": all(item in normalized for item in required),
        "must_include_any": not alternatives or any(item in normalized for item in alternatives),
        "must_not_include": all(item not in normalized for item in forbidden),
    }
    structural_checks = {
        "nonempty_answer": bool(answer.strip()),
        "document_mode": result.get("retrieval_mode") == "document",
        "local_provider": result.get("search_provider") == "local-markdown",
        "document_sources": bool(sources)
        and all(str(item.get("url") or "").startswith("document://") for item in sources),
        "valid_citations": references <= evidence_ids,
        "no_reasoning_markup": "<think" not in answer.casefold(),
    }
    return {
        "content_pass": all(exact_checks.values()),
        "structural_pass": all(structural_checks.values()),
        "pass": all(exact_checks.values()) and all(structural_checks.values()),
        "checks": {**exact_checks, **structural_checks},
        "references": sorted(references),
    }


def _run_case(
    case: dict[str, Any],
    documents: dict[str, dict[str, str]],
    settings,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    case_id = case["case_id"]
    store_path = output / "stores" / f"{case_id}.sqlite3"
    store = ConversationStore(store_path)
    conversation_id = f"conv_{case_id}_reference"
    client_id = f"client_{case_id}_reference"
    store.ensure(conversation_id, client_id)
    uploaded = []
    for document_id in case["documents"]:
        document = documents[document_id]
        uploaded.append(
            store.add_document(
                conversation_id,
                client_id,
                document["filename"],
                document["content"],
                max_bytes=settings.document_max_bytes,
                chunk_chars=settings.document_chunk_chars,
            )
        )

    turns: list[dict[str, Any]] = []
    for index, expected in enumerate(case["turns"], start=1):
        question = str(expected["question"]).strip()
        history = store.history(
            conversation_id,
            client_id,
            settings.conversation_turns,
            query=question,
        )
        document_query = " ".join(
            [*(turn["question"] for turn in history[-2:]), question]
        )
        passages = store.search_documents(
            conversation_id,
            client_id,
            document_query,
            settings.document_max_chunks,
        )
        turn_started = time.perf_counter()
        result = SearchReaderPipeline(settings).answer(
            question,
            session_id=conversation_id,
            conversation_history=history,
            document_passages=passages,
        )
        payload = result.to_dict()
        store.append_turn(conversation_id, client_id, question, payload)
        grade = _grade_turn(expected, payload)
        turns.append(
            {
                "turn": index,
                "question": question,
                "elapsed_seconds": round(time.perf_counter() - turn_started, 3),
                "answer": payload.get("answer", ""),
                "retrieval_mode": payload.get("retrieval_mode", ""),
                "search_provider": payload.get("search_provider", ""),
                "source_count": len(payload.get("sources") or []),
                "evidence_count": len(payload.get("evidence") or []),
                "warnings": payload.get("warnings") or [],
                "grade": grade,
            }
        )
    return {
        "case_id": case_id,
        "category": case["category"],
        "status": "ok",
        "document_count": len(uploaded),
        "turn_count": len(turns),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "turns": turns,
    }


def _append(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def _build_summary(records: list[dict[str, Any]], cases: list[dict[str, Any]], settings) -> dict[str, Any]:
    turns = [turn for record in records for turn in record.get("turns") or []]
    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"cases": 0, "questions": 0, "passed": 0, "content_passed": 0}
    )
    for case in cases:
        category = str(case.get("category") or "unknown")
        categories[category]["cases"] += 1
        categories[category]["questions"] += len(case.get("turns") or [])
    for record in records:
        category = str(record.get("category") or "unknown")
        for turn in record.get("turns") or []:
            categories[category]["passed"] += int(bool(turn["grade"]["pass"]))
            categories[category]["content_passed"] += int(
                bool(turn["grade"]["content_pass"])
            )
    failed_checks = Counter(
        name
        for turn in turns
        for name, passed in turn["grade"]["checks"].items()
        if not passed
    )
    elapsed = [float(turn.get("elapsed_seconds") or 0) for turn in turns]
    execution_errors = sum(record.get("status") == "error" for record in records)
    if execution_errors:
        failed_checks["execution_error"] += execution_errors
    return {
        "schema_version": "rwkv-search-reader-rag-eval.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": settings.rwkv_model,
        "counts": {
            "cases": len(cases),
            "recorded_cases": len(records),
            "questions": sum(len(case["turns"]) for case in cases),
            "recorded_questions": len(turns),
            "execution_errors": execution_errors,
            "passed": sum(bool(turn["grade"]["pass"]) for turn in turns),
            "structural_passed": sum(
                bool(turn["grade"]["structural_pass"]) for turn in turns
            ),
            "content_passed": sum(
                bool(turn["grade"]["content_pass"]) for turn in turns
            ),
            "empty_answers": sum(not str(turn.get("answer") or "").strip() for turn in turns),
            "reasoning_markup_leaks": sum(
                "<think" in str(turn.get("answer") or "").casefold() for turn in turns
            ),
        },
        "latency_seconds": {
            "mean": round(statistics.fmean(elapsed), 3) if elapsed else 0.0,
            "max": round(max(elapsed), 3) if elapsed else 0.0,
        },
        "failed_checks": dict(failed_checks),
        "categories": dict(sorted(categories.items())),
    }


def _report(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# RWKV SearchReader RAG 参考资料测试报告",
        "",
        f"- 模型：`{summary['model']}`",
        f"- 场景：{counts['recorded_cases']} / {counts['cases']}",
        f"- 问题：{counts['recorded_questions']} / {counts['questions']}",
        f"- 完整通过：{counts['passed']} / {counts['questions']}",
        f"- 内容命中：{counts['content_passed']} / {counts['questions']}",
        f"- 结构通过：{counts['structural_passed']} / {counts['questions']}",
        f"- 执行错误：{counts['execution_errors']}",
        f"- 空回答 / 思考标签泄漏：{counts['empty_answers']} / {counts['reasoning_markup_leaks']}",
        f"- 平均 / 最大耗时：{summary['latency_seconds']['mean']}s / {summary['latency_seconds']['max']}s",
        "",
        "## 分类",
        "",
        "| 分类 | 场景 | 问题 | 完整通过 | 内容命中 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, row in summary["categories"].items():
        lines.append(
            f"| {category} | {row['cases']} | {row['questions']} | "
            f"{row['passed']} | {row['content_passed']} |"
        )
    if summary["failed_checks"]:
        lines.extend(["", "## 未通过检查", ""])
        for name, count in summary["failed_checks"].items():
            lines.append(f"- `{name}`：{count}")
    lines.extend(
        [
            "",
            "> 内容命中使用数据集中预先声明的唯一事实词检查；它比纯结构指标更严格，但仍需人工检查答案表达和推理质量。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local-reference RAG evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = args.dataset.resolve()
    documents, cases = _load_suite(dataset)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.output or PROJECT_ROOT / "runs" / f"rag-reference-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "stores").mkdir(exist_ok=True)
    records_path = output / "results.jsonl"
    settings = replace(get_settings(), trace_enabled=False)
    workers = max(1, min(int(args.workers), 4))
    print(
        json.dumps(
            {
                "dataset": str(dataset),
                "cases": len(cases),
                "questions": sum(len(case["turns"]) for case in cases),
                "workers": workers,
                "output": str(output),
                "model": settings.rwkv_model,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_case, case, documents, settings, output): case
            for case in cases
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - checkpoint every live case
                record = {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                    "turns": [],
                }
            records.append(record)
            _append(records_path, record)
            print(
                f"[{completed}/{len(cases)}] {record['case_id']} {record['status']} "
                f"{record.get('elapsed_seconds', 0)}s",
                flush=True,
            )
    order = {case["case_id"]: index for index, case in enumerate(cases)}
    records.sort(key=lambda item: order.get(item["case_id"], len(order)))
    summary = _build_summary(records, cases, settings)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
