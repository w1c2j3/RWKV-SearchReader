from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

from .config import PROJECT_ROOT, get_settings
from .conversations import ConversationStore
from .documents import (
    PreparedDocument,
    full_document_read_requested,
    limit_full_document_passages,
    prepare_document,
    rank_markdown_passages,
)
from .pipeline import SearchReaderPipeline
from .rag_evaluation import _grade_turn, _match_text

DEFAULT_DATASET = PROJECT_ROOT / "evals" / "pdf_reference_10_20260902.json"
_WRITE_LOCK = threading.Lock()


def _load_suite(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    cases = payload.get("cases")
    if not isinstance(documents, dict) or not isinstance(cases, list):
        raise TypeError("PDF dataset must contain documents{} and cases[]")
    clean_documents: dict[str, dict[str, Any]] = {}
    for key, row in documents.items():
        if not isinstance(row, dict):
            continue
        filename = str(row.get("filename") or "").strip()
        url = str(row.get("url") or "").strip()
        digest = str(row.get("sha256") or "").strip().casefold()
        if filename.casefold().endswith(".pdf") and url and len(digest) == 64:
            clean_documents[str(key)] = dict(row)
    clean_cases: list[dict[str, Any]] = []
    for index, row in enumerate(cases, start=1):
        if not isinstance(row, dict):
            continue
        document_ids = [str(item) for item in row.get("documents") or []]
        turns = [
            item
            for item in row.get("turns") or []
            if isinstance(item, dict) and item.get("question")
        ]
        if not document_ids or not turns:
            continue
        if any(item not in clean_documents for item in document_ids):
            raise ValueError(f"case {index} references an unknown PDF")
        clean_cases.append(
            {
                "case_id": str(row.get("case_id") or f"pdf_{index:03d}"),
                "category": str(row.get("category") or "unknown"),
                "documents": document_ids,
                "turns": turns,
            }
        )
    return clean_documents, clean_cases


def _download_pdf(document_id: str, row: dict[str, Any], cache: Path) -> bytes:
    target = cache / f"{document_id}.source.pdf"
    expected = str(row["sha256"]).casefold()
    if target.exists():
        content = target.read_bytes()
        if hashlib.sha256(content).hexdigest() == expected:
            return content
    response = requests.get(
        str(row["url"]),
        timeout=120,
        headers={"User-Agent": "RWKV-SearchReader-PDF-Evaluation/1.0"},
    )
    response.raise_for_status()
    content = response.content
    if not content.lstrip().startswith(b"%PDF-"):
        raise ValueError(f"{document_id} did not download a PDF")
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(f"{document_id} sha256 mismatch: {actual}")
    target.write_bytes(content)
    return content


def _transform_pdf(content: bytes, transform: dict[str, Any]) -> bytes:
    if not transform:
        return content
    import pypdfium2 as pdfium

    source = pdfium.PdfDocument(BytesIO(content))
    if transform.get("scan_first_page"):
        image = source[0].render(scale=2.4).to_pil().convert("RGB")
        output = BytesIO()
        image.save(output, format="PDF", resolution=172.8)
        return output.getvalue()
    repeats = int(transform.get("repeat_pages") or 0)
    if repeats > 1:
        output_pdf = pdfium.PdfDocument.new()
        for _ in range(repeats):
            output_pdf.import_pages(source)
        output = BytesIO()
        output_pdf.save(output)
        return output.getvalue()
    return content


def _conversion_grade(row: dict[str, Any], markdown: str) -> dict[str, Any]:
    normalized = _match_text(markdown)
    required = [
        _match_text(item) for item in row.get("conversion_must_include_all") or []
    ]
    alternatives = [
        _match_text(item) for item in row.get("conversion_must_include_any") or []
    ]
    checks = {
        "nonempty_markdown": bool(markdown.strip()),
        "must_include_all": all(item in normalized for item in required),
        "must_include_any": not alternatives
        or any(item in normalized for item in alternatives),
    }
    return {"pass": all(checks.values()), "checks": checks}


def _prepare_pdf(
    document_id: str,
    row: dict[str, Any],
    cache: Path,
    settings,
) -> tuple[PreparedDocument, dict[str, Any]]:
    started = time.perf_counter()
    source = _download_pdf(document_id, row, cache)
    transformed = _transform_pdf(source, dict(row.get("transform") or {}))
    prepared = prepare_document(
        str(row["filename"]),
        content_base64=base64.b64encode(transformed).decode("ascii"),
        max_markdown_bytes=settings.document_max_bytes,
        max_pdf_bytes=settings.document_pdf_max_bytes,
        max_pdf_pages=settings.document_pdf_max_pages,
    )
    grade = _conversion_grade(row, prepared.markdown)
    return prepared, {
        "document_id": document_id,
        "filename": prepared.filename,
        "status": "ok",
        "source_bytes": len(source),
        "uploaded_bytes": len(transformed),
        "markdown_chars": len(prepared.markdown),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "grade": grade,
    }


def _run_case(
    case: dict[str, Any],
    documents: dict[str, PreparedDocument],
    settings,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    case_id = case["case_id"]
    store = ConversationStore(output / "stores" / f"{case_id}.sqlite3")
    conversation_id = f"conv_{case_id}_pdf"
    client_id = f"client_{case_id}_pdf"
    store.ensure(conversation_id, client_id)
    for document_id in case["documents"]:
        store.add_prepared_document(
            conversation_id,
            client_id,
            documents[document_id],
            chunk_chars=settings.document_chunk_chars,
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
        all_passages = store.document_passages(conversation_id, client_id)
        passages = (
            limit_full_document_passages(
                all_passages, settings.document_full_read_max_chunks
            )
            if full_document_read_requested(question)
            else rank_markdown_passages(
                document_query, all_passages, settings.document_max_chunks
            )
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
        turns.append(
            {
                "turn": index,
                "question": question,
                "answer": payload.get("answer", ""),
                "elapsed_seconds": round(time.perf_counter() - turn_started, 3),
                "source_count": len(payload.get("sources") or []),
                "warnings": payload.get("warnings") or [],
                "grade": _grade_turn(expected, payload),
            }
        )
    return {
        "case_id": case_id,
        "category": case["category"],
        "status": "ok",
        "document_count": len(case["documents"]),
        "turn_count": len(turns),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "turns": turns,
    }


def _append(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def _summary(
    conversions: list[dict[str, Any]],
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    settings,
) -> dict[str, Any]:
    turns = [turn for record in records for turn in record.get("turns") or []]
    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"cases": 0, "questions": 0, "passed": 0}
    )
    for case in cases:
        category = case["category"]
        categories[category]["cases"] += 1
        categories[category]["questions"] += len(case["turns"])
    for record in records:
        for turn in record.get("turns") or []:
            categories[record["category"]]["passed"] += int(bool(turn["grade"]["pass"]))
    failed_checks = Counter(
        name
        for turn in turns
        for name, passed in turn["grade"]["checks"].items()
        if not passed
    )
    elapsed = [float(turn.get("elapsed_seconds") or 0) for turn in turns]
    return {
        "schema_version": "rwkv-search-reader-pdf-eval.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": settings.rwkv_model,
        "counts": {
            "documents": len(conversions),
            "conversion_passed": sum(
                item.get("status") == "ok" and item.get("grade", {}).get("pass")
                for item in conversions
            ),
            "conversion_errors": sum(
                item.get("status") == "error" for item in conversions
            ),
            "cases": len(cases),
            "recorded_cases": len(records),
            "questions": sum(len(case["turns"]) for case in cases),
            "recorded_questions": len(turns),
            "passed": sum(bool(turn["grade"]["pass"]) for turn in turns),
            "content_passed": sum(
                bool(turn["grade"]["content_pass"]) for turn in turns
            ),
            "structural_passed": sum(
                bool(turn["grade"]["structural_pass"]) for turn in turns
            ),
            "execution_errors": sum(
                record.get("status") == "error" for record in records
            ),
        },
        "latency_seconds": {
            "mean": round(statistics.fmean(elapsed), 3) if elapsed else 0.0,
            "max": round(max(elapsed), 3) if elapsed else 0.0,
        },
        "failed_checks": dict(failed_checks),
        "categories": dict(sorted(categories.items())),
        "conversions": conversions,
    }


def _report(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    counts = summary["counts"]
    lines = [
        "# RWKV SearchReader PDF 质量测试报告",
        "",
        f"- 模型：`{summary['model']}`",
        f"- PDF 转换：{counts['conversion_passed']} / {counts['documents']}",
        f"- 问答通过：{counts['passed']} / {counts['questions']}",
        f"- 内容命中：{counts['content_passed']} / {counts['questions']}",
        f"- 结构通过：{counts['structural_passed']} / {counts['questions']}",
        f"- 执行错误：{counts['execution_errors']}",
        f"- 平均 / 最大问答耗时：{summary['latency_seconds']['mean']}s / {summary['latency_seconds']['max']}s",
        "",
        "## 文档转换",
        "",
        "| 文档 | 结果 | 原始/上传大小 | Markdown 字符 | 耗时 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in summary["conversions"]:
        passed = item.get("status") == "ok" and item.get("grade", {}).get("pass")
        lines.append(
            f"| {item['document_id']} | {'通过' if passed else '未通过'} | "
            f"{item.get('source_bytes', 0)} / {item.get('uploaded_bytes', 0)} | "
            f"{item.get('markdown_chars', 0)} | {item.get('elapsed_seconds', 0)}s |"
        )
    lines.extend(["", "## 问答", ""])
    for record in records:
        lines.append(f"### {record['case_id']}")
        lines.append("")
        if record.get("status") != "ok":
            lines.append(f"- 执行错误：{record.get('error', 'unknown')}")
            lines.append("")
            continue
        for turn in record.get("turns") or []:
            lines.extend(
                [
                    f"- 问题：{turn['question']}",
                    f"- 结果：{'通过' if turn['grade']['pass'] else '未通过'}",
                    f"- 回答：{turn['answer']}",
                    "",
                ]
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-PDF RAG quality evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="run only the named case; may be repeated",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = args.dataset.resolve()
    document_rows, cases = _load_suite(dataset)
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case["case_id"] in selected]
        unknown = selected - {case["case_id"] for case in cases}
        if unknown:
            raise ValueError(f"unknown PDF case: {', '.join(sorted(unknown))}")
        required_documents = {
            document_id for case in cases for document_id in case["documents"]
        }
        document_rows = {
            key: row for key, row in document_rows.items() if key in required_documents
        }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.output or PROJECT_ROOT / "runs" / f"pdf-reference-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "downloads").mkdir(exist_ok=True)
    (output / "stores").mkdir(exist_ok=True)
    settings = replace(get_settings(), trace_enabled=False)

    prepared: dict[str, PreparedDocument] = {}
    conversions: list[dict[str, Any]] = []
    for index, (document_id, row) in enumerate(document_rows.items(), start=1):
        try:
            document, record = _prepare_pdf(
                document_id, row, output / "downloads", settings
            )
            prepared[document_id] = document
        except Exception as exc:  # noqa: BLE001 - retain every conversion failure
            record = {
                "document_id": document_id,
                "filename": row["filename"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
                "grade": {"pass": False, "checks": {}},
            }
        conversions.append(record)
        print(
            f"[PDF {index}/{len(document_rows)}] {document_id} {record['status']} "
            f"{record.get('elapsed_seconds', 0)}s",
            flush=True,
        )

    runnable = [
        case
        for case in cases
        if all(document_id in prepared for document_id in case["documents"])
    ]
    records: list[dict[str, Any]] = []
    records_path = output / "results.jsonl"
    workers = max(1, min(int(args.workers), 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_case, case, prepared, settings, output): case
            for case in runnable
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
                f"[QA {completed}/{len(runnable)}] {record['case_id']} "
                f"{record['status']} {record.get('elapsed_seconds', 0)}s",
                flush=True,
            )
    order = {case["case_id"]: index for index, case in enumerate(cases)}
    records.sort(key=lambda item: order.get(item["case_id"], len(order)))
    summary = _summary(conversions, records, cases, settings)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(_report(summary, records), encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
