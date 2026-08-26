from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, get_settings
from .models import SearchReaderError
from .pipeline import SearchReaderPipeline

DEFAULT_DATASET = Path(
    "/home/chase/GitHub/RWKV-ECRA/data/evaluation/"
    "retrieval_production_live_100_v1_20260812.json"
)
_WRITE_LOCK = threading.Lock()


def _load_cases(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError("dataset must be a list or contain cases[]")
    cases: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        question = str(row.get("query") or row.get("question") or "").strip()
        if not question:
            continue
        cases.append(
            {
                "case_id": str(row.get("case_id") or row.get("question_id") or index),
                "category": str(row.get("category") or "unknown"),
                "temporal_scope": str(row.get("temporal_scope") or ""),
                "question": question,
            }
        )
    return cases


def _existing_records(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return latest
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("case_id"):
            latest[str(row["case_id"])] = row
    return latest


def _append(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _run_case(case: dict[str, str], settings, attempt: int) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.perf_counter()
    base = {
        **case,
        "attempt": attempt,
        "started_at": started_at,
    }
    try:
        result = SearchReaderPipeline(settings).answer(case["question"])
        return {
            **base,
            "status": "ok",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "result": result.to_dict(),
        }
    except SearchReaderError as exc:
        return {
            **base,
            "status": "error",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error_stage": exc.stage,
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }
    except Exception as exc:  # noqa: BLE001 - each benchmark case must be checkpointed
        return {
            **base,
            "status": "error",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error_stage": "unexpected",
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return round(ordered[index], 3)


def build_summary(
    *,
    records: dict[str, dict[str, Any]],
    cases: list[dict[str, str]],
    dataset: Path,
    settings,
) -> dict[str, Any]:
    ordered = [records[case["case_id"]] for case in cases if case["case_id"] in records]
    successful = [row for row in ordered if row.get("status") == "ok"]
    failed = [row for row in ordered if row.get("status") != "ok"]
    elapsed = [float(row.get("elapsed_seconds") or 0.0) for row in successful]
    degraded = [row for row in successful if (row.get("result") or {}).get("degraded")]
    answers = [str((row.get("result") or {}).get("answer") or "") for row in successful]
    answer_lengths = [len(answer) for answer in answers]
    cited: list[dict[str, Any]] = []
    validly_cited: list[dict[str, Any]] = []
    invalidly_cited: list[dict[str, Any]] = []
    for row in successful:
        result = row.get("result") or {}
        references = set(re.findall(r"\[(S\d+)\]", str(result.get("answer") or "")))
        known_sources = {
            str(item.get("source_id"))
            for item in result.get("evidence") or []
            if isinstance(item, dict) and item.get("source_id")
        }
        if references:
            cited.append(row)
            if references <= known_sources:
                validly_cited.append(row)
            else:
                invalidly_cited.append(row)
    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "ok": 0, "error": 0, "degraded": 0, "cited": 0}
    )
    for row in ordered:
        category = str(row.get("category") or "unknown")
        categories[category]["total"] += 1
        if row.get("status") == "ok":
            categories[category]["ok"] += 1
            result = row.get("result") or {}
            categories[category]["degraded"] += int(bool(result.get("degraded")))
            categories[category]["cited"] += int(
                "[S" in str(result.get("answer") or "")
            )
        else:
            categories[category]["error"] += 1
    return {
        "schema_version": "rwkv-search-reader-eval.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "model": {
            "base_url": settings.rwkv_base_url,
            "model": settings.rwkv_model,
            "protocol": settings.rwkv_protocol,
            "max_tokens": settings.rwkv_max_tokens,
            "temperature": settings.rwkv_temperature,
        },
        "search_provider": settings.search_provider,
        "reader_mode_configured": "rwkv-parallel-funnel-final-summary",
        "counts": {
            "dataset": len(cases),
            "recorded": len(ordered),
            "ok": len(successful),
            "error": len(failed),
            "degraded": len(degraded),
            "with_source_citations": len(cited),
            "with_only_valid_source_citations": len(validly_cited),
            "with_invalid_source_citations": len(invalidly_cited),
            "empty_answers": sum(not answer.strip() for answer in answers),
            "reasoning_markup_leaks": sum(
                "<think" in answer.casefold() or "</think" in answer.casefold()
                for answer in answers
            ),
            "answers_over_1000_chars": sum(len(answer) > 1000 for answer in answers),
        },
        "rates": {
            "completion": round(len(successful) / max(1, len(cases)), 4),
            "degraded_among_ok": round(len(degraded) / max(1, len(successful)), 4),
            "citation_among_ok": round(len(cited) / max(1, len(successful)), 4),
        },
        "latency_seconds": {
            "mean": round(statistics.fmean(elapsed), 3) if elapsed else 0.0,
            "p50": _percentile(elapsed, 0.5),
            "p95": _percentile(elapsed, 0.95),
            "max": round(max(elapsed), 3) if elapsed else 0.0,
        },
        "answer_chars": {
            "mean": round(statistics.fmean(answer_lengths), 1)
            if answer_lengths
            else 0.0,
            "p50": _percentile([float(value) for value in answer_lengths], 0.5),
            "p95": _percentile([float(value) for value in answer_lengths], 0.95),
            "max": max(answer_lengths, default=0),
        },
        "error_stages": dict(
            Counter(str(row.get("error_stage") or "unknown") for row in failed)
        ),
        "categories": dict(sorted(categories.items())),
        "quality_scope": (
            "Structural run metrics only. The natural-100 dataset contains no gold answers; "
            "factual correctness requires separate human or reference-snapshot review."
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    latency = summary["latency_seconds"]
    lines = [
        "# RWKV SearchReader normal-100 测试报告",
        "",
        f"- 模型：`{summary['model']['model']}`",
        f"- 检索：`{summary['search_provider']}`（固定先检索）",
        f"- 完成：{counts['ok']} / {counts['dataset']}",
        f"- 失败：{counts['error']}",
        f"- 含 [S#] 引用：{counts['with_source_citations']} / {max(1, counts['ok'])}",
        f"- 引用编号全部有效：{counts['with_only_valid_source_citations']} / {max(1, counts['ok'])}",
        f"- 含无效引用编号：{counts['with_invalid_source_citations']}",
        f"- 降级回答：{counts['degraded']} / {max(1, counts['ok'])}",
        f"- 空回答 / 思考标签泄漏：{counts['empty_answers']} / {counts['reasoning_markup_leaks']}",
        f"- 超过 1000 字符的回答：{counts['answers_over_1000_chars']}",
        f"- 平均 / P50 / P95：{latency['mean']}s / {latency['p50']}s / {latency['p95']}s",
        "",
        "> 数据集不含标准答案；以上是运行与结构指标，不代表事实正确率。",
        "",
        "## 分类统计",
        "",
        "| 分类 | 总数 | 成功 | 失败 | 含引用 | 降级 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, row in summary["categories"].items():
        lines.append(
            f"| {category} | {row['total']} | {row['ok']} | {row['error']} | "
            f"{row['cited']} | {row['degraded']} |"
        )
    if summary["error_stages"]:
        lines.extend(["", "## 失败阶段", ""])
        for stage, count in summary["error_stages"].items():
            lines.append(f"- `{stage}`：{count}")
    lines.extend(
        [
            "",
            "## 配置",
            "",
            f"- Base URL：`{summary['model']['base_url']}`",
            f"- 协议：`{summary['model']['protocol']}`",
            f"- 最大输出：{summary['model']['max_tokens']} tokens",
            f"- Temperature：{summary['model']['temperature']}",
            f"- 阅读模式：`{summary['reader_mode_configured']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable SearchReader dataset evaluation"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = args.dataset.resolve()
    cases = _load_cases(dataset)
    if args.limit > 0:
        cases = cases[: args.limit]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.output or PROJECT_ROOT / "runs" / f"normal100-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "results.jsonl"
    existing = _existing_records(records_path)
    pending = [
        case
        for case in cases
        if case["case_id"] not in existing
        or (args.retry_errors and existing[case["case_id"]].get("status") != "ok")
    ]
    settings = replace(get_settings(), trace_enabled=False)
    workers = max(1, min(int(args.workers), 16))
    print(
        json.dumps(
            {
                "dataset": str(dataset),
                "cases": len(cases),
                "already_recorded": len(cases) - len(pending),
                "pending": len(pending),
                "workers": workers,
                "output": str(output),
                "model": settings.rwkv_model,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_case,
                case,
                settings,
                int(existing.get(case["case_id"], {}).get("attempt") or 0) + 1,
            ): case
            for case in pending
        }
        for future in as_completed(futures):
            record = future.result()
            _append(records_path, record)
            existing[record["case_id"]] = record
            completed += 1
            print(
                f"[{completed}/{len(pending)}] {record['case_id']} "
                f"{record['status']} {record['elapsed_seconds']}s",
                flush=True,
            )

    summary = build_summary(
        records=existing,
        cases=cases,
        dataset=dataset,
        settings=settings,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
