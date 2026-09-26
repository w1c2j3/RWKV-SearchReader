"""Evaluate the dedicated classifier without searching or creating jobs."""
import json
import hashlib
from collections import Counter, defaultdict
import os
from pathlib import Path
import time
from rwkv_search_reader.routing import decide_search

root = Path(__file__).resolve().parents[1]
rows = []
dataset = root / os.environ.get("ROUTER_EVAL_DATASET", "evals/search_router.jsonl")
for line in dataset.read_text().splitlines():
    case = json.loads(line)
    if "expected" not in case:
        case["expected"] = case["needs_search"]
    assert type(case["expected"]) is bool
    started = time.monotonic()
    try:
        prediction = decide_search(case["messages"])
        error = None
        error_message = None
    except Exception as exc:
        prediction = None
        error = type(exc).__name__
        error_message = str(exc)[:500]
    row = {**case, "prediction": prediction, "correct": prediction is case["expected"],
           "error": error, "error_message": error_message, "seconds": round(time.monotonic()-started, 3)}
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
output = root / "runs" / (os.environ.get("ROUTER_EVAL_NAME", "search-router-evaluation") + ".json")
output.parent.mkdir(exist_ok=True)
summary = {"total": len(rows), "correct": sum(r["correct"] for r in rows),
           "format_errors": sum("Invalid decision output" in (r["error_message"] or "") for r in rows),
           "failed_calls": sum(r["prediction"] is None for r in rows),
           "mean_seconds": round(sum(r["seconds"] for r in rows)/len(rows), 3)}
counts = Counter()
categories = defaultdict(lambda: {"total": 0, "correct": 0})
for row in rows:
    if row["prediction"] is not None:
        counts[("true_" if row["correct"] else "false_") + ("positive" if row["prediction"] else "negative")] += 1
    category = row.get("category", row.get("group", "ungrouped"))
    categories[category]["total"] += 1
    categories[category]["correct"] += row["correct"]
summary.update({k: counts[k] for k in ["true_positive", "true_negative", "false_positive", "false_negative"]})
summary["categories"] = dict(categories)
output.write_text(json.dumps({"dataset": str(dataset), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    "model": os.environ.get("SEARCH_READER_ROUTER_MODEL"),
    "evaluated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "summary": summary, "cases": rows}, ensure_ascii=False, indent=2))
print(json.dumps(summary), flush=True)
