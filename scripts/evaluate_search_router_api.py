"""Evaluate the public decision API; never creates jobs or changes the model."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    key = os.environ["SEARCH_READER_EVAL_API_KEY"]
    dataset = Path(args.dataset)
    cases = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    assert cases and len({c["id"] for c in cases}) == len(cases)
    assert all(type(c["expected"]) is bool for c in cases)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rows = []
    for index, case in enumerate(cases, 1):
        start = time.monotonic()
        prediction, error, status = None, None, None
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/search/decision",
            data=json.dumps({"messages": case["messages"], "locale": "zh-CN"}).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener.open(request, timeout=30) as response:
                status = response.status
                body = json.load(response)
            if set(body) != {"needs_search"} or type(body["needs_search"]) is not bool:
                raise ValueError("Unexpected decision response schema")
            prediction = body["needs_search"]
        except urllib.error.HTTPError as exc:
            status, error = exc.code, "HTTPError"
        except Exception as exc:
            error = type(exc).__name__
        rows.append({**case, "prediction": prediction, "correct": prediction is case["expected"],
                     "http_status": status, "error": error, "seconds": round(time.monotonic()-start, 3)})
        if index % 10 == 0:
            print(f"{index}/{len(cases)} correct={sum(r['correct'] for r in rows)}", flush=True)
    counts = Counter()
    categories = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        if row["prediction"] is None:
            counts["failed_calls"] += 1
        else:
            counts[("true_" if row["correct"] else "false_") + ("positive" if row["prediction"] else "negative")] += 1
        categories[row["category"]]["total"] += 1
        categories[row["category"]]["correct"] += row["correct"]
    summary = {"total": len(rows), "correct": sum(r["correct"] for r in rows),
               **{k: counts[k] for k in ["true_positive", "true_negative", "false_positive", "false_negative", "failed_calls"]},
               "mean_seconds": round(sum(r["seconds"] for r in rows)/len(rows), 3), "categories": dict(categories)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": str(dataset), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "base_url": args.base_url, "evaluated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary, "cases": rows}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
