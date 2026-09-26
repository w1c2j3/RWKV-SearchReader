"""Compare isolated full pipelines with identical frozen real search material."""
from __future__ import annotations
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from pathlib import Path

from rwkv_search_reader.config import get_settings
from rwkv_search_reader.models import SearchHit
from rwkv_search_reader.pipeline import SearchReaderPipeline
from rwkv_search_reader.retrieval import Retriever
from rwkv_search_reader.search import SearchProvider

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/isolation-comparison-20260911"
QUESTIONS = ["RWKV 创始人是谁？", "RWKV 官方 GitHub 账号是什么？", "RWKV 官方代码仓库地址是什么？", "RWKV 缩写代表什么？", "RWKV-4 论文标题是什么？", "RWKV-7 代号是什么？", "RWKV-7 论文标题是什么？", "RWKV 开源组织名称是什么？"]


def main():
    OUT.mkdir(exist_ok=True)
    records = {}
    for line in (ROOT / "logs/runs-2026-09-11.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("question") in QUESTIONS:
            records[row["question"]] = row
    settings = replace(get_settings(), rwkv_temperature=0, trace_enabled=False)
    lock = threading.Lock()
    active = 0
    peak = 0
    allowed = {f.name for f in fields(SearchHit)}

    class FrozenProvider(SearchProvider):
        name = "frozen-real-material"
        results_are_material = True

        def __init__(self, question):
            super().__init__(settings)
            self.material = [SearchHit(**{k: v for k, v in h.items() if k in allowed}) for h in records[question]["sources"]]

        def search(self, query, max_results):
            # Both modes receive exactly the same ordered real source corpus.
            return copy.deepcopy(self.material[:max_results])

    def run(question, mode):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        start = time.monotonic()
        pipeline = SearchReaderPipeline(settings)
        pipeline.provider.session.close()
        pipeline.provider = FrozenProvider(question)
        pipeline.retriever = Retriever(settings, pipeline.provider)
        try:
            answer = pipeline.answer(question)
            result = answer.to_dict()
            result["model_calls"] = pipeline.reader.call_records()
        except Exception as exc:
            result = {"question": question, "error": type(exc).__name__, "message": str(exc)}
        finally:
            pipeline.provider.session.close()
            with lock:
                active -= 1
        result["wall_seconds"] = time.monotonic() - start
        path = OUT / f"{mode}-{QUESTIONS.index(question):02d}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(mode, question, round(result["wall_seconds"], 1), "warnings", len(result.get("warnings", [])), "error", result.get("error"), flush=True)
        return result

    summary = {}
    for mode, concurrency in [("serial", 1), ("parallel", 8)]:
        peak = 0
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            values = list(pool.map(lambda q: run(q, mode), QUESTIONS))
        summary[mode] = {"peak": peak, "seconds": time.monotonic() - start, "results": values}
        (OUT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print("FINISHED", mode, "peak", peak, flush=True)

if __name__ == "__main__":
    main()
