"""RWKV Chat integration: classify first, explicitly search only when needed."""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

import requests


def needs_search(messages, locale="zh-CN"):
    """Only classify. HTTP/model failures raise; they never become false."""
    base = os.environ.get("SEARCH_READER_BASE_URL", "http://47.115.88.183:18080").rstrip("/")
    with requests.Session() as session:
        session.headers["Authorization"] = "Bearer " + os.environ["SEARCH_READER_API_KEY"]
        response = session.post(base + "/v1/search/decision",
            json={"messages": messages, "locale": locale}, timeout=25)
        response.raise_for_status()
        decision = response.json().get("needs_search")
        if type(decision) is not bool:
            raise ValueError("Invalid search decision response")
        return decision


def search_answer(messages, locale="zh-CN"):
    """Explicitly submit a search job and wait for the server's search answer."""
    base = os.environ.get("SEARCH_READER_BASE_URL", "http://47.115.88.183:18080").rstrip("/")
    key = os.environ["SEARCH_READER_API_KEY"]
    deadline = time.monotonic() + 900
    # Persist this value with your business request to reuse it across process restarts.
    idem = os.environ.get("SEARCH_READER_IDEMPOTENCY_KEY") or str(uuid.uuid4())
    print("idempotency_key=" + idem, file=sys.stderr)
    payload = {"messages": messages, "locale": locale}
    job = None
    with requests.Session() as session:
        session.headers["Authorization"] = "Bearer " + key
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                if job is None:
                    response = session.post(base + "/v1/jobs", json=payload,
                        headers={"Idempotency-Key": idem}, timeout=min(20, remaining))
                else:
                    response = session.get(base + "/v1/jobs/" + job["job_id"],
                        timeout=min(20, remaining))
            except (requests.Timeout, requests.ConnectionError):
                # Retry POST with the SAME idempotency key; never create a duplicate job.
                time.sleep(min(20, max(0, deadline - time.monotonic())))
                continue
            if response.status_code in (429, 502, 503, 504):
                time.sleep(min(20, max(0, deadline - time.monotonic())))
                continue
            response.raise_for_status()
            current = response.json()
            if job is None:
                print("job_id=" + current["job_id"], file=sys.stderr)
            job = current
            if job["status"] == "done":
                return job["result"]
            if job["status"] == "failed":
                raise RuntimeError(json.dumps(job["error"], ensure_ascii=False))
            if job["status"] != "running":
                raise RuntimeError("Unknown task status: " + str(job["status"]))
            time.sleep(min(20, max(0, deadline - time.monotonic())))
    raise TimeoutError("Stopped waiting after 15 minutes; job may still be queried by its ID")


# Existing callers retain the explicit-job behavior of answer().
answer = search_answer


def answer_if_search_needed(messages, locale="zh-CN"):
    """Return None for RWKV Chat to answer locally; otherwise return the job result."""
    if not needs_search(messages, locale):
        return None
    return search_answer(messages, locale)


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Python 3.13.0 正式发布于哪一天？"
    result = answer_if_search_needed([{"role": "user", "content": question}])
    if result is None:
        print(json.dumps({"needs_search": False}, ensure_ascii=False))
    else:
        print(result["content"])
        print(json.dumps(result["results"], ensure_ascii=False, indent=2))
