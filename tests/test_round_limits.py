from __future__ import annotations

import pytest
from pydantic import ValidationError

from rwkv_search_reader.config import Settings
from rwkv_search_reader.jobs import JobRequest
from rwkv_search_reader.pipeline import SearchReaderPipeline


def _messages(prior_turns: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for index in range(prior_turns):
        messages.extend(
            [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ]
        )
    messages.append({"role": "user", "content": "current question"})
    return messages


def test_job_request_accepts_100_prior_turns() -> None:
    request = JobRequest(messages=_messages(100))

    assert len(request.messages) == 201


def test_job_request_rejects_more_than_100_prior_turns() -> None:
    with pytest.raises(ValidationError):
        JobRequest(messages=_messages(101))


def test_pipeline_passes_25_turns_to_history_reader(monkeypatch) -> None:
    pipeline = SearchReaderPipeline(Settings(conversation_turns=100, trace_enabled=False))
    observed: dict[str, object] = {}

    def capture_history(question: str, history: list[dict[str, object]]) -> str:
        observed["question"] = question
        observed["history"] = history
        return "ZX-9472"

    monkeypatch.setattr(pipeline.reader, "answer_from_history", capture_history)
    history = [
        {"question": "最早提到的测试编号是 ZX-9472。", "answer": "已记录 ZX-9472。"}
    ]
    history.extend(
        {"question": f"第{index}轮问题。", "answer": f"第{index}轮回答。"}
        for index in range(2, 26)
    )

    result = pipeline.answer(
        "请回顾我们之前聊过的内容，并复述我最早提到的测试编号。",
        conversation_history=history,
    )

    assert result.retrieval_mode == "conversation-history"
    assert result.answer == "ZX-9472"
    assert len(observed["history"]) == 25
    assert "ZX-9472" in observed["history"][0]["question"]
