from __future__ import annotations

import json

from rwkv_search_reader.model_client import OpenAICompatibleClient


class FakeResponse:
    status_code = 200
    content = json.dumps({"choices": [{"text": "answer"}]}).encode()
    text = content.decode()


def test_completion_payload_never_contains_tools() -> None:
    client = OpenAICompatibleClient(
        base_url="http://model.test/v1",
        api_key="secret",
        model="rwkv",
        timeout_seconds=10,
    )
    captured = {}

    def post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return FakeResponse()

    client.session.post = post
    answer = client.completion(
        system="read evidence",
        user="question",
        max_tokens=128,
        temperature=0.1,
        stage="read",
    )

    assert answer == "answer"
    assert captured["url"].endswith("/v1/completions")
    assert "tools" not in captured["json"]
    assert "tool_choice" not in captured["json"]
    assert captured["json"]["prompt"].endswith("Assistant:")
