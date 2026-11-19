from __future__ import annotations

from dataclasses import replace
import json

import pytest

from rwkv_search_reader.config import Settings
from rwkv_search_reader.epoch_compress import (
    DEFAULT_EPOCH_BUDGET_TOKENS,
    estimate_rwkv_tokens,
    fold_epoch_messages,
    slim_sweet_spot_text,
)
from rwkv_search_reader.model_client import (
    OpenAICompatibleClient,
    _ReasoningFilter,
    build_lightning_prompt,
)
from rwkv_search_reader.models import SearchHit, SearchReaderError
from rwkv_search_reader.pi_agent import (
    PROMPT_VERSION,
    WEB_TOOLS_SCHEMA,
    PiLoopGuard,
    compress_pi_epoch,
    parse_pi_xml_calls,
    render_pi_prompt,
    split_continuation,
)
from rwkv_search_reader.pipeline import SearchReaderPipeline


class _JsonResp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.text = self.content.decode("utf-8")
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class _SseResp:
    def __init__(self, lines: list[bytes], *, interrupt_after: int | None = None):
        self.status_code = 200
        self.headers = {"Content-Type": "text/event-stream"}
        self._lines = lines
        self._interrupt_after = interrupt_after
        self.text = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def iter_lines(self, **kwargs):
        del kwargs
        for i, line in enumerate(self._lines):
            if self._interrupt_after is not None and i >= self._interrupt_after:
                raise ConnectionError("simulated client/network abort")
            yield line


def test_rwkv_pi_xml_continuation_v3_renderer_byte_parity() -> None:
    assert PROMPT_VERSION == "rwkv-pi-xml-continuation-v3"
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Find Python 3.13 release date"},
        {
            "role": "assistant",
            "reasoning": "Need official page.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"Python 3.13 release"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "[S1] Released Oct 7, 2024"},
    ]
    rendered_think = render_pi_prompt(messages, WEB_TOOLS_SCHEMA, thinking=True)
    assert rendered_think.startswith("System: You are an agent.\nTools:\n[")
    assert 'output one XML envelope per call: <tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>.' in rendered_think
    assert 'Assistant: <think>Need official page.</think>\n<tool_call>{"name":"web_search","arguments":{"query":"Python 3.13 release"}}</tool_call>\n\n' in rendered_think
    assert "User: <tool_response>\n[S1] Released Oct 7, 2024\n</tool_response>\n\n" in rendered_think
    assert rendered_think.endswith("Assistant: <think")

    rendered_fakethink = render_pi_prompt(messages, WEB_TOOLS_SCHEMA, thinking=False)
    assert rendered_fakethink.endswith("Assistant: <think></think>\n")


def test_pi_xml_parser_and_fenced_example_isolation() -> None:
    raw = (
        ">\nLet's fetch tomllib docs first.</think>\n"
        "```xml\n<tool_call>{\"name\":\"ignored\",\"arguments\":{}}</tool_call>\n```\n"
        '<tool_call>{"name":"web_fetch","arguments":{"url":"https://docs.python.org/3.13/library/tomllib.html"}}</tool_call><EOD>'
    )
    parsed = split_continuation(
        raw,
        thinking=True,
        finish_reason="stop",
        matched_stop="<EOD>",
        tools=WEB_TOOLS_SCHEMA,
        recover=False,
    )
    assert parsed["reasoning"] == "\nLet's fetch tomllib docs first."
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["function"]["name"] == "web_fetch"
    assert "<tool_call>" in parsed["content"]  # fenced example remains ordinary text

    with pytest.raises(ValueError, match="Duplicate JSON key"):
        parse_pi_xml_calls(
            '<tool_call>{"name":"web_search","name":"web_fetch","arguments":{"query":"x"}}</tool_call>',
            WEB_TOOLS_SCHEMA,
            recover=False,
        )


def test_loop_guard_and_epoch_compression() -> None:
    guard = PiLoopGuard(repeat_failure_limit=3, identical_call_limit=2, target_failure_limit=2)
    args = {"query": "Python 3.14"}
    assert guard.check_before_call("web_search", args) is None
    guard.record("web_search", args, "ok1", is_error=False)
    assert guard.check_before_call("web_search", args) is None
    guard.record("web_search", args, "ok2", is_error=False)
    assert "[LoopGuard]" in (guard.check_before_call("web_search", args) or "")

    bad_url = {"url": "https://dead.example/404"}
    guard.record("web_fetch", bad_url, "err", is_error=True)
    guard.record("web_fetch", bad_url, "err", is_error=True)
    assert "[LoopGuard]" in (guard.check_before_call("web_fetch", bad_url) or "")

    long_obs = "Line of fetched content [S1]\n" * 400
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Compare docs"},
        {
            "role": "assistant",
            "reasoning": "step 1",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_fetch", "arguments": '{"url":"https://docs.example/1"}'}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[S1] Pinned doc header"},
        {
            "role": "assistant",
            "reasoning": "step 2",
            "content": "",
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "web_search", "arguments": '{"query":"q2"}'}}],
        },
        {"role": "tool", "tool_call_id": "c2", "content": long_obs},
        {
            "role": "assistant",
            "reasoning": "step 3",
            "content": "",
            "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "web_search", "arguments": '{"query":"q3"}'}}],
        },
        {"role": "tool", "tool_call_id": "c3", "content": "recent result 3 [S3]"},
        {
            "role": "assistant",
            "reasoning": "step 4",
            "content": "",
            "tool_calls": [{"id": "c4", "type": "function", "function": {"name": "web_search", "arguments": '{"query":"q4"}'}}],
        },
        {"role": "tool", "tool_call_id": "c4", "content": "recent result 4 [S4]"},
    ]
    assert DEFAULT_EPOCH_BUDGET_TOKENS == 9600
    folded = fold_epoch_messages(msgs, WEB_TOOLS_SCHEMA, budget_tokens=1200, keep_recent_pairs=2)
    assert folded.folded is True
    assert folded.epoch_count >= 1
    user_merged = next(m["content"] for m in folded.messages if m["role"] == "user")
    assert "[State Ledger (earlier steps folded):" in user_merged
    assert "web_search" in user_merged
    # Pinned unedited web_fetch <= 1800 tokens stays preserved with reasoning stripped!
    assert any("[S1] Pinned doc header" in str(m.get("content", "")) for m in folded.messages)
    assert folded.messages[-1]["content"] == "recent result 4 [S4]"

    slimmed = slim_sweet_spot_text("HEAD_ANCHOR " + ("中文内容测试 " * 800) + " TAIL_ANCHOR", 400)
    assert "HEAD_ANCHOR" in slimmed
    assert "TAIL_ANCHOR" in slimmed
    assert "context sweet spot" in slimmed
    assert estimate_rwkv_tokens(slimmed) < estimate_rwkv_tokens("中文内容测试 " * 800)


def test_openthink_filter_token_count_and_server_stop() -> None:
    assert build_lightning_prompt("", "问题", thinking=True) == "User: 问题\n\nAssistant: <think"
    rf = _ReasoningFilter(initially_hidden=True)
    chunks = [">\n先核对 tomllib 是否支持写操作...", "</thi", "nk>\n结论：tomllib 不支持写入 [S2]。"]
    out = "".join(rf.feed(c) for c in chunks) + rf.finish()
    assert out == "结论：tomllib 不支持写入 [S2]。"

    calls: list[tuple[str, dict]] = []

    def post(url, **kwargs):
        calls.append((url, kwargs.get("json", {})))
        if url.endswith("/tokens/count"):
            return _JsonResp(200, {"tokens": 34000})
        if url.endswith("/server/stop"):
            return _JsonResp(200, {"stopped": True})
        return _SseResp(
            [
                b'data: {"id":"req-abc-123","choices":[{"delta":{"content":">\\nthinking"},"finish_reason":null,"index":0}]}',
                b'data: {"id":"req-abc-123","choices":[{"delta":{"content":"</think>\\nPart 1"},"finish_reason":null,"index":0}]}',
                b"data: [DONE]",
            ],
            interrupt_after=2,
        )

    client = OpenAICompatibleClient(
        base_url="http://lightning-stop.test/v1",
        api_key="k",
        model="rwkv7-g1l",
        timeout_seconds=10,
        context_window=34304,
        adapter_config={"adapter_id": "nora-web-v1", "adapter_scale": 1.0},
    )
    client.session.post = post

    with pytest.raises(SearchReaderError):
        list(
            client.stream_answer(
                user="x" * 4500,
                max_tokens=2048,
                temperature=0.3,
                stage="finalize",
                protocol="lightning",
                thinking=True,
            )
        )

    urls_called = [u for u, _ in calls]
    assert "http://lightning-stop.test/v1/tokens/count" in urls_called
    batch_payload = next(p for u, p in calls if u.endswith("/batch/completions"))
    assert batch_payload["max_tokens"] == 304  # clamped to 34304 - 34000!
    assert batch_payload["adapter_id"] == "nora-web-v1"
    assert batch_payload["temperature"] == 0.96
    assert batch_payload["top_p"] == 0.76
    stop_payload = next(p for u, p in calls if u.endswith("/server/stop"))
    assert stop_payload == {"request_id": "req-abc-123"}


def test_pi_web_agent_end_to_end_via_pipeline(monkeypatch, tmp_path) -> None:
    settings = replace(
        Settings(),
        execution_mode="pi_agent",
        rwkv_thinking_mode="on",
        rwkv_model="rwkv7-g1l",
        deepseek_api_key="sk-test",
        trace_directory=tmp_path,
    )
    pipeline = SearchReaderPipeline(settings)
    pipeline.provider.search_with_context = lambda q, limit, ori_question="": [
        SearchHit(
            title="tomllib — Parse TOML files",
            url="https://docs.python.org/3.13/library/tomllib.html",
            snippet="Parse TOML",
            content="This module does not support writing TOML.",
            score=2.5,
        )
    ]
    pipeline.provider.fetch_url = lambda url: (
        "tomllib — Parse TOML files",
        "# tomllib\n\nThis module does not support writing TOML. tomllib.load(fp) requires a readable binary file object.",
    )

    turn_outputs = iter(
        [
            (
                '>\n先搜索 tomllib 官方文档。</think>\n<tool_call>{"name":"web_search","arguments":{"query":"python 3.13 tomllib"}}</tool_call><EOD>',
                "stop",
                "<EOD>",
                120,
            ),
            (
                '>\n再读取官方文档确认写入能力与 load 参数。</think>\n<tool_call>{"name":"web_fetch","arguments":{"url":"https://docs.python.org/3.13/library/tomllib.html"}}</tool_call><EOD>',
                "stop",
                "<EOD>",
                380,
            ),
            (
                ">\n文档明确写明不支持写入 TOML，load 需要二进制文件对象。</think>\n`tomllib` 仅支持解析读取 TOML，不支持写入 TOML，且 `tomllib.load(fp)` 要求传入可读二进制文件对象 [S1]。<EOD>",
                "stop",
                "<EOD>",
                640,
            ),
        ]
    )
    monkeypatch.setattr(
        OpenAICompatibleClient,
        "raw_continuation",
        lambda self, *a, **kw: next(turn_outputs),
    )

    result = pipeline.answer("Python 3.13 的 tomllib 支持写入吗？load() 对文件对象有什么要求？")
    assert "不支持写入" in result.answer
    assert "[S1]" in result.answer
    assert result.reader_mode == "rwkv-pi-xml-continuation-v3"
    assert result.queries == ["python 3.13 tomllib"]
    assert len(result.sources) == 1
    traj_files = list(tmp_path.glob("pi-trajectories-*.jsonl"))
    assert len(traj_files) == 1
    saved = json.loads(traj_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert saved["schema_version"] == "rwkv-pi-xml-continuation-v3"
