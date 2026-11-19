"""Native rwkv-pi-xml-continuation-v3 web search & fetch agent loop.

Implements byte-identical prompt rendering, strict XML tool-call envelope parsing,
loop guards (repeated failure / identical call / target failure), epoch token-budget
compression, and SFT/NoRA trajectory export compatible with `rwkv-pi`.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

from .config import Settings
from .epoch_compress import (
    DEFAULT_EPOCH_BUDGET_TOKENS,
    estimate_rwkv_tokens,
    fold_epoch_messages,
    slim_sweet_spot_text,
)
from .model_client import OpenAICompatibleClient
from .models import Evidence, PipelineResult, SearchHit, SearchReaderError
from .retrieval import normalize_url
from .search import SearchProvider

PROMPT_VERSION = "rwkv-pi-xml-continuation-v3"
STOP_TEXTS: tuple[str, ...] = ("<EOD>", "✿", "\nUser:", "\n### User")
MAX_XML_BYTES = 1024 * 1024
MAX_CALLS_PER_TURN = 16

WEB_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current facts, official announcements, or documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query (<= 128 chars).",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": "Maximum search results to return (default 6).",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch full Markdown content from a specific http/https URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Exact http/https URL to fetch and read.",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optional keyword or section name to focus on for long pages.",
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant: {value}")


_STRICT_DECODER = json.JSONDecoder(
    object_pairs_hook=_unique_object,
    parse_constant=_reject_constant,
)


def strict_json_loads(source: str) -> Any:
    return json.loads(
        source,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        for part in content
    ):
        return "\n".join(part["text"] for part in content)
    raise ValueError("RWKV continuation accepts text messages only.")


def render_pi_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    thinking: bool = True,
    generation: bool = True,
) -> str:
    """Byte-exact `rwkv-pi-xml-continuation-v3` prompt renderer."""

    if not messages:
        raise ValueError("Empty conversation.")
    tool_list = list(tools or [])
    system = "\n\n".join(
        _message_text(m.get("content")).strip()
        for m in messages
        if m.get("role") == "system" and _message_text(m.get("content")).strip()
    )
    output = ""
    if system or tool_list:
        output += f"System: {system}"
        if tool_list:
            sep = "\n" if system else ""
            tools_json = json.dumps(tool_list, ensure_ascii=False, separators=(",", ":"))
            output += (
                f"{sep}Tools:\n{tools_json}\n"
                'When using tools, output one XML envelope per call: '
                '<tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>. '
                "Use only the declared tool names and arguments. "
                "Tool results arrive inside <tool_response>. Otherwise answer normally."
            )
        output += "\n\n"
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            output += f"User: {_message_text(m.get('content')).strip()}\n\n"
        elif role == "tool":
            output += f"User: <tool_response>\n{_message_text(m.get('content'))}\n</tool_response>\n\n"
        elif role == "assistant":
            reasoning = _message_text(m.get("reasoning_content") or m.get("reasoning"))
            if re.search(r"</?think>", reasoning):
                raise ValueError("Ambiguous reasoning delimiters.")
            parts = [p for p in [_message_text(m.get("content")).strip()] if p]
            for call in m.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call.get("function"), dict) else call
                raw_args = fn.get("arguments")
                args = strict_json_loads(raw_args) if isinstance(raw_args, str) else raw_args
                envelope_json = json.dumps(
                    {"name": fn["name"], "arguments": args},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                parts.append(f"<tool_call>{envelope_json}</tool_call>")
            joined = "\n".join(parts)
            output += f"Assistant: <think>{reasoning}</think>\n{joined}\n\n"
        else:
            raise ValueError(f"Unsupported role: {role}")
    if generation:
        output += "Assistant: <think" if thinking else "Assistant: <think></think>\n"
    return output


def call_start(text: str, *, partial: bool = False) -> int | None:
    """Locate an XML tool-call envelope at a line boundary outside Markdown code fences."""

    offset = 0
    fence: str | None = None
    tags = ("<tool_call>", "<tool_calls>")
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" \t")
        marker = next((m for m in ("```", "~~~") if stripped.startswith(m)), None)
        if marker:
            fence = None if fence == marker else (marker if fence is None else fence)
        elif fence is None and (
            stripped.startswith(tags)
            or (partial and stripped and any(tag.startswith(stripped) for tag in tags))
        ):
            return offset + len(line) - len(stripped)
        offset += len(line)
    return None


def _validate_tool_args(schema: dict[str, Any], args: dict[str, Any]) -> None:
    required = schema.get("required") or []
    for req in required:
        if req not in args:
            raise ValueError(f"Missing required argument: {req}")
    props = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        unknown = set(args) - set(props)
        if unknown:
            raise ValueError(f"Unexpected arguments: {sorted(unknown)}")
    for key, val in args.items():
        spec = props.get(key)
        if not isinstance(spec, dict):
            continue
        expected_type = spec.get("type")
        if expected_type == "string" and not isinstance(val, str):
            raise ValueError(f"Argument {key} must be string")
        if expected_type == "integer" and (isinstance(val, bool) or not isinstance(val, int)):
            raise ValueError(f"Argument {key} must be integer")


def parse_pi_xml_calls(
    source: str,
    tools: list[dict[str, Any]],
    *,
    recover: bool = True,
) -> list[dict[str, Any]]:
    """Parse `<tool_call>` / `<tool_calls>` envelopes following rwkv-pi `xml.ts`."""

    if len(source.encode("utf-8")) > MAX_XML_BYTES:
        raise ValueError("XML calls exceed 1 MiB.")
    allowed = {
        t["function"]["name"]: t["function"].get("parameters", {})
        for t in tools
        if isinstance(t.get("function"), dict) and "name" in t["function"]
    }
    calls: list[dict[str, Any]] = []
    cursor = 0
    length = len(source)
    while cursor < length:
        while cursor < length and source[cursor].isspace():
            cursor += 1
        if cursor == length:
            break
        opening = next(
            (tag for tag in ("<tool_call>", "<tool_calls>") if source.startswith(tag, cursor)),
            None,
        )
        if opening is None:
            if recover and calls:
                break
            raise ValueError("Expected only XML tool-call envelopes.")
        cursor += len(opening)
        while cursor < length and source[cursor].isspace():
            cursor += 1
        closing = "</tool_calls>" if opening == "<tool_calls>" else "</tool_call>"
        try:
            value, end = _STRICT_DECODER.raw_decode(source, cursor)
        except (json.JSONDecodeError, ValueError) as exc:
            if not recover:
                raise
            err_detail = getattr(exc, "msg", str(exc))
            close_idx = source.find(closing, cursor)
            raw_chunk = source[cursor : close_idx if close_idx >= 0 else length].strip()
            cursor = (close_idx + len(closing)) if close_idx >= 0 else length
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": "__invalid_json__", "arguments": raw_chunk},
                    "error": f"Invalid JSON in tool_call: {err_detail}",
                }
            )
            continue
        cursor = end
        while cursor < length and source[cursor].isspace():
            cursor += 1
        alt_closing = "</tool_call>" if closing == "</tool_calls>" else "</tool_calls>"
        if source.startswith(closing, cursor):
            cursor += len(closing)
        elif recover and source.startswith(alt_closing, cursor):
            cursor += len(alt_closing)
        elif recover:
            next_close = source.find(closing, cursor)
            cursor = (next_close + len(closing)) if next_close >= 0 else length
        else:
            raise ValueError("Missing XML closing tag.")
        batch = value if opening == "<tool_calls>" else [value]
        if not isinstance(batch, list) or not batch or len(calls) + len(batch) > MAX_CALLS_PER_TURN:
            raise ValueError("Expected 1-16 tool calls.")
        for item in batch:
            if isinstance(item, dict) and isinstance(item.get("function"), dict):
                item = item["function"]
            if (
                recover
                and isinstance(item, dict)
                and "name" in item
                and "arguments" not in item
            ):
                for alias in ("input", "parameters", "args"):
                    if alias in item:
                        item = {**item, "arguments": item[alias]}
                        break
            if not isinstance(item, dict) or "name" not in item or "arguments" not in item:
                if not recover:
                    raise ValueError("Expected name and arguments.")
                calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {"name": "__malformed__", "arguments": "{}"},
                        "error": "Tool call must contain 'name' and 'arguments'.",
                    }
                )
                continue
            name = str(item["name"])
            raw_args = item["arguments"]
            if isinstance(raw_args, str):
                raw_args = strict_json_loads(raw_args)
            if name not in allowed:
                if not recover:
                    raise ValueError(f"Tool {name} is not allowed.")
                calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(raw_args, ensure_ascii=False)},
                        "error": f"Tool '{name}' not found. Available tools: {sorted(allowed)}",
                    }
                )
                continue
            if not isinstance(raw_args, dict):
                if not recover:
                    raise ValueError("Arguments must be an object.")
                calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                        "error": f"Arguments for '{name}' must be a JSON object.",
                    }
                )
                continue
            err_msg = ""
            try:
                _validate_tool_args(allowed[name], raw_args)
            except ValueError as exc:
                if not recover:
                    raise
                err_msg = str(exc)
            entry: dict[str, Any] = {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(raw_args, ensure_ascii=False, separators=(",", ":")),
                },
            }
            if err_msg:
                entry["error"] = err_msg
            calls.append(entry)
    return calls


def split_continuation(
    raw: str,
    *,
    thinking: bool,
    finish_reason: str,
    matched_stop: str | None,
    tools: list[dict[str, Any]],
    recover: bool = True,
) -> dict[str, Any]:
    """Split raw Lightning continuation into reasoning, visible text, and XML tool calls."""

    body = raw
    if matched_stop and body.endswith(matched_stop):
        body = body[: -len(matched_stop)]
    for stop in STOP_TEXTS:
        if body.endswith(stop):
            body = body[: -len(stop)]
    if thinking:
        full = "<think" + body
        start_gt = full.find(">")
        end_tag = full.find("</think>")
        if end_tag < 0:
            if finish_reason != "length":
                raise SearchReaderError("pi_agent", "Unclosed reasoning block in continuation")
            reasoning = full[start_gt + 1 :] if start_gt >= 0 else full
            text = ""
        else:
            reasoning = full[start_gt + 1 : end_tag] if start_gt >= 0 else ""
            text = full[end_tag + len("</think>") :]
            if text.startswith("\r\n"):
                text = text[2:]
            elif text.startswith("\n"):
                text = text[1:]
    else:
        reasoning = ""
        text = body
        if "</think>" in text:
            reasoning_part, _, rest = text.partition("</think>")
            reasoning = reasoning_part.removeprefix("<think>")
            text = rest.lstrip("\r\n")

    envelope_pos = call_start(text, partial=False)
    if envelope_pos is None:
        return {"reasoning": reasoning, "content": text.strip(), "tool_calls": []}
    if finish_reason != "stop":
        raise SearchReaderError("pi_agent", "Truncated tool call; execution refused")
    visible = text[:envelope_pos].strip()
    calls = parse_pi_xml_calls(text[envelope_pos:], tools, recover=recover)
    return {"reasoning": reasoning, "content": visible, "tool_calls": calls}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class PiLoopGuard:
    """Port of rwkv-pi `loop-guard.ts`: blocks identical calls, repeated failures, and per-URL loops."""

    def __init__(
        self,
        *,
        repeat_failure_limit: int = 3,
        identical_call_limit: int = 2,
        target_failure_limit: int = 3,
    ) -> None:
        self.repeat_failure_limit = repeat_failure_limit
        self.identical_call_limit = identical_call_limit
        self.target_failure_limit = target_failure_limit
        self.history: list[dict[str, Any]] = []

    def check_before_call(self, name: str, args: dict[str, Any]) -> str | None:
        sig = _canonical_json([name, args])
        success_matches = sum(
            1 for h in self.history if not h["is_error"] and h["sig"] == sig
        )
        if self.identical_call_limit and success_matches >= self.identical_call_limit:
            return (
                f"[LoopGuard] 相同参数的 {name} 已成功执行过 {success_matches} 次；"
                "请直接使用已有 <tool_response> 中的证据回答，或改用新的查询词/URL。"
            )
        target = str(args.get("url") or "").strip()
        if target and self.target_failure_limit:
            streak = 0
            for h in reversed(self.history):
                if h["is_error"] and h["name"] == name and h["target"] == target:
                    streak += 1
                else:
                    break
            if streak >= self.target_failure_limit:
                return (
                    f"[LoopGuard] {name} 对 {target} 已连续失败 {streak} 次，本次不再重复请求该地址；"
                    "请改用其他来源或依据已有材料说明。"
                )
        return None

    def record(self, name: str, args: dict[str, Any], content: str, *, is_error: bool) -> str | None:
        sig = _canonical_json([name, args])
        self.history.append(
            {
                "name": name,
                "sig": sig,
                "target": str(args.get("url") or "").strip(),
                "result_sig": _canonical_json([name, args, content[:400]]),
                "is_error": is_error,
            }
        )
        if is_error and self.repeat_failure_limit and len(self.history) >= self.repeat_failure_limit:
            tail = self.history[-self.repeat_failure_limit :]
            if all(item["is_error"] and item["result_sig"] == tail[0]["result_sig"] for item in tail):
                return f"Identical failing tool call repeated {self.repeat_failure_limit} times"
        return None


def compress_pi_epoch(
    messages: list[dict[str, Any]],
    *,
    budget_tokens: int = DEFAULT_EPOCH_BUDGET_TOKENS,
    keep_recent_pairs: int = 2,
    tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """9.6K sweet-spot intra-epoch append-only compression (from `rwkv-pi/src/epoch.ts`)."""

    return fold_epoch_messages(
        messages,
        tools if tools is not None else WEB_TOOLS_SCHEMA,
        budget_tokens=budget_tokens,
        keep_recent_pairs=keep_recent_pairs,
        count_fn=estimate_rwkv_tokens,
    ).messages


SYSTEM_PROMPT_ZH = (
    "你是严谨的中文搜索阅读助手，只能依据真实工具返回的证据回答外部事实问题。\n"
    "工作准则：\n"
    "1. 需要查询最新信息、软件版本、新闻动态或官方文档时，调用 `web_search`；若用户已给出明确 URL 或搜索结果中有关键官方页面需要核实细节，调用 `web_fetch` 读取原文。\n"
    "2. 严格核对每个对象的归属、版本号、实际发布日期与明确声明“不支持”的功能（例如文档写明不支持写入时，绝不能编造对称的写入函数）；区分游戏/软件本体名称与版本主题名称。\n"
    "3. 证据冲突或官方材料未提供某项指标（如缺少性能基准测试或收盘价）时，如实说明无法确认，严禁猜测数字。\n"
    "4. 最终回答使用清晰中文，在每个关键事实后使用 `[S1]`、`[S2]` 格式标注工具返回中对应的来源编号，不要在回答末尾输出任何 `<tool_call>` 标签。"
)


class PiWebAgent:
    """Executes multi-step Search + Fetch reasoning in `rwkv-pi-xml-continuation-v3` format."""

    def __init__(self, settings: Settings, provider: SearchProvider) -> None:
        self.settings = settings
        self.provider = provider
        self._rr = 0

    def _next_client(self) -> OpenAICompatibleClient:
        urls = self.settings.rwkv_base_urls or (self.settings.rwkv_base_url,)
        url = urls[self._rr % len(urls)]
        self._rr += 1
        adapter_cfg: dict[str, Any] = {}
        if self.settings.rwkv_adapter_id:
            adapter_cfg["adapter_id"] = self.settings.rwkv_adapter_id
            if self.settings.rwkv_adapter_version:
                adapter_cfg["adapter_version"] = self.settings.rwkv_adapter_version
            adapter_cfg["adapter_scale"] = self.settings.rwkv_adapter_scale
        if self.settings.rwkv_state_id:
            adapter_cfg["state_id"] = self.settings.rwkv_state_id
        return OpenAICompatibleClient(
            base_url=url,
            api_key=self.settings.rwkv_api_key,
            model=self.settings.rwkv_model,
            timeout_seconds=self.settings.model_timeout_seconds,
            extra_headers=self.settings.rwkv_access_headers,
            context_window=self.settings.rwkv_context_window,
            adapter_config=adapter_cfg,
        )

    def run(
        self,
        question: str,
        *,
        session_id: str = "",
        conversation_history: list[dict[str, Any]] | None = None,
        on_progress: Callable[[str, dict[str, object]], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        on_sources: Callable[[list[SearchHit]], None] | None = None,
    ) -> PipelineResult:
        del on_reasoning_delta
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        trace: list[dict[str, object]] = []
        warnings: list[str] = []
        queries: list[str] = []
        sources: list[SearchHit] = []
        source_by_url: dict[str, SearchHit] = {}
        model_calls: list[dict[str, object]] = []

        def emit(stage: str, detail: dict[str, object]) -> None:
            event = {
                "stage": stage,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                **detail,
            }
            trace.append(event)
            if on_progress is not None:
                on_progress(stage, detail)

        today = datetime.now().astimezone().date().isoformat()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"当前日期：{today}\n{SYSTEM_PROMPT_ZH}"}
        ]
        for turn in (conversation_history or [])[-6:]:
            q_prev = str(turn.get("question") or "").strip()
            a_prev = str(turn.get("answer") or "").strip()
            if q_prev and a_prev:
                messages.append({"role": "user", "content": q_prev})
                messages.append({"role": "assistant", "reasoning": "", "content": a_prev})
        messages.append({"role": "user", "content": question})

        guard = PiLoopGuard()
        # Match rwkv-pi default (`--thinking off` for base-model tool envelopes unless
        # explicitly `on` or an adapter is loaded; OpenThink on synthesis without tools).
        tool_thinking = self.settings.rwkv_thinking_mode == "on" or bool(
            self.settings.rwkv_adapter_id
        )
        synth_thinking = self.settings.rwkv_thinking_mode != "off"

        def register_hit(hit: SearchHit) -> SearchHit:
            norm = normalize_url(hit.url) or hit.url
            key = norm.split("#", 1)[0].casefold()
            existing = source_by_url.get(key)
            if existing is not None:
                if len(hit.content) > len(existing.content):
                    existing.content = hit.content[: self.settings.max_evidence_chars]
                    existing.content_markdown = hit.content_markdown or hit.content
                    existing.content_status = "fetched"
                return existing
            hit.source_id = f"S{len(sources) + 1}"
            hit.canonical_url = norm
            sources.append(hit)
            source_by_url[key] = hit
            return hit

        def exec_tool(call: dict[str, Any]) -> tuple[str, bool]:
            if call.get("error"):
                return f"Error: {call['error']}", True
            fn = call["function"]
            name = fn["name"]
            args = strict_json_loads(fn["arguments"])
            blocked = guard.check_before_call(name, args)
            if blocked:
                guard.record(name, args, blocked, is_error=True)
                return blocked, True
            if name == "web_search":
                q = str(args.get("query") or "").strip()
                if not q:
                    msg = "Error: query must be non-empty."
                    guard.record(name, args, msg, is_error=True)
                    return msg, True
                limit = max(1, min(8, int(args.get("max_results") or 6)))
                emit("searching", {"query": q, "message": f"正在搜索：{q[:48]}"})
                try:
                    hits = self.provider.search_with_context(q, limit, ori_question=question)
                except SearchReaderError as exc:
                    msg = f"Error: search failed ({exc})"
                    guard.record(name, args, msg, is_error=True)
                    return msg, True
                queries.append(q)
                if not hits:
                    msg = f"No search results found for query: {q}"
                    guard.record(name, args, msg, is_error=False)
                    return msg, False
                rendered_rows: list[str] = []
                for h in hits[:limit]:
                    reg = register_hit(h)
                    body_preview = (h.content or h.snippet or "").strip()[:1400]
                    date_note = f" | date={h.published_date}" if h.published_date else ""
                    rendered_rows.append(
                        f"[{reg.source_id}] {reg.title}\nURL: {reg.url}{date_note}\nExcerpt: {body_preview}"
                    )
                if on_sources is not None:
                    on_sources(list(sources))
                out = "\n\n".join(rendered_rows)
                guard.record(name, args, out, is_error=False)
                return out, False
            if name == "web_fetch":
                url = normalize_url(str(args.get("url") or "").strip())
                focus = str(args.get("focus") or "").strip()
                if not url:
                    msg = "Error: invalid http/https URL."
                    guard.record(name, args, msg, is_error=True)
                    return msg, True
                emit("reading", {"url": url, "message": f"正在读取网页：{urlsplit(url).hostname or url}"})
                fetched = self.provider.fetch_url(url)
                if not fetched or not fetched[1].strip():
                    msg = f"Error: unable to fetch readable content from {url}"
                    guard.record(name, args, msg, is_error=True)
                    return msg, True
                title, content = fetched
                hit = register_hit(
                    SearchHit(
                        title=title or url,
                        url=url,
                        content=content[: self.settings.max_evidence_chars],
                        content_markdown=content,
                        content_status="fetched",
                        provider="direct-url",
                    )
                )
                if on_sources is not None:
                    on_sources(list(sources))
                excerpt = slim_sweet_spot_text(content, 1800, estimate_rwkv_tokens)
                if len(content) > 6500 and (focus or question):
                    from .retrieval import _focus_direct_url_content

                    focused = _focus_direct_url_content(content, f"{focus} {question}".strip(), 6500)
                    excerpt = slim_sweet_spot_text(focused, 1800, estimate_rwkv_tokens)
                out = f"[{hit.source_id}] {hit.title}\nURL: {hit.url}\nContent:\n{excerpt}"
                guard.record(name, args, out, is_error=False)
                return out, False
            msg = f"Error: unknown tool '{name}'"
            guard.record(name, args, msg, is_error=True)
            return msg, True

        final_answer = ""
        for turn_idx in range(1, self.settings.pi_agent_max_turns + 1):
            folded_res = fold_epoch_messages(
                messages,
                WEB_TOOLS_SCHEMA,
                budget_tokens=self.settings.pi_epoch_budget_tokens,
                count_fn=estimate_rwkv_tokens,
            )
            active_messages = folded_res.messages
            use_turn_think = tool_thinking
            prompt = render_pi_prompt(
                active_messages,
                WEB_TOOLS_SCHEMA,
                thinking=use_turn_think,
                generation=True,
            )
            client = self._next_client()
            t_call = time.perf_counter()
            raw, reason, matched_stop, prompt_tokens = client.raw_continuation(
                prompt,
                max_tokens=min(4096, self.settings.rwkv_max_tokens + 2048),
                thinking=use_turn_think,
                has_tools=True,
                stage="pi_agent",
            )
            call_ms = round((time.perf_counter() - t_call) * 1000, 1)
            try:
                parsed = split_continuation(
                    raw,
                    thinking=use_turn_think,
                    finish_reason=reason,
                    matched_stop=matched_stop,
                    tools=WEB_TOOLS_SCHEMA,
                    recover=True,
                )
            except SearchReaderError:
                if not use_turn_think:
                    raise
                use_turn_think = False
                prompt = render_pi_prompt(
                    active_messages,
                    WEB_TOOLS_SCHEMA,
                    thinking=False,
                    generation=True,
                )
                raw, reason, matched_stop, prompt_tokens = client.raw_continuation(
                    prompt,
                    max_tokens=min(4096, self.settings.rwkv_max_tokens + 2048),
                    thinking=False,
                    has_tools=True,
                    stage="pi_agent",
                )
                parsed = split_continuation(
                    raw,
                    thinking=False,
                    finish_reason=reason,
                    matched_stop=matched_stop,
                    tools=WEB_TOOLS_SCHEMA,
                    recover=True,
                )
            model_calls.append(
                {
                    "turn": turn_idx,
                    "stage": "pi_agent",
                    "elapsed_ms": call_ms,
                    "prompt_tokens": prompt_tokens,
                    "estimated_epoch_tokens": folded_res.prompt_tokens,
                    "epoch_count": folded_res.epoch_count,
                    "epoch_folded": folded_res.folded,
                    "tool_call_count": len(parsed["tool_calls"]),
                    "reasoning_chars": len(parsed["reasoning"]),
                }
            )
            if not parsed["tool_calls"]:
                candidate_answer = str(parsed["content"] or "").strip()
                if candidate_answer:
                    final_answer = candidate_answer
                    messages.append(
                        {
                            "role": "assistant",
                            "reasoning": parsed["reasoning"],
                            "content": final_answer,
                        }
                    )
                break

            messages.append(
                {
                    "role": "assistant",
                    "reasoning": parsed["reasoning"],
                    "content": parsed["content"],
                    "tool_calls": [
                        {"id": c["id"], "type": "function", "function": c["function"]}
                        for c in parsed["tool_calls"]
                    ],
                }
            )
            abort_reason: str | None = None
            for call in parsed["tool_calls"]:
                obs, is_err = exec_tool(call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": obs,
                    }
                )
                if is_err and len(guard.history) >= guard.repeat_failure_limit:
                    tail = guard.history[-guard.repeat_failure_limit :]
                    if all(x["is_error"] and x["result_sig"] == tail[0]["result_sig"] for x in tail):
                        abort_reason = "repeated identical tool error"
                        break
            if abort_reason:
                warnings.append(f"pi_agent loop guard triggered: {abort_reason}")
                break

        if not final_answer:
            # Force final synthesis turn without tools (`choice="none"` equivalent).
            emit("finalizing", {"source_count": len(sources), "message": "正在汇总证据生成回答"})
            folded_synth = fold_epoch_messages(
                messages,
                [],
                budget_tokens=self.settings.pi_epoch_budget_tokens,
                count_fn=estimate_rwkv_tokens,
            )
            synth_messages = [
                *folded_synth.messages,
                {
                    "role": "user",
                    "content": "请不再调用任何工具，直接根据上述 <tool_response> 中已取得的证据回答原问题，并用 [S1] 格式标注来源；未确认的事项如实说明。",
                },
            ]
            for try_think in ([synth_thinking, False] if synth_thinking else [False]):
                prompt = render_pi_prompt(synth_messages, [], thinking=try_think, generation=True)
                client = self._next_client()
                raw, reason, matched_stop, _ = client.raw_continuation(
                    prompt,
                    max_tokens=self.settings.rwkv_max_tokens,
                    thinking=try_think,
                    has_tools=False,
                    stage="finalize",
                )
                try:
                    parsed = split_continuation(
                        raw,
                        thinking=try_think,
                        finish_reason=reason,
                        matched_stop=matched_stop,
                        tools=[],
                        recover=True,
                    )
                except SearchReaderError:
                    if try_think:
                        continue
                    raise
                final_answer = str(parsed["content"] or "").strip()
                if final_answer:
                    break

        if not final_answer:
            raise SearchReaderError("pi_agent", "RWKV pi_agent produced an empty final answer")

        if on_delta is not None:
            on_delta(final_answer)

        evidence = [
            Evidence(
                source_id=s.source_id,
                title=s.title,
                url=s.url,
                claim=(s.snippet or s.content[:300]).strip(),
                quote=s.content[:2000],
                score=s.score,
                published_date=s.published_date,
                authority=s.authority,
            )
            for s in sources
        ]
        total_ms = round((time.perf_counter() - started) * 1000, 1)
        emit("completed", {"duration_ms": total_ms})

        if self.settings.trace_enabled:
            self._append_pi_trajectory(
                self.settings.trace_directory,
                {
                    "schema_version": PROMPT_VERSION,
                    "request_id": request_id,
                    "session_id": session_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model": self.settings.rwkv_model,
                    "question": question,
                    "messages": messages,
                    "tools": WEB_TOOLS_SCHEMA,
                    "sources": [{"id": s.source_id, "title": s.title, "url": s.url} for s in sources],
                    "model_calls": model_calls,
                    "warnings": warnings,
                    "duration_ms": total_ms,
                },
            )

        return PipelineResult(
            request_id=request_id,
            session_id=session_id,
            question=question,
            answer=final_answer,
            queries=queries,
            sources=sources,
            evidence=evidence,
            retrieval_mode="pi-xml-agent",
            search_provider=self.provider.name,
            reader_mode=PROMPT_VERSION,
            degraded=bool(warnings),
            warnings=warnings,
            timings_ms={"total": total_ms},
            trace=trace,
        )

    @staticmethod
    def _append_pi_trajectory(directory: Path, record: dict[str, Any]) -> None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            target = directory / f"pi-trajectories-{day}.jsonl"
            with target.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
