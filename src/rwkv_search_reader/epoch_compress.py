"""RWKV 9.6K/9.8K Context Sweet-Spot & Intra-Epoch Append-Only Compression.

Faithful Python port of `rwkv-pi/src/epoch.ts` and `estimateRwkvTokens` from
`rwkv-pi/src/worklist.ts`, preserving RWKV-7 G1L RNN state fidelity within the
9,600-token sweet spot while maintaining 100% intra-epoch Prefix Cache reuse.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
import re
from typing import Any

DEFAULT_EPOCH_BUDGET_TOKENS = 9600
DEFAULT_EPOCH_HARD_SLACK_TOKENS = 2400
DEFAULT_EPOCH_BACKBONE_MAX_TOKENS = 3600
DEFAULT_EPOCH_KEEP_RECENT_PAIRS = 2
DEFAULT_EPOCH_MAX_RESULT_TOKENS = 2000

MAX_PINNED_UNEDITED_READ_TOKENS = 1800
MAX_PRIOR_TURN_LAST_RESULT_CHARS = 1600

READ_NAMES = frozenset({"read", "read_file", "web_fetch"})
SEARCH_NAMES = frozenset({"grep", "find", "ls", "web_search"})
WRITE_NAMES = frozenset({"edit", "edit_file", "write", "write_file"})


def estimate_rwkv_tokens(text: str) -> int:
    """Exact port of `estimateRwkvTokens` from `rwkv-pi/src/worklist.ts`."""

    if not text:
        return 0
    weight = 0.0
    space_run = 0
    for ch in text:
        code = ord(ch)
        if code == 32:
            space_run += 1
            weight += 0.22 if space_run > 1 else 0.26
            continue
        space_run = 0
        if (
            (0x4E00 <= code <= 0x9FFF)
            or (0x3000 <= code <= 0x303F)
            or (0xFF00 <= code <= 0xFFEF)
        ):
            weight += 0.96
        elif code in (10, 9):
            weight += 0.55
        elif 48 <= code <= 57:
            weight += 0.50
        elif (65 <= code <= 90) or (97 <= code <= 122):
            weight += 0.25
        elif code < 0x80:
            weight += 0.50
        else:
            weight += 0.85
    return max(1, math.ceil(weight))


def slim_sweet_spot_text(
    raw: str,
    max_tokens: int,
    count_fn: Callable[[str], int] = estimate_rwkv_tokens,
) -> str:
    """Keep 75% head + 25% tail when a single observation/document exceeds `max_tokens`."""

    if not raw or max_tokens <= 0 or count_fn(raw) <= max_tokens:
        return raw
    ratio = len(raw) / max(1, count_fn(raw))
    max_chars = max(800, math.floor(max_tokens * max(1.05, min(3.1, ratio * 0.92))))
    if len(raw) <= max_chars:
        return raw
    head_chars = math.floor(max_chars * 0.75)
    tail_chars = max_chars - head_chars
    return (
        f"{raw[:head_chars]}\n\n"
        "[... middle content omitted to preserve context sweet spot ...]\n\n"
        f"{raw[len(raw) - tail_chars :]}"
    )


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]
        )
    return ""


def _is_error_result_text(text: str) -> bool:
    trimmed = text.strip()
    return bool(
        re.match(
            r"^(?:Error:|\[LoopGuard\]|RWKV_|Tool .* failed|invalid |unknown |ENOENT)",
            trimmed,
            flags=re.IGNORECASE,
        )
        or re.search(r"Command exited with code [1-9]\d*", trimmed, flags=re.IGNORECASE)
    )


def _clean_one_line(text: str, max_len: int = 96) -> str:
    clean = re.sub(r"\s+", " ", text.strip())
    return f"{clean[: max_len - 3]}..." if len(clean) > max_len else clean


@dataclass(frozen=True)
class ToolCallEntry:
    id: str
    name: str
    args: dict[str, Any]


class EpochGroup:
    """Identity-hashed message group (`user`, `final_assistant`, or `tool_group`)."""

    __slots__ = ("kind", "messages", "calls", "ok")

    def __init__(
        self,
        kind: str,
        messages: list[dict[str, Any]],
        calls: list[ToolCallEntry],
        ok: bool,
    ) -> None:
        self.kind = kind
        self.messages = messages
        self.calls = calls
        self.ok = ok


def _extract_tool_calls(msg: dict[str, Any]) -> list[ToolCallEntry]:
    raw_calls = msg.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    out: list[ToolCallEntry] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        name = str(fn.get("name") or "")
        raw_args = fn.get("arguments")
        args: dict[str, Any] = {}
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        out.append(ToolCallEntry(id=str(call.get("id") or ""), name=name, args=args))
    return out


def _render_single_message_estimate(msg: dict[str, Any]) -> str:
    role = msg.get("role")
    if role == "system":
        return f"System: {_message_text(msg.get('content'))}\n\n"
    if role == "user":
        return f"User: {_message_text(msg.get('content'))}\n\n"
    if role == "tool":
        return f"User: <tool_response>\n{_message_text(msg.get('content'))}\n</tool_response>\n\n"
    if role == "assistant":
        reasoning = _message_text(msg.get("reasoning_content") or msg.get("reasoning"))
        body = _message_text(msg.get("content"))
        calls = "\n".join(
            f"<tool_call>{json.dumps({'name': c.name, 'arguments': c.args}, ensure_ascii=False, separators=(',', ':'))}</tool_call>"
            for c in _extract_tool_calls(msg)
        )
        joined = "\n".join(part for part in (body, calls) if part)
        return f"Assistant: <think>{reasoning}</think>\n{joined}\n\n"
    return ""


def _slim_oversized_tool_message(
    msg: dict[str, Any],
    max_result_tokens: int,
    count_fn: Callable[[str], int],
) -> dict[str, Any]:
    if msg.get("role") != "tool" or max_result_tokens <= 0:
        return msg
    raw = _message_text(msg.get("content"))
    if not raw or count_fn(raw) <= max_result_tokens:
        return msg
    slimmed = slim_sweet_spot_text(raw, max_result_tokens, count_fn)
    if slimmed == raw:
        return msg
    return {**msg, "content": slimmed}


def _parse_epoch_groups(
    msgs: list[dict[str, Any]],
    max_result_tokens: int,
    count_fn: Callable[[str], int],
) -> list[EpochGroup]:
    groups: list[EpochGroup] = []
    i = 0
    n = len(msgs)
    while i < n:
        msg = msgs[i]
        if (
            msg.get("role") == "assistant"
            and isinstance(msg.get("tool_calls"), list)
            and len(msg["tool_calls"]) > 0
        ):
            calls = _extract_tool_calls(msg)
            pending = {c.id for c in calls if c.id}
            group_msgs: list[dict[str, Any]] = [msg]
            j = i + 1
            all_ok = True
            while j < n and msgs[j].get("role") == "tool":
                raw_tool = msgs[j]
                tool_msg = _slim_oversized_tool_message(raw_tool, max_result_tokens, count_fn)
                group_msgs.append(tool_msg)
                tc_id = raw_tool.get("tool_call_id")
                if isinstance(tc_id, str):
                    pending.discard(tc_id)
                if raw_tool.get("isError") is True or _is_error_result_text(
                    _message_text(raw_tool.get("content"))
                ):
                    all_ok = False
                j += 1
                if not pending and (j >= n or msgs[j].get("role") != "tool"):
                    break
            groups.append(EpochGroup("tool_group", group_msgs, calls, all_ok))
            i = j
            continue
        if msg.get("role") == "assistant":
            groups.append(EpochGroup("final_assistant", [msg], [], True))
        else:
            groups.append(EpochGroup("user", [msg], [], True))
        i += 1
    return groups


def _strip_group_reasoning(group: EpochGroup) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, msg in enumerate(group.messages):
        if idx != 0 or msg.get("role") != "assistant":
            out.append(msg)
            continue
        copy = dict(msg)
        had_reasoning = "reasoning_content" in copy or "reasoning" in copy
        copy.pop("reasoning", None)
        if isinstance(copy.get("tool_calls"), list) and copy["tool_calls"] and had_reasoning:
            copy["reasoning_content"] = ""
        else:
            copy.pop("reasoning_content", None)
        out.append(copy)
    return out


def _append_ledger_to_user_message(msg: dict[str, Any], ledger_text: str) -> dict[str, Any]:
    existing = _message_text(msg.get("content")).rstrip()
    return {
        **msg,
        "role": "user",
        "content": f"{existing}\n\n{ledger_text}" if existing else ledger_text,
    }


def _target_of_args(args: dict[str, Any]) -> str | None:
    raw = args.get("path") or args.get("file_path") or args.get("url")
    if isinstance(raw, str) and raw.strip():
        return re.sub(r"^\./", "", raw.strip())
    return None


def build_epoch_state_ledger(
    groups: list[EpochGroup],
    header: str = "State Ledger (earlier steps folded)",
) -> dict[str, Any]:
    """Deterministic Ground-Truth State Ledger extracted from folded `(assistant + tool)` groups."""

    succeeded_keys: set[str] = set()
    inspected: list[str] = []
    inspected_seen: set[str] = set()
    web_summaries: list[str] = []
    failed: list[str] = []

    for group in groups:
        tool_results = [m for m in group.messages if m.get("role") == "tool"]
        for idx, call in enumerate(group.calls):
            res_msg = tool_results[idx] if idx < len(tool_results) else None
            res_text = _message_text(res_msg.get("content")) if res_msg else ""
            call_failed = (not group.ok) or bool(
                res_msg
                and (res_msg.get("isError") is True or _is_error_result_text(res_text))
            )
            target = _target_of_args(call.args)
            source_ids = re.findall(r"\[S\d+\]", res_text[:2000])
            sid_tag = f" {''.join(dict.fromkeys(source_ids[:4]))}" if source_ids else ""

            if call.name in ("web_search", "web_fetch"):
                q = _clean_one_line(str(call.args.get("query") or call.args.get("url") or ""), 72)
                status = "failed" if call_failed else "ok"
                web_summaries.append(
                    f"{call.name} {json.dumps(q, ensure_ascii=False)} ({status}{sid_tag})"
                )
                if not call_failed and target:
                    succeeded_keys.add(f"{call.name}:{target}")
                if call_failed:
                    first_err = _clean_one_line(
                        next((ln for ln in res_text.splitlines() if ln.strip()), "failed"), 96
                    )
                    failed.append(f"{call.name} {q} -> {first_err}")
                elif call.name == "web_fetch" and target:
                    entry = f"fetch {target}{sid_tag}"
                    if entry not in inspected_seen:
                        inspected_seen.add(entry)
                        inspected.append(entry)
                continue

            if call_failed:
                label = target or _clean_one_line(
                    str(call.args.get("command") or call.args.get("pattern") or ""), 72
                )
                first_err = _clean_one_line(
                    next((ln for ln in res_text.splitlines() if ln.strip()), "failed"), 96
                )
                failed.append(f"{call.name}{f' {label}' if label else ''} -> {first_err}")
                continue

            if target:
                succeeded_keys.add(f"{call.name}:{target}")
            if call.name in READ_NAMES and target:
                entry = f"read {target}"
                if entry not in inspected_seen:
                    inspected_seen.add(entry)
                    inspected.append(entry)
            elif call.name in SEARCH_NAMES:
                q = _clean_one_line(str(call.args.get("pattern") or target or "."), 64)
                entry = f"{call.name} {json.dumps(q, ensure_ascii=False)}"
                if entry not in inspected_seen:
                    inspected_seen.add(entry)
                    inspected.append(entry)

    unresolved_failed = [
        line
        for line in failed
        if not any(
            line.startswith(f"{key.split(':', 1)[0]} {key.split(':', 1)[1]} ->")
            for key in succeeded_keys
            if ":" in key
        )
    ]

    sections: list[str] = [f"[{header}:"]
    if inspected:
        shown = inspected[-12:]
        omitted = len(inspected) - len(shown)
        suffix = f" (+{omitted} earlier)" if omitted > 0 else ""
        sections.append(f"- Inspected: {', '.join(shown)}{suffix}")
    if web_summaries:
        sections.append(f"- Web queries: {'; '.join(web_summaries[-8:])}")
    if unresolved_failed:
        sections.append(
            f"- Failed attempts (do not repeat verbatim): {'; '.join(unresolved_failed[-5:])}"
        )
    sections[-1] = f"{sections[-1]}]"
    return {"role": "user", "content": "\n".join(sections)}


def _find_latest_unedited_read_group(
    folded_groups: list[EpochGroup],
    group_tokens: dict[EpochGroup, int],
) -> EpochGroup | None:
    for group in reversed(folded_groups):
        if group.ok and any(c.name in WRITE_NAMES for c in group.calls):
            return None
        first = group.calls[0] if group.calls else None
        if (
            group.ok
            and len(group.calls) == 1
            and first is not None
            and first.name in READ_NAMES
            and _target_of_args(first.args) is not None
        ):
            return (
                group
                if group_tokens.get(group, 0) <= MAX_PINNED_UNEDITED_READ_TOKENS
                else None
            )
    return None


def _compact_prior_last_tool_group(group: EpochGroup) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in _strip_group_reasoning(group):
        if msg.get("role") != "tool":
            out.append(msg)
            continue
        raw = _message_text(msg.get("content"))
        if len(raw) <= MAX_PRIOR_TURN_LAST_RESULT_CHARS:
            out.append(msg)
        else:
            out.append(
                {
                    **msg,
                    "content": f"{raw[:1200]}\n...(prior-turn output folded)...\n{raw[-450:]}",
                }
            )
    return out


def _fold_completed_prior_turn(
    turn_slice: list[dict[str, Any]],
    max_result_tokens: int,
    count_fn: Callable[[str], int],
) -> list[dict[str, Any]]:
    groups = _parse_epoch_groups(turn_slice, max_result_tokens, count_fn)
    tool_groups = [g for g in groups if g.kind == "tool_group"]
    if not tool_groups:
        return [
            m
            for g in groups
            for m in (_strip_group_reasoning(g) if g.kind == "final_assistant" else g.messages)
        ]
    last_tool = next((g for g in reversed(tool_groups) if g.ok), tool_groups[-1])
    earlier_tools = [g for g in tool_groups if g is not last_tool]
    has_final = any(g.kind == "final_assistant" for g in groups)

    out: list[dict[str, Any]] = []
    prior_ledger_text = (
        _message_text(
            build_epoch_state_ledger(earlier_tools, "Prior-turn State Ledger").get("content")
        )
        if earlier_tools
        else ""
    )
    ledger_merged = False
    for g in groups:
        if g.kind == "user":
            if not ledger_merged and prior_ledger_text and g.messages:
                out.append(_append_ledger_to_user_message(g.messages[0], prior_ledger_text))
                out.extend(g.messages[1:])
                ledger_merged = True
            else:
                out.extend(g.messages)
        elif g.kind == "final_assistant":
            out.extend(_strip_group_reasoning(g))
        elif g is last_tool:
            if not ledger_merged and prior_ledger_text:
                out.append({"role": "user", "content": prior_ledger_text})
                ledger_merged = True
            out.extend(
                _compact_prior_last_tool_group(g) if has_final else _strip_group_reasoning(g)
            )
    return out


@dataclass(frozen=True)
class FoldEpochResult:
    messages: list[dict[str, Any]]
    prompt_tokens: int
    epoch_count: int
    folded: bool


def fold_epoch_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    budget_tokens: int = DEFAULT_EPOCH_BUDGET_TOKENS,
    hard_cap_tokens: int | None = None,
    backbone_max_tokens: int = DEFAULT_EPOCH_BACKBONE_MAX_TOKENS,
    keep_recent_pairs: int = DEFAULT_EPOCH_KEEP_RECENT_PAIRS,
    max_result_tokens: int = DEFAULT_EPOCH_MAX_RESULT_TOKENS,
    count_fn: Callable[[str], int] = estimate_rwkv_tokens,
) -> FoldEpochResult:
    """Port of `foldEpochRequest` from `rwkv-pi/src/epoch.ts` (9.6K sweet spot)."""

    effective_hard_cap = (
        hard_cap_tokens
        if hard_cap_tokens is not None
        else budget_tokens + DEFAULT_EPOCH_HARD_SLACK_TOKENS
    )
    keep_pairs = max(1, keep_recent_pairs)
    raw_tools = list(tools or [])

    system_messages: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_messages.append(msg)
        else:
            conversation.append(msg)

    tools_header_tokens = (
        count_fn(f"Tools:\n{json.dumps(raw_tools, ensure_ascii=False)}\n") + 45
        if raw_tools
        else 0
    )

    def turn_tokens(seq: list[dict[str, Any]]) -> int:
        return sum(count_fn(_render_single_message_estimate(m)) for m in seq)

    system_tokens = turn_tokens(system_messages) + tools_header_tokens
    user_indices = [i for i, m in enumerate(conversation) if m.get("role") == "user"]
    if not user_indices:
        tok = system_tokens + turn_tokens(conversation)
        return FoldEpochResult(messages=list(messages), prompt_tokens=tok, epoch_count=0, folded=False)

    prior_turns: list[list[dict[str, Any]]] = []
    for t in range(len(user_indices) - 1):
        start_i = 0 if t == 0 else user_indices[t]
        end_i = user_indices[t + 1]
        prior_turns.append(
            _fold_completed_prior_turn(conversation[start_i:end_i], max_result_tokens, count_fn)
        )
    prior_tokens = sum(turn_tokens(turn) for turn in prior_turns)
    while prior_turns and prior_tokens > backbone_max_tokens:
        dropped = prior_turns.pop(0)
        prior_tokens -= turn_tokens(dropped)
    prior_flat = [m for turn in prior_turns for m in turn]

    active_start = 0 if len(user_indices) == 1 else user_indices[-1]
    active_slice = conversation[active_start:]
    active_groups = _parse_epoch_groups(active_slice, max_result_tokens, count_fn)
    tool_groups = [g for g in active_groups if g.kind == "tool_group"]

    full_group_tokens: dict[EpochGroup, int] = {}
    bare_group_tokens: dict[EpochGroup, int] = {}
    for g in active_groups:
        full_group_tokens[g] = turn_tokens(g.messages)
        bare_group_tokens[g] = (
            turn_tokens(_strip_group_reasoning(g))
            if g.kind == "tool_group"
            else full_group_tokens[g]
        )

    flat_parsed = [m for g in active_groups for m in g.messages]
    any_slimmed = len(flat_parsed) != len(active_slice) or any(
        a is not b for a, b in zip(flat_parsed, active_slice, strict=False)
    )
    raw_active_tokens = sum(full_group_tokens[g] for g in active_groups)
    if not prior_turns and not any_slimmed and system_tokens + raw_active_tokens <= budget_tokens:
        return FoldEpochResult(
            messages=list(messages),
            prompt_tokens=system_tokens + raw_active_tokens,
            epoch_count=0,
            folded=False,
        )

    def select_frozen_pinned_read(cut_idx: int, cut_trigger_count: int) -> EpochGroup | None:
        if cut_idx <= 0:
            return None
        folded_tools = tool_groups[:cut_idx]
        candidate = _find_latest_unedited_read_group(folded_tools, full_group_tokens)
        if candidate is None:
            return None
        cand_target = _target_of_args(candidate.calls[0].args if candidate.calls else {})
        initial_active = tool_groups[cut_idx:cut_trigger_count]
        superseded = any(
            g.ok
            and any(
                (
                    c.name in READ_NAMES
                    and (cand_target is None or _target_of_args(c.args) == cand_target)
                )
                or (
                    c.name in WRITE_NAMES
                    and cand_target is not None
                    and _target_of_args(c.args) == cand_target
                )
                for c in g.calls
            )
            for g in initial_active
        )
        return None if superseded else candidate

    def build_window_for_cut(
        cut_idx: int,
        frozen_pinned_read: EpochGroup | None,
        end_tool_count: int,
        include_final_assistant: bool,
        enforce_hard_cap: bool,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        sub_tools = tool_groups[:end_tool_count]
        folded_tools = sub_tools[:cut_idx]
        active_tools = sub_tools[cut_idx:]
        active_set = set(active_tools)

        pinned_read = frozen_pinned_read if folded_tools else None
        ledger_base = (
            [g for g in folded_tools if g is not pinned_read]
            if pinned_read is not None
            else list(folded_tools)
        )

        kept_groups = [
            g
            for g in active_groups
            if g.kind == "user"
            or (g.kind == "final_assistant" and include_final_assistant)
            or g is pinned_read
            or g in active_set
        ]
        kept_set = set(kept_groups)
        cleared_reasoning: set[EpochGroup] = set()
        if pinned_read is not None:
            cleared_reasoning.add(pinned_read)
        extra_dropped: list[EpochGroup] = []

        def assemble() -> tuple[list[dict[str, Any]], int]:
            out: list[dict[str, Any]] = []
            tok = system_tokens + prior_tokens
            ledger_merged = False
            all_folded = [*ledger_base, *extra_dropped]
            ledger_msg = build_epoch_state_ledger(all_folded) if all_folded else None
            ledger_text = _message_text(ledger_msg.get("content")) if ledger_msg else ""

            for g in kept_groups:
                if g.kind == "user":
                    if not ledger_merged and ledger_text and g.messages:
                        merged_first = _append_ledger_to_user_message(g.messages[0], ledger_text)
                        rest = g.messages[1:]
                        seq = [merged_first, *rest]
                        out.extend(seq)
                        tok += turn_tokens(seq)
                        ledger_merged = True
                    else:
                        out.extend(g.messages)
                        tok += full_group_tokens.get(g, 0)
                    continue
                if not ledger_merged and ledger_msg is not None:
                    out.append(ledger_msg)
                    tok += count_fn(_render_single_message_estimate(ledger_msg))
                    ledger_merged = True
                if g.kind == "tool_group" and g in cleared_reasoning:
                    out.extend(_strip_group_reasoning(g))
                    tok += bare_group_tokens.get(g, 0)
                else:
                    out.extend(g.messages)
                    tok += full_group_tokens.get(g, 0)
            return out, tok

        built_msgs, built_tok = assemble()
        if enforce_hard_cap and built_tok > effective_hard_cap:
            for g in kept_groups:
                if built_tok <= effective_hard_cap:
                    break
                if g.kind == "tool_group" and g not in cleared_reasoning:
                    has_r = bool(
                        _message_text(
                            g.messages[0].get("reasoning_content")
                            or g.messages[0].get("reasoning")
                        )
                    )
                    if has_r:
                        cleared_reasoning.add(g)
                        built_msgs, built_tok = assemble()
            droppable = [g for g in active_tools if g in kept_set]
            while len(droppable) > 1 and built_tok > effective_hard_cap:
                oldest = droppable.pop(0)
                extra_dropped.append(oldest)
                kept_set.discard(oldest)
                kept_groups = [g for g in kept_groups if g is not oldest]
                built_msgs, built_tok = assemble()

        was_folded = bool(prior_turns or cut_idx > 0 or extra_dropped or any_slimmed)
        return built_msgs, built_tok, was_folded

    cut_idx = 0
    frozen_pinned_read: EpochGroup | None = None
    epoch_count = 0
    for k in range(len(tool_groups)):
        _, win_tok, _ = build_window_for_cut(cut_idx, frozen_pinned_read, k + 1, False, False)
        if win_tok > budget_tokens and (k + 1 - cut_idx) > 1:
            epoch_count += 1
            cut_idx = max(cut_idx + 1, (k + 1) - keep_pairs)
            frozen_pinned_read = select_frozen_pinned_read(cut_idx, k + 1)
            _, win_tok, _ = build_window_for_cut(cut_idx, frozen_pinned_read, k + 1, False, False)
            if frozen_pinned_read is not None and win_tok > budget_tokens:
                frozen_pinned_read = None
                _, win_tok, _ = build_window_for_cut(cut_idx, None, k + 1, False, False)
            while win_tok > budget_tokens and cut_idx < k:
                cut_idx += 1
                frozen_pinned_read = select_frozen_pinned_read(cut_idx, k + 1)
                _, win_tok, _ = build_window_for_cut(cut_idx, frozen_pinned_read, k + 1, False, False)
                if frozen_pinned_read is not None and win_tok > budget_tokens:
                    frozen_pinned_read = None
                    _, win_tok, _ = build_window_for_cut(cut_idx, None, k + 1, False, False)

    has_final = any(g.kind == "final_assistant" for g in active_groups)
    final_msgs, final_tok, final_folded = build_window_for_cut(
        cut_idx, frozen_pinned_read, len(tool_groups), has_final, True
    )
    return FoldEpochResult(
        messages=[*system_messages, *prior_flat, *final_msgs],
        prompt_tokens=final_tok,
        epoch_count=epoch_count,
        folded=final_folded,
    )
