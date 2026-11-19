#!/usr/bin/env python3
"""Build a 100% DSH-native NoRA training dataset (`rwkv_nora_tune_dp` segments v2).

Zero cross-harness format conversion:
- Uses only genuine DSH (`dsh-rwkv-plugin` + `deepseek-harness`) trajectories with real
  OpenThink (`<think>...</think>`), `<tool_call>`, `<tool_response>`, and cited answers.
- Deduplicates across `dsh_20261114-mix-relay`, `overnight-20261106`, and `r3-20261108-partial`.
- Aligns pure web-search / web-fetch / direct-answer rows to the exact DSH Web-Only
  `System:` prompt used by `RWKV-SearchReader` (`dsh_agent.py`), while retaining an
  untouched 8-tool DSH subset + relay handoff recaps + bilingual daily chat so the
  adapter works seamlessly in both `RWKV-SearchReader` and full DeepSeek Harness.
- Enforces strict hold-out exclusion against `searchchat40.jsonl` and `search_router_v6r2/test.jsonl`.
- Outputs strict single-field `{"segments": [...]}` JSONL rows required by `dataset.cpp`.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rwkv_search_reader.dsh_agent import render_dsh_system_prompt  # noqa: E402
from rwkv_search_reader.epoch_compress import estimate_rwkv_tokens  # noqa: E402

LOCAL_TOOLS = ("bash", "read", "write", "edit", "glob", "grep")
WEB_TOOLS = ("web_search", "web_fetch")

RUNTIME_CWD_NOTE_RE = re.compile(
    r'\n\nUser: Current working directory: "[^"]*"\.\n'
    r"Current DSH file policy: [^\n]*(?:\n[^\n]+)*?(?=\n\n(?:User:|Assistant:))"
)

MENTIONS_LOCAL_RE = re.compile(
    r"\b(bash|read_file|write_file|edit_file|glob|grep|/home/user/project|workspace-write)\b"
)

SAMPLE_DATES = (
    "",
    "",
    "Current date: 2026-11-06 (Asia/Shanghai, UTC+08:00).",
    "Current date: 2026-11-11 (Asia/Shanghai, UTC+08:00).",
    "Current date: 2026-11-18 (Asia/Shanghai, UTC+08:00).",
)


def _norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _extract_first_user_question(prompt: str) -> str:
    idx = prompt.find("\n\nUser: ")
    if idx < 0:
        if prompt.startswith("User: "):
            rest = prompt[6:]
            end = rest.find("\n\n")
            return rest[:end] if end >= 0 else rest
        return ""
    rest = prompt[idx + 8 :]
    end = rest.find("\n\n")
    return rest[:end] if end >= 0 else rest


def _load_heldout_questions() -> set[str]:
    heldout: set[str] = set()
    candidates = [
        Path("/mnt/f/pikv-ageval-data/work/pikv-agent-eval/chat_eval/tasks/searchchat40.jsonl"),
        PROJECT_ROOT / "data" / "search_router_v6r2" / "test.jsonl",
    ]
    for fp in candidates:
        if not fp.is_file():
            continue
        with fp.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                for key in ("question", "prompt", "instruction", "query"):
                    if isinstance(obj.get(key), str) and obj[key].strip():
                        heldout.add(_norm_question(obj[key]))
                for msg in obj.get("messages") or []:
                    if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                        heldout.add(_norm_question(msg["content"]))
    return heldout


def _swap_to_web_only_system(prompt: str, rng: random.Random) -> str | None:
    """Replace 8-tool DSH System block + CWD note with byte-exact DSH Web-Only System block."""

    user_idx = prompt.find("\n\nUser: ")
    if not prompt.startswith("System: ") or user_idx < 0:
        return None
    tail = prompt[user_idx:]
    tail = RUNTIME_CWD_NOTE_RE.sub("", tail)
    date_line = rng.choice(SAMPLE_DATES)
    web_sys = render_dsh_system_prompt(thinking=True, date_line=date_line)
    # `renderPrompt` applies `cleanChatText` to System entry (no `\n\n` inside System)
    while "\n\n" in web_sys:
        web_sys = web_sys.replace("\n\n", "\n")
    return f"System: {web_sys}{tail}"


def _slice_utf16_loss_span(text: str, start_cu: int, end_cu: int) -> tuple[str, str]:
    raw = text.encode("utf-16-le")
    b0 = start_cu * 2
    b1 = end_cu * 2
    if b0 < 0 or b1 > len(raw) or b0 >= b1:
        raise ValueError("Invalid UTF-16 span")
    prefix = raw[:b0].decode("utf-16-le")
    target = raw[b0:b1].decode("utf-16-le")
    return prefix, target


def _make_strict_row(seg0: str, seg1: str) -> dict[str, object] | None:
    if not seg0 or not seg1:
        return None
    # Verify exact C++ dataset.cpp contract
    return {
        "segments": [
            {"text": seg0, "train": False},
            {"text": seg1, "train": True, "end_target": 0},
            {"text": "<EOD>", "train": False},
        ]
    }


def build_dataset(out_dir: Path, max_ctx_tokens: int = 24576, seed: int = 20261119) -> dict[str, object]:
    rng = random.Random(seed)
    heldout = _load_heldout_questions()

    seen_step_sigs: set[str] = set()
    stats = Counter()

    pure_web_rows: list[tuple[str, str]] = []
    direct_rows: list[tuple[str, str]] = []
    relay_rows: list[tuple[str, str]] = []
    hybrid_web_doc_rows: list[tuple[str, str]] = []
    daily_chat_rows: list[tuple[str, str]] = []

    def try_register(seg0: str, seg1: str, source_tag: str) -> None:
        q = _extract_first_user_question(seg0)
        if q and _norm_question(q) in heldout:
            stats["dropped_heldout"] += 1
            return
        est_toks = estimate_rwkv_tokens(seg0 + seg1) + 16
        if est_toks > max_ctx_tokens - 256:
            stats["dropped_over_ctx"] += 1
            return
        # Dedup by conversation body after </tools> + target
        end_tools = seg0.find("</tools>")
        body_key_src = (seg0[end_tools + 8 :] if end_tools >= 0 else seg0) + "\x00" + seg1
        sig = hashlib.sha1(body_key_src.encode("utf-8")).hexdigest()
        if sig in seen_step_sigs:
            stats["dropped_duplicate"] += 1
            return

        if not seg0.startswith("System: "):
            seen_step_sigs.add(sig)
            daily_chat_rows.append((seg0, seg1))
            stats[f"kept_chat_{source_tag}"] += 1
            return

        if not seg0.endswith("Assistant: <think") or not seg1.startswith(">"):
            stats["dropped_wire_boundary"] += 1
            return

        after_sys = seg0[end_tools + 8 :] if end_tools >= 0 else seg0
        combined = after_sys + seg1
        uses_local = any(
            f'"name":"{lt}"' in combined or f'"tool":"{lt}"' in combined for lt in LOCAL_TOOLS
        )
        uses_web = any(
            f'"name":"{wt}"' in combined or f'"tool":"{wt}"' in combined for wt in WEB_TOOLS
        )
        is_recap = "Stop work. Write only a concise handoff recap" in after_sys
        mentions_local = bool(MENTIONS_LOCAL_RE.search(seg1))

        if uses_web and not uses_local and not mentions_local:
            seen_step_sigs.add(sig)
            pure_web_rows.append((seg0, seg1))
            stats[f"kept_pure_web_{source_tag}"] += 1
        elif not uses_web and not uses_local and not is_recap and not mentions_local:
            seen_step_sigs.add(sig)
            direct_rows.append((seg0, seg1))
            stats[f"kept_direct_{source_tag}"] += 1
        elif is_recap:
            seen_step_sigs.add(sig)
            relay_rows.append((seg0, seg1))
            stats[f"kept_relay_{source_tag}"] += 1
        elif uses_web and uses_local and ('"name":"web_fetch"' in seg1 or '"name":"web_search"' in seg1):
            seen_step_sigs.add(sig)
            hybrid_web_doc_rows.append((seg0, seg1))
            stats[f"kept_hybrid_webcall_{source_tag}"] += 1

    # 1. Primary source: dsh_20261114-mix-relay_b8ec24fa6d7e.jsonl
    relay_shard = Path("/mnt/f/rwkv-pi-data/mega-corpus-9k6-20261118/shards/dsh_20261114-mix-relay_b8ec24fa6d7e.jsonl")
    if relay_shard.is_file():
        with relay_shard.open("r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                segs = obj.get("segments", [])
                if len(segs) >= 2 and not segs[0].get("train") and segs[1].get("train"):
                    try_register(segs[0]["text"], segs[1]["text"], "mix20261114")

    # 2. Additional DSH native pure-web & direct trajectories from overnight-20261106 and r3-20261108-partial
    for tag, extra_fp in [
        ("overnight1106", Path("/mnt/f/rwkv-pi-data/org-hf-20261111/rwkv-agent-data.git/train/overnight-20261106/train-full.jsonl")),
        ("r3_1108", Path("/mnt/f/rwkv-pi-data/org-hf-20261111/rwkv-agent-data.git/train/r3-20261108-partial/train.jsonl")),
    ]:
        if not extra_fp.is_file():
            continue
        with extra_fp.open("r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                spans = obj.get("loss_spans") or []
                txt = obj.get("text", "")
                if len(spans) != 1 or not txt:
                    continue
                try:
                    p_str, t_str = _slice_utf16_loss_span(txt, int(spans[0][0]), int(spans[0][1]))
                except ValueError:
                    continue
                # In raw export, prefix ends with `Assistant: <think>` and target starts right after `<think` or `<think>`
                if p_str.endswith("Assistant: <think>"):
                    p_str = p_str[:-1]
                    t_str = ">" + t_str
                try_register(p_str, t_str, tag)

    # Assemble balanced DSH-native SearchReader corpus:
    # - 100% of pure_web_rows aligned to DSH Web-Only System prompt (byte-exact match to DshWebAgent!)
    # - + 30% subsample of pure_web_rows in untouched 8-tool DSH System prompt (full DSH compatibility!)
    # - 100% of direct_rows aligned to DSH Web-Only System prompt
    # - 250 relay handoff recap rows + 300 hybrid web_fetch/web_search tool-call steps (8-tool DSH)
    # - 1,800 bilingual daily_chat rows (~50:50 agent vs general chat balance!)
    final_rows: list[dict[str, object]] = []
    bucket_counts = Counter()

    for seg0, seg1 in pure_web_rows:
        swapped = _swap_to_web_only_system(seg0, rng)
        if swapped:
            row = _make_strict_row(swapped, seg1)
            if row:
                final_rows.append(row)
                bucket_counts["dsh_web_only_pure_web"] += 1
        if rng.random() < 0.30:
            row_orig = _make_strict_row(seg0, seg1)
            if row_orig:
                final_rows.append(row_orig)
                bucket_counts["dsh_8tool_pure_web"] += 1

    for seg0, seg1 in direct_rows:
        swapped = _swap_to_web_only_system(seg0, rng)
        if swapped:
            row = _make_strict_row(swapped, seg1)
            if row:
                final_rows.append(row)
                bucket_counts["dsh_web_only_direct"] += 1

    rng.shuffle(relay_rows)
    for seg0, seg1 in relay_rows[:250]:
        row = _make_strict_row(seg0, seg1)
        if row:
            final_rows.append(row)
            bucket_counts["dsh_relay_recap"] += 1

    rng.shuffle(hybrid_web_doc_rows)
    for seg0, seg1 in hybrid_web_doc_rows[:300]:
        row = _make_strict_row(seg0, seg1)
        if row:
            final_rows.append(row)
            bucket_counts["dsh_hybrid_web_docs"] += 1

    rng.shuffle(daily_chat_rows)
    for seg0, seg1 in daily_chat_rows[:1800]:
        row = _make_strict_row(seg0, seg1)
        if row:
            final_rows.append(row)
            bucket_counts["dsh_daily_chat"] += 1

    # DP4 length-grouped stochastic batching (`-dp4` recipe):
    # Sort rows by estimated token length, partition into contiguous groups of 8,
    # and shuffle the groups so each 4-GPU step has near-zero straggler `reduce_wait_s`
    # while preserving stochastic cross-domain gradient mixing!
    rng.shuffle(final_rows)
    final_rows.sort(
        key=lambda r: estimate_rwkv_tokens(r["segments"][0]["text"] + r["segments"][1]["text"])  # type: ignore[index]
    )
    group_size = 8
    groups = [final_rows[i : i + group_size] for i in range(0, len(final_rows), group_size)]
    rng.shuffle(groups)
    final_rows = [row for grp in groups for row in grp]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    total_est_tokens = 0
    total_sup_tokens = 0
    max_est_tokens = 0

    with train_path.open("w", encoding="utf-8") as fh:
        for row in final_rows:
            segs = row["segments"]  # type: ignore[index]
            # Strict validation matching C++ `dataset.cpp`
            assert list(row.keys()) == ["segments"]
            assert len(segs) == 3
            assert segs[0]["train"] is False and "end_target" not in segs[0]
            assert segs[1]["train"] is True and segs[1]["end_target"] == 0 and len(segs[1]["text"]) > 0
            assert segs[2]["train"] is False and segs[2]["text"] == "<EOD>"
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            fh.write(line + "\n")
            t_all = estimate_rwkv_tokens(segs[0]["text"] + segs[1]["text"]) + 4
            t_sup = estimate_rwkv_tokens(segs[1]["text"]) + 1
            total_est_tokens += t_all
            total_sup_tokens += t_sup
            if t_all > max_est_tokens:
                max_est_tokens = t_all

    sha256 = hashlib.file_digest(train_path.open("rb"), "sha256").hexdigest()
    summary = {
        "name": "dsh-searchreader-native-v1",
        "format": "rwkv-lightning-segments-eos-target-v2",
        "rows": len(final_rows),
        "bytes": train_path.stat().st_size,
        "sha256": sha256,
        "estimated_total_tokens": total_est_tokens,
        "estimated_supervised_tokens": total_sup_tokens,
        "max_estimated_row_tokens": max_est_tokens,
        "buckets": dict(bucket_counts),
        "ingest_stats": dict(stats),
        "heldout_patterns_checked": len(heldout),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/chase/.gemini/antigravity-cli/brain/8dae0695-ec4f-4680-9c3f-97e168ede018/scratch/dsh-searchreader-native-v1"),
    )
    parser.add_argument("--max-ctx", type=int, default=20480)
    args = parser.parse_args()
    summary = build_dataset(args.out, max_ctx_tokens=args.max_ctx)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
