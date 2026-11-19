#!/usr/bin/env python3
"""Download, verify, and offline-merge the DSH-native NoRA adapter into RWKV-7 G1L.

Supports two merge backends:
1. `rwkv_peft_weight_merge` (streaming pure-CPU ZIP64 merger from `rwkv-lightning`, ~15s, 0 GPU/PyTorch needed)
2. `torch` fallback (`torch.load` / `torch.save`, if `rwkv_peft_weight_merge` is not installed)

Optionally runs `rwkv_quantize --format w8a16` afterwards to produce a 7.2 GB `.rwkvq`
weight file capable of `--decode-batch-size 256` per 24 GB GPU (1,024 slots across 4× RTX 4090).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

DEFAULT_RELEASE_URL = (
    "https://github.com/w1c2j3/RWKV-SearchReader/releases/download/"
    "v0.2.0-dsh-nora/dsh-searchreader-g1l-r32-adapter.pth"
)
EXPECTED_ADAPTER_SHA256 = (
    "212e08c4abd2ddea24f6281b2e1ad9af3e4dc04cdfd5045e285e8be943312934"
)
EXPECTED_BASE_SHA256 = (
    "ce81c2d568bc9fc8c9f7b8ddb47da070d88bf179827b311f221e1e6c5148af91"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_adapter(url: str, dest: Path, expected_sha256: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        actual = sha256_file(dest)
        if not expected_sha256 or actual == expected_sha256:
            print(f"[ok] Cached adapter verified: {dest} (sha256={actual[:16]}...)")
            return dest
        print(f"[warn] Cached adapter SHA256 mismatch ({actual}), re-downloading...")
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=300) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            out.write(buf)
            done += len(buf)
            if total > 0 and (done % (16 << 20) == 0 or done == total):
                pct = done * 100.0 / total
                print(f"  {done / (1 << 20):.1f} / {total / (1 << 20):.1f} MiB ({pct:.1f}%)")
    actual = sha256_file(tmp)
    if expected_sha256 and actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded adapter SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    tmp.replace(dest)
    print(f"[ok] Adapter verified: sha256={actual}")
    return dest


def merge_with_torch(base_path: Path, adapter_path: Path, output_path: Path, scale_override: float | None) -> None:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Neither `rwkv_peft_weight_merge` binary nor Python `torch` was found. "
            "Install `rwkv_peft_weight_merge` (from rwkv-lightning) or `pip install torch`."
        ) from exc

    with zipfile.ZipFile(adapter_path, "r") as zf:
        nora_entries = [n for n in zf.namelist() if n.endswith("/nora.json") or n == "nora.json"]
        if not nora_entries:
            raise RuntimeError(f"nora.json manifest not found inside {adapter_path}")
        manifest = json.loads(zf.read(nora_entries[0]).decode("utf-8"))

    base_fp = manifest.get("base_fingerprint") or manifest.get("base")
    print(f"[check] Verifying base model SHA256 ({base_path})...")
    actual_base_fp = sha256_file(base_path)
    if base_fp and actual_base_fp != base_fp:
        raise RuntimeError(
            f"Base model SHA256 mismatch: adapter expects {base_fp}, got {actual_base_fp}"
        )

    scale = float(scale_override if scale_override is not None else manifest.get("scale", 1.0))
    print(f"[merge:torch] Loading base checkpoint {base_path} ...")
    state = torch.load(str(base_path), map_location="cpu", weights_only=True)
    adapter_tensors = torch.load(str(adapter_path), map_location="cpu", weights_only=True)

    targets = manifest.get("targets") or []
    for idx, t in enumerate(targets, 1):
        name = t["name"] if isinstance(t, dict) else str(t)
        a_key = t.get("a", f"{name}.nora_a") if isinstance(t, dict) else f"{name}.nora_a"
        b_key = t.get("b", f"{name}.nora_b") if isinstance(t, dict) else f"{name}.nora_b"
        if a_key not in adapter_tensors:
            a_key = f"targets.{idx - 1}.a"
            b_key = f"targets.{idx - 1}.b"
        w = state[name]
        orig_dtype = w.dtype
        a = adapter_tensors[a_key].to(torch.float32)
        b = adapter_tensors[b_key].to(torch.float32)
        merged = (w.to(torch.float32) + scale * (b @ a)).to(orig_dtype)
        state[name] = merged
        if idx % 32 == 0 or idx == len(targets):
            print(f"  [{idx}/{len(targets)}] {name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(output_path))
    print(f"[ok] Saved merged checkpoint: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download & merge the trained DSH-native NoRA adapter into RWKV-7 G1L."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        type=Path,
        help="Path to unquantized base rwkv7-g1l-7.2b-20261031-ctx34304.pth",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for merged .pth checkpoint (must not already exist)",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Local path to dsh-searchreader-g1l-r32-adapter.pth (auto-downloaded from GitHub Releases if omitted)",
    )
    parser.add_argument(
        "--adapter-url",
        default=DEFAULT_RELEASE_URL,
        help="Release asset URL when --adapter is omitted",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip expected adapter SHA256 verification (for custom-trained adapters)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional NoRA merge scale override (default: manifest scale = 1.0)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="CPU thread count for rwkv_peft_weight_merge (default: 16)",
    )
    parser.add_argument(
        "--merge-bin",
        default=shutil.which("rwkv_peft_weight_merge") or "",
        help="Path to rwkv_peft_weight_merge binary (falls back to PyTorch if absent)",
    )
    parser.add_argument(
        "--quantize-w8a16",
        type=Path,
        default=None,
        help="Optional output .rwkvq path to run `rwkv_quantize --format w8a16` after merging",
    )
    parser.add_argument(
        "--quantize-bin",
        default=shutil.which("rwkv_quantize") or "",
        help="Path to rwkv_quantize binary",
    )
    args = parser.parse_args()

    if not args.base_model.exists():
        print(f"error: base model not found: {args.base_model}", file=sys.stderr)
        return 1

    adapter_path = args.adapter
    if adapter_path is None:
        cache_dir = Path("adapters/dsh-searchreader-g1l-r32")
        adapter_path = download_adapter(
            args.adapter_url,
            cache_dir / "dsh-searchreader-g1l-r32-adapter.pth",
            "" if args.skip_sha256 else EXPECTED_ADAPTER_SHA256,
        )
    elif not args.skip_sha256:
        actual = sha256_file(adapter_path)
        if actual != EXPECTED_ADAPTER_SHA256:
            print(
                f"error: adapter SHA256 mismatch ({actual} != {EXPECTED_ADAPTER_SHA256}); pass --skip-sha256 for custom checkpoints",
                file=sys.stderr,
            )
            return 1

    if not args.output.exists():
        if args.merge_bin and Path(args.merge_bin).exists():
            cmd = [
                str(args.merge_bin),
                "--method",
                "nora",
                "--model",
                str(args.base_model),
                "--adapter",
                str(adapter_path),
                "--output",
                str(args.output),
                "--threads",
                str(args.threads),
            ]
            if args.scale is not None:
                cmd.extend(["--scale", str(args.scale)])
            print("[merge:cpp]", " ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            merge_with_torch(args.base_model, adapter_path, args.output, args.scale)
    else:
        print(f"[skip] Merged .pth already exists: {args.output}")

    if args.quantize_w8a16 is not None:
        if not args.quantize_bin or not Path(args.quantize_bin).exists():
            print(
                "error: --quantize-w8a16 requested but `rwkv_quantize` binary not found; pass --quantize-bin",
                file=sys.stderr,
            )
            return 1
        q_cmd = [
            str(args.quantize_bin),
            "--format",
            "w8a16",
            str(args.output),
            str(args.quantize_w8a16),
        ]
        print("[quantize:w8a16]", " ".join(q_cmd))
        subprocess.run(q_cmd, check=True)
        print(f"[ok] Saved W8A16 quantized model: {args.quantize_w8a16}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
