#!/usr/bin/env python3
"""Download, verify, and offline-merge the DSH-native NoRA adapter into RWKV-7 G1L.

Supports three merge backends (in automatic priority order):
1. `rwkv_peft_weight_merge` (C++ streaming ZIP64 merger from `rwkv-lightning`, if installed)
2. Built-in `numpy` streaming ZIP64 merger (pure CPU, <400 MiB RAM, zero PyTorch / zero C++ build required)
3. `torch` fallback (`torch.load` / `torch.save`)

Also auto-downloads both the official base checkpoint (`BlinkDL/rwkv7-g1`) with HTTP Range
resume and the trained DSH-native NoRA adapter (`chase-jay/rwkv7-g1l-dsh-searchreader-nora`)
with automatic `hf-mirror.com` and GitHub Releases fallback.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
import zlib

BASE_HF_REPO_ID = "BlinkDL/rwkv7-g1"
BASE_FILENAME = "rwkv7-g1l-7.2b-20261031-ctx34304.pth"
EXPECTED_BASE_SHA256 = (
    "ce81c2d568bc9fc8c9f7b8ddb47da070d88bf179827b311f221e1e6c5148af91"
)

HF_REPO_ID = "chase-jay/rwkv7-g1l-dsh-searchreader-nora"
ADAPTER_FILENAME = "dsh-searchreader-g1l-r32-adapter.pth"
EXPECTED_ADAPTER_SHA256 = (
    "212e08c4abd2ddea24f6281b2e1ad9af3e4dc04cdfd5045e285e8be943312934"
)
GITHUB_RELEASE_URL = (
    "https://github.com/w1c2j3/RWKV-SearchReader/releases/download/"
    f"v0.2.0-dsh-nora/{ADAPTER_FILENAME}"
)

VOCAB_FILENAME = "rwkv_vocab_v20230424.txt"
EXPECTED_VOCAB_SHA256 = (
    "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"
)


def _default_vocab_urls() -> list[str]:
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    urls = [f"{hf_endpoint}/BlinkDL/rwkv-7-world/resolve/main/{VOCAB_FILENAME}"]
    if hf_endpoint == "https://huggingface.co":
        urls.append(f"https://hf-mirror.com/BlinkDL/rwkv-7-world/resolve/main/{VOCAB_FILENAME}")
    urls.append("https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v5/rwkv_vocab_v20230424.txt")
    return urls

DEFAULT_BASE_MODEL_PATH = Path("models") / BASE_FILENAME
DEFAULT_MERGED_MODEL_PATH = Path("models/rwkv7-g1l-7.2b-dsh-r32-merged.pth")
DEFAULT_VOCAB_PATH = Path("models") / VOCAB_FILENAME

DTYPE_ITEMSIZE = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


def _default_base_urls(override_url: str = "") -> list[str]:
    if override_url:
        return [override_url]
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    urls = [f"{hf_endpoint}/{BASE_HF_REPO_ID}/resolve/main/{BASE_FILENAME}"]
    if hf_endpoint == "https://huggingface.co":
        urls.append(f"https://hf-mirror.com/{BASE_HF_REPO_ID}/resolve/main/{BASE_FILENAME}")
    return urls


def _default_candidate_urls(override_url: str = "") -> list[str]:
    if override_url:
        return [override_url]
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    urls = [
        f"{hf_endpoint}/{HF_REPO_ID}/resolve/main/{ADAPTER_FILENAME}",
    ]
    if hf_endpoint == "https://huggingface.co":
        urls.append(f"https://hf-mirror.com/{HF_REPO_ID}/resolve/main/{ADAPTER_FILENAME}")
    urls.append(GITHUB_RELEASE_URL)
    return urls


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file_with_resume(
    urls: list[str],
    dest: Path,
    expected_sha256: str,
    label: str = "file",
    progress_step_mib: int = 64,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if not expected_sha256:
            print(f"[ok] Cached {label} found: {dest}")
            return dest
        print(f"[check] Verifying cached {label}: {dest} ...")
        actual = sha256_file(dest)
        if actual == expected_sha256:
            print(f"[ok] Cached {label} verified (sha256={actual[:16]}...)")
            return dest
        print(f"[warn] Cached {label} SHA256 mismatch ({actual}), re-downloading...")
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    step_bytes = max(1, progress_step_mib) << 20

    for url in urls:
        try:
            existing = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": "rwkv-search-reader/0.2"}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                print(f"[download] Resuming {label} at {existing / (1 << 20):.1f} MiB from {url}")
            else:
                print(f"[download] {url} -> {dest}")

            req = urllib.request.Request(url, headers=headers)
            try:
                resp_ctx = urllib.request.urlopen(req, timeout=25)
            except urllib.error.HTTPError as http_err:
                if http_err.code == 416 and existing > 0:
                    # Partial file already reached full size or invalid range; check hash first.
                    actual = sha256_file(tmp)
                    if not expected_sha256 or actual == expected_sha256:
                        tmp.replace(dest)
                        print(f"[ok] {label.capitalize()} verified: sha256={actual}")
                        return dest
                    tmp.unlink(missing_ok=True)
                    existing = 0
                    req = urllib.request.Request(url, headers={"User-Agent": "rwkv-search-reader/0.2"})
                    resp_ctx = urllib.request.urlopen(req, timeout=25)
                else:
                    raise

            with resp_ctx as resp:
                status = getattr(resp, "status", 200)
                if existing > 0 and status == 206:
                    mode = "ab"
                    content_len = int(resp.headers.get("Content-Length") or 0)
                    total = existing + content_len if content_len > 0 else 0
                    done = existing
                else:
                    mode = "wb"
                    total = int(resp.headers.get("Content-Length") or 0)
                    done = 0

                next_report = ((done // step_bytes) + 1) * step_bytes
                with tmp.open(mode) as out:
                    while True:
                        buf = resp.read(4 << 20)
                        if not buf:
                            break
                        out.write(buf)
                        done += len(buf)
                        if total > 0 and (done >= next_report or done == total):
                            pct = done * 100.0 / total
                            print(
                                f"  {done / (1 << 20):.1f} / {total / (1 << 20):.1f} MiB ({pct:.1f}%)",
                                flush=True,
                            )
                            while next_report <= done:
                                next_report += step_bytes

            print(f"[check] Computing SHA256 for {tmp.name} ...")
            actual = sha256_file(tmp)
            if expected_sha256 and actual != expected_sha256:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Downloaded {label} SHA256 mismatch: expected {expected_sha256}, got {actual}"
                )
            tmp.replace(dest)
            print(f"[ok] {label.capitalize()} verified: sha256={actual}")
            return dest
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[warn] Mirror failed ({url}): {exc}", file=sys.stderr)
    raise RuntimeError(f"Failed to download {label} from all mirrors; last error: {last_err}")


def download_adapter(urls: list[str], dest: Path, expected_sha256: str) -> Path:
    return download_file_with_resume(
        urls=urls,
        dest=dest,
        expected_sha256=expected_sha256,
        label="adapter",
        progress_step_mib=16,
    )


# ---------------------------------------------------------------------------
# Safe PyTorch ZIP64 .pth Reader & Streaming Writer (Zero PyTorch / C++ Deps)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StorageTypeStub:
    dtype: str


@dataclass(frozen=True)
class _StorageRef:
    dtype: str
    key: str
    size: int


@dataclass(frozen=True)
class _TensorRecord:
    name: str
    dtype: str
    storage_key: str
    storage_size: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def _rebuild_tensor_stub(
    storage: _StorageRef,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    *_rest: object,
) -> _TensorRecord:
    if not isinstance(storage, _StorageRef):
        raise pickle.UnpicklingError(f"Expected _StorageRef, got {type(storage)!r}")
    shape_t = tuple(int(x) for x in size)
    stride_t = tuple(int(x) for x in stride)
    if len(shape_t) != len(stride_t):
        raise pickle.UnpicklingError("Tensor shape/stride rank mismatch")
    return _TensorRecord(
        name="",
        dtype=storage.dtype,
        storage_key=storage.key,
        storage_size=storage.size,
        storage_offset=int(storage_offset),
        shape=shape_t,
        stride=stride_t,
    )


class _SafePthUnpickler(pickle.Unpickler):
    _STORAGE_MAP = {
        ("torch", "BFloat16Storage"): _StorageTypeStub("bfloat16"),
        ("torch", "HalfStorage"): _StorageTypeStub("float16"),
        ("torch", "FloatStorage"): _StorageTypeStub("float32"),
    }

    def find_class(self, module: str, name: str) -> object:
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) in self._STORAGE_MAP:
            return self._STORAGE_MAP[(module, name)]
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_stub
        raise pickle.UnpicklingError(f"Unsupported pickle global in .pth: {module}.{name}")

    def persistent_load(self, pid: object) -> _StorageRef:
        if (
            not isinstance(pid, tuple)
            or len(pid) < 5
            or pid[0] != "storage"
            or not isinstance(pid[1], _StorageTypeStub)
        ):
            raise pickle.UnpicklingError(f"Unsupported persistent_load id: {pid!r}")
        return _StorageRef(
            dtype=pid[1].dtype,
            key=str(pid[2]),
            size=int(pid[4]),
        )


def _is_contiguous_2d(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    if len(shape) != 2 or len(stride) != 2:
        return False
    rows, cols = shape
    if rows == 0 or cols == 0:
        return True
    if stride[1] != 1 and cols != 1:
        return False
    if stride[0] != cols and rows != 1:
        return False
    return True


def _tensor_byte_span(rec: _TensorRecord) -> tuple[int, int]:
    if len(rec.shape) != len(rec.stride):
        raise RuntimeError(f"Invalid shape/stride rank on tensor {rec.name}")
    last = rec.storage_offset
    for dim, st in zip(rec.shape, rec.stride):
        if dim < 0 or st < 0:
            raise RuntimeError(f"Negative shape/stride on tensor {rec.name}")
        if dim == 0:
            return (0, 0)
        last += (dim - 1) * st
    if last >= rec.storage_size:
        raise RuntimeError(f"Tensor {rec.name} exceeds storage bounds ({last} >= {rec.storage_size})")
    width = DTYPE_ITEMSIZE[rec.dtype]
    return (rec.storage_offset * width, (last + 1) * width)


def _parse_pth_archive(zf: zipfile.ZipFile) -> tuple[str, dict[str, _TensorRecord]]:
    seen_entries: set[str] = set()
    root: str | None = None
    for info in zf.infolist():
        if info.filename in seen_entries:
            raise RuntimeError(f"Duplicate ZIP entry in .pth: {info.filename}")
        seen_entries.add(info.filename)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"Compressed .pth entry is unsupported: {info.filename}")
        if info.filename == "data.pkl":
            if root is not None:
                raise RuntimeError("Ambiguous .pth archive root")
            root = ""
        elif info.filename.endswith("/data.pkl"):
            if root is not None:
                raise RuntimeError("Ambiguous .pth archive root")
            root = info.filename[: -len("/data.pkl")]

    if root is None:
        raise RuntimeError("data.pkl not found in .pth archive")

    byteorder_entry = f"{root}/byteorder" if root else "byteorder"
    if byteorder_entry in seen_entries:
        if zf.read(byteorder_entry) != b"little":
            raise RuntimeError("Only little-endian .pth archives are supported")

    pkl_entry = f"{root}/data.pkl" if root else "data.pkl"
    obj = _SafePthUnpickler(io.BytesIO(zf.read(pkl_entry))).load()
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected state_dict mapping in {pkl_entry}, got {type(obj)!r}")

    records: dict[str, _TensorRecord] = {}
    for k, v in obj.items():
        if isinstance(v, _TensorRecord):
            key_str = str(k)
            records[key_str] = _TensorRecord(
                name=key_str,
                dtype=v.dtype,
                storage_key=v.storage_key,
                storage_size=v.storage_size,
                storage_offset=v.storage_offset,
                shape=v.shape,
                stride=v.stride,
            )
    if not records:
        raise RuntimeError(f"No tensor records parsed from {pkl_entry}")
    return root, records


def _storage_entry_path(root: str, storage_key: str) -> str:
    return f"{root}/data/{storage_key}" if root else f"data/{storage_key}"


class _StoredZip64Writer:
    """Streams uncompressed ZIP64 entries with valid local/central CRCs (matches rwkv_peft_weight_merge)."""

    def __init__(self, path: Path) -> None:
        self._fh = path.open("wb")
        self._entries: list[tuple[bytes, int, int, int]] = []

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def add_entry(self, name: str, size: int, chunks_iter) -> None:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 65535:
            raise RuntimeError(f"ZIP entry name too long: {name}")
        offset = self._fh.tell()
        header = struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,
            45,
            0x0800,
            0,
            0,
            0,
            0,
            0xFFFFFFFF,
            0xFFFFFFFF,
            len(name_bytes),
            20,
        )
        extra = struct.pack("<HHQQ", 0x0001, 16, size, size)
        self._fh.write(header)
        self._fh.write(name_bytes)
        self._fh.write(extra)

        crc = 0
        written = 0
        for chunk in chunks_iter:
            if not chunk:
                continue
            written += len(chunk)
            if written > size:
                raise RuntimeError(f"ZIP entry {name} exceeded declared size {size}")
            crc = zlib.crc32(chunk, crc) & 0xFFFFFFFF
            self._fh.write(chunk)

        if written != size:
            raise RuntimeError(f"ZIP entry {name} shorter than declared size ({written} != {size})")

        end_pos = self._fh.tell()
        self._fh.seek(offset + 14)
        self._fh.write(struct.pack("<I", crc))
        self._fh.seek(end_pos)
        self._entries.append((name_bytes, offset, size, crc))

    def finish(self) -> None:
        cd_start = self._fh.tell()
        for name_bytes, offset, size, crc in self._entries:
            cd_header = struct.pack(
                "<IHHHHHHIIIHHHHHII",
                0x02014B50,
                45,
                45,
                0x0800,
                0,
                0,
                0,
                crc,
                0xFFFFFFFF,
                0xFFFFFFFF,
                len(name_bytes),
                28,
                0,
                0,
                0,
                0,
                0xFFFFFFFF,
            )
            cd_extra = struct.pack("<HHQQQ", 0x0001, 24, size, size, offset)
            self._fh.write(cd_header)
            self._fh.write(name_bytes)
            self._fh.write(cd_extra)

        cd_end = self._fh.tell()
        count = len(self._entries)
        self._fh.write(
            struct.pack(
                "<IQHHIIQQQQ",
                0x06064B50,
                44,
                45,
                45,
                0,
                0,
                count,
                count,
                cd_end - cd_start,
                cd_start,
            )
        )
        self._fh.write(struct.pack("<IIQI", 0x07064B50, 0, cd_end, 1))
        self._fh.write(
            struct.pack(
                "<IHHHHIIH",
                0x06054B50,
                0,
                0,
                0xFFFF,
                0xFFFF,
                0xFFFFFFFF,
                0xFFFFFFFF,
                0,
            )
        )
        self._fh.close()


@dataclass
class _NoraTarget:
    weight_name: str
    out_features: int
    in_features: int
    rank: int
    b: object  # np.ndarray float32 [out, rank]
    a: object  # np.ndarray float32 [rank, in]


def _resolve_target_names(entry: dict[str, object], idx0: int) -> tuple[str, str, str]:
    if "layer" in entry and "target" in entry:
        weight_name = f"blocks.{int(entry['layer'])}.{entry['target']}"  # type: ignore[arg-type]
        b_name = str(entry.get("name") or f"{weight_name}.B")
        a_name = str(entry.get("a_name") or f"{weight_name}.A")
        return weight_name, b_name, a_name
    weight_name = str(entry["name"])
    b_name = str(entry.get("b") or f"{weight_name}.B")
    a_name = str(entry.get("a") or f"{weight_name}.A")
    return weight_name, b_name, a_name


def merge_with_numpy_zip64(
    base_path: Path,
    adapter_path: Path,
    output_path: Path,
    scale_override: float | None,
    skip_base_sha256: bool = False,
) -> None:
    import numpy as np

    def _decode_to_f32(raw: bytes, dtype: str, shape: tuple[int, int]) -> np.ndarray:
        if dtype == "bfloat16":
            u16 = np.frombuffer(raw, dtype="<u2").reshape(shape)
            return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)
        if dtype == "float16":
            return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)
        if dtype == "float32":
            return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        raise RuntimeError(f"Unsupported tensor dtype: {dtype}")

    def _encode_from_f32(arr_f32: np.ndarray, dtype: str, name: str) -> bytes:
        if not bool(np.all(np.isfinite(arr_f32))):
            raise RuntimeError(f"Merged weight contains NaN/Inf: {name}")
        if dtype == "bfloat16":
            bits = np.ascontiguousarray(arr_f32, dtype=np.float32).view(np.uint32)
            lsb = (bits >> np.uint32(16)) & np.uint32(1)
            u16 = ((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype("<u2")
            if bool(np.any((u16 & np.uint16(0x7F80)) == np.uint16(0x7F80))):
                raise RuntimeError(f"Merged weight overflowed bfloat16: {name}")
            return u16.tobytes()
        if dtype == "float16":
            f16 = np.ascontiguousarray(arr_f32, dtype=np.float32).astype("<f2")
            if not bool(np.all(np.isfinite(f16))):
                raise RuntimeError(f"Merged weight overflowed float16: {name}")
            return f16.tobytes()
        if dtype == "float32":
            return np.ascontiguousarray(arr_f32, dtype="<f4").tobytes()
        raise RuntimeError(f"Unsupported tensor dtype: {dtype}")

    with zipfile.ZipFile(adapter_path, "r") as azf:
        a_root, a_records = _parse_pth_archive(azf)
        nora_entry = f"{a_root}/nora.json" if a_root else "nora.json"
        if nora_entry not in azf.namelist():
            raise RuntimeError(f"nora.json manifest not found inside {adapter_path}")
        manifest = json.loads(azf.read(nora_entry).decode("utf-8"))
        if manifest.get("method", "nora") != "nora":
            raise RuntimeError(f"Unsupported adapter method: {manifest.get('method')}")

        base_fp = str(manifest.get("base_fingerprint") or manifest.get("base") or "")
        scale = float(scale_override if scale_override is not None else manifest.get("scale", 1.0))
        master = manifest.get("tensor_source") == "master"

        def _load_factor(key: str, rows: int, cols: int) -> np.ndarray:
            lookup = f"{key}.master" if master and f"{key}.master" in a_records else key
            if lookup not in a_records:
                raise RuntimeError(f"Missing adapter factor tensor: {lookup}")
            rec = a_records[lookup]
            if rec.shape != (rows, cols) or not _is_contiguous_2d(rec.shape, rec.stride):
                raise RuntimeError(f"Mismatched shape/stride for {lookup}: {rec.shape} / {rec.stride}")
            span_begin, span_end = _tensor_byte_span(rec)
            entry_path = _storage_entry_path(a_root, rec.storage_key)
            raw_storage = azf.read(entry_path)
            arr = _decode_to_f32(raw_storage[span_begin:span_end], rec.dtype, (rows, cols))
            if not bool(np.all(np.isfinite(arr))):
                raise RuntimeError(f"Non-finite adapter factor: {lookup}")
            return arr

        raw_targets = manifest.get("targets") or []
        if not raw_targets:
            raise RuntimeError("Adapter manifest contains zero targets")

        targets: list[_NoraTarget] = []
        seen_names: set[str] = set()
        for idx0, item in enumerate(raw_targets):
            if not isinstance(item, dict):
                raise RuntimeError(f"Invalid target entry at index {idx0}")
            weight_name, b_name, a_name = _resolve_target_names(item, idx0)
            if weight_name in seen_names:
                raise RuntimeError(f"Duplicate target in adapter manifest: {weight_name}")
            seen_names.add(weight_name)
            b_shape = item["shape"]
            a_shape = item["a_shape"]
            out_features, rank = int(b_shape[0]), int(b_shape[1])  # type: ignore[index]
            rank_a, in_features = int(a_shape[0]), int(a_shape[1])  # type: ignore[index]
            if rank != rank_a:
                raise RuntimeError(f"Rank mismatch in {weight_name}: {rank} != {rank_a}")
            b_arr = _load_factor(b_name, out_features, rank)
            a_arr = _load_factor(a_name, rank, in_features)
            targets.append(
                _NoraTarget(
                    weight_name=weight_name,
                    out_features=out_features,
                    in_features=in_features,
                    rank=rank,
                    b=b_arr,
                    a=a_arr,
                )
            )

    if base_fp and not skip_base_sha256:
        print(f"[check] Verifying base model SHA256 ({base_path})...")
        actual_base_fp = sha256_file(base_path)
        if actual_base_fp != base_fp:
            raise RuntimeError(
                f"Base model SHA256 mismatch: adapter expects {base_fp}, got {actual_base_fp}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_out = output_path.with_name(f"{output_path.name}.merge-{os.getpid()}.part")
    temp_out.unlink(missing_ok=True)

    try:
        with zipfile.ZipFile(base_path, "r") as bzf:
            b_root, weights = _parse_pth_archive(bzf)
            info_by_name = {info.filename: info for info in bzf.infolist()}

            by_storage_key: dict[str, list[tuple[str, int, int]]] = {}
            for name, rec in weights.items():
                sb, se = _tensor_byte_span(rec)
                by_storage_key.setdefault(rec.storage_key, []).append((name, sb, se))

            replacements: dict[str, list[tuple[_NoraTarget, str, int, int]]] = {}
            for t in targets:
                if t.weight_name not in weights:
                    raise RuntimeError(f"Base model missing target tensor: {t.weight_name}")
                w = weights[t.weight_name]
                if w.shape != (t.out_features, t.in_features) or not _is_contiguous_2d(w.shape, w.stride):
                    raise RuntimeError(
                        f"Base target {t.weight_name} must be contiguous [{t.out_features}, {t.in_features}], "
                        f"got shape={w.shape} stride={w.stride}"
                    )
                sb, se = _tensor_byte_span(w)
                for other_name, osb, ose in by_storage_key.get(w.storage_key, ()):
                    if other_name == t.weight_name:
                        continue
                    if not (se <= osb or ose <= sb):
                        raise RuntimeError(
                            f"Target {t.weight_name} shares overlapping storage with {other_name}"
                        )
                storage_entry = _storage_entry_path(b_root, w.storage_key)
                expected_bytes = w.storage_size * DTYPE_ITEMSIZE[w.dtype]
                if storage_entry not in info_by_name or info_by_name[storage_entry].file_size != expected_bytes:
                    raise RuntimeError(f"Base storage entry size mismatch for {t.weight_name}")
                replacements.setdefault(storage_entry, []).append((t, w.dtype, sb, se))

            for patch_list in replacements.values():
                patch_list.sort(key=lambda item: item[2])

            print(
                f"[merge:numpy-zip64] Streaming {len(targets)} NoRA targets (scale={scale}) -> {output_path}",
                flush=True,
            )
            writer = _StoredZip64Writer(temp_out)
            merged_count = 0
            try:
                for info in bzf.infolist():
                    patches = replacements.get(info.filename)
                    if not patches:
                        def _copy_all(zinfo: zipfile.ZipInfo = info):
                            with bzf.open(zinfo, "r") as src:
                                while True:
                                    buf = src.read(8 << 20)
                                    if not buf:
                                        break
                                    yield buf

                        writer.add_entry(info.filename, info.file_size, _copy_all())
                    else:
                        def _patched_stream(
                            zinfo: zipfile.ZipInfo = info,
                            plist: list[tuple[_NoraTarget, str, int, int]] = patches,
                        ):
                            nonlocal merged_count
                            with bzf.open(zinfo, "r") as src:
                                at = 0
                                for t, dtype, begin, end in plist:
                                    while at < begin:
                                        step = min(8 << 20, begin - at)
                                        chunk = src.read(step)
                                        if not chunk:
                                            raise RuntimeError(f"Unexpected EOF in {zinfo.filename}")
                                        at += len(chunk)
                                        yield chunk
                                    raw_len = end - begin
                                    raw = src.read(raw_len)
                                    if len(raw) != raw_len:
                                        raise RuntimeError(f"Truncated read on {t.weight_name}")
                                    at += raw_len
                                    if scale == 0.0:
                                        yield raw
                                    else:
                                        w_f32 = _decode_to_f32(
                                            raw, dtype, (t.out_features, t.in_features)
                                        )
                                        delta = np.matmul(t.b, t.a)
                                        if scale != 1.0:
                                            delta *= np.float32(scale)
                                        w_f32 += delta
                                        yield _encode_from_f32(w_f32, dtype, t.weight_name)
                                    merged_count += 1
                                    if merged_count % 32 == 0 or merged_count == len(targets):
                                        print(
                                            f"  [{merged_count}/{len(targets)}] {t.weight_name}",
                                            flush=True,
                                        )
                                while at < zinfo.file_size:
                                    step = min(8 << 20, zinfo.file_size - at)
                                    chunk = src.read(step)
                                    if not chunk:
                                        raise RuntimeError(f"Unexpected EOF in {zinfo.filename}")
                                    at += len(chunk)
                                    yield chunk

                        writer.add_entry(info.filename, info.file_size, _patched_stream())
                writer.finish()
            finally:
                writer.close()

            with zipfile.ZipFile(temp_out, "r") as czf:
                _, check_weights = _parse_pth_archive(czf)
                if len(check_weights) != len(weights):
                    raise RuntimeError("Output .pth self-check tensor count mismatch")

        temp_out.replace(output_path)
        print(f"[ok] Saved lossless merged checkpoint: {output_path}")
    except Exception:
        temp_out.unlink(missing_ok=True)
        raise


def merge_with_torch(
    base_path: Path,
    adapter_path: Path,
    output_path: Path,
    scale_override: float | None,
    skip_base_sha256: bool = False,
) -> None:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Neither `rwkv_peft_weight_merge`, `numpy`, nor `torch` was available."
        ) from exc

    with zipfile.ZipFile(adapter_path, "r") as zf:
        nora_entries = [n for n in zf.namelist() if n.endswith("/nora.json") or n == "nora.json"]
        if not nora_entries:
            raise RuntimeError(f"nora.json manifest not found inside {adapter_path}")
        manifest = json.loads(zf.read(nora_entries[0]).decode("utf-8"))

    base_fp = manifest.get("base_fingerprint") or manifest.get("base")
    if base_fp and not skip_base_sha256:
        print(f"[check] Verifying base model SHA256 ({base_path})...")
        actual_base_fp = sha256_file(base_path)
        if actual_base_fp != base_fp:
            raise RuntimeError(
                f"Base model SHA256 mismatch: adapter expects {base_fp}, got {actual_base_fp}"
            )

    scale = float(scale_override if scale_override is not None else manifest.get("scale", 1.0))
    print(f"[merge:torch] Loading base checkpoint {base_path} ...")
    state = torch.load(str(base_path), map_location="cpu", weights_only=True)
    adapter_tensors = torch.load(str(adapter_path), map_location="cpu", weights_only=True)

    targets = manifest.get("targets") or []
    for idx, t in enumerate(targets, 1):
        if isinstance(t, dict):
            weight_name, b_key, a_key = _resolve_target_names(t, idx - 1)
        else:
            weight_name = str(t)
            b_key = f"{weight_name}.B"
            a_key = f"{weight_name}.A"
        w = state[weight_name]
        orig_dtype = w.dtype
        a = adapter_tensors[a_key].to(torch.float32)
        b = adapter_tensors[b_key].to(torch.float32)
        merged = (w.to(torch.float32) + scale * (b @ a)).to(orig_dtype)
        state[weight_name] = merged
        if idx % 32 == 0 or idx == len(targets):
            print(f"  [{idx}/{len(targets)}] {weight_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(output_path))
    print(f"[ok] Saved merged checkpoint: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-download & offline-merge the trained DSH-native NoRA adapter into RWKV-7 G1L."
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path(os.environ.get("BASE_MODEL_PATH", str(DEFAULT_BASE_MODEL_PATH))),
        help=(
            f"Path to unquantized base {BASE_FILENAME} "
            f"(default: {DEFAULT_BASE_MODEL_PATH}; auto-downloaded from {BASE_HF_REPO_ID} if absent)"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_MODEL_URL", ""),
        help=f"Optional explicit base model URL (defaults to Hugging Face {BASE_HF_REPO_ID} with hf-mirror fallback)",
    )
    parser.add_argument(
        "--no-download-base",
        action="store_true",
        help="Fail instead of auto-downloading base model when --base-model does not exist locally",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("MERGED_MODEL_PATH", str(DEFAULT_MERGED_MODEL_PATH))),
        help=f"Output path for merged .pth checkpoint (default: {DEFAULT_MERGED_MODEL_PATH})",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Local path to dsh-searchreader-g1l-r32-adapter.pth (auto-downloaded from Hugging Face if omitted)",
    )
    parser.add_argument(
        "--adapter-url",
        default="",
        help="Optional explicit adapter download URL (defaults to Hugging Face chase-jay/rwkv7-g1l-dsh-searchreader-nora with hf-mirror & GitHub Releases fallback)",
    )
    parser.add_argument(
        "--vocab-output",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help=f"Path to ensure RWKV world vocab exists (default: {DEFAULT_VOCAB_PATH})",
    )
    parser.add_argument(
        "--skip-vocab",
        action="store_true",
        help="Skip downloading rwkv_vocab_v20230424.txt",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip expected SHA256 verification (for custom checkpoints)",
    )
    parser.add_argument(
        "--cleanup-base",
        action="store_true",
        help="Delete downloaded --base-model after verifying merged --output to save 14.4 GB disk space",
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
        default=shutil.which("rwkv_peft_weight_merge")
        or (
            str(Path("lightning/bundle/rwkv_peft_weight_merge"))
            if Path("lightning/bundle/rwkv_peft_weight_merge").exists()
            else ""
        ),
        help="Optional path to rwkv_peft_weight_merge binary (falls back to built-in NumPy ZIP64 merger)",
    )
    parser.add_argument(
        "--quantize-w8a16",
        type=Path,
        default=None,
        help="Optional output .rwkvq path to run `rwkv_quantize --format w8a16` after merging",
    )
    parser.add_argument(
        "--quantize-bin",
        default=shutil.which("rwkv_quantize")
        or (
            str(Path("lightning/bundle/rwkv_quantize"))
            if Path("lightning/bundle/rwkv_quantize").exists()
            else ""
        ),
        help="Path to rwkv_quantize binary",
    )
    args = parser.parse_args()

    if not args.skip_vocab and args.vocab_output is not None:
        download_file_with_resume(
            urls=_default_vocab_urls(),
            dest=args.vocab_output,
            expected_sha256="" if args.skip_sha256 else EXPECTED_VOCAB_SHA256,
            label="vocab",
            progress_step_mib=4,
        )

    if args.output.exists() and args.output.stat().st_size > 0:
        print(f"[skip] Merged .pth already exists: {args.output}")
    else:
        downloaded_base = False
        if not args.base_model.exists():
            if args.no_download_base:
                print(f"error: base model not found: {args.base_model}", file=sys.stderr)
                return 1
            download_file_with_resume(
                urls=_default_base_urls(args.base_url),
                dest=args.base_model,
                expected_sha256="" if args.skip_sha256 else EXPECTED_BASE_SHA256,
                label="base model (14.4 GB)",
                progress_step_mib=256,
            )
            downloaded_base = True

        adapter_path = args.adapter
        if adapter_path is None:
            cache_dir = Path("adapters/dsh-searchreader-g1l-r32")
            adapter_path = download_adapter(
                _default_candidate_urls(args.adapter_url),
                cache_dir / ADAPTER_FILENAME,
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

        if args.merge_bin and Path(args.merge_bin).exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)
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
            try:
                import numpy  # noqa: F401

                has_numpy = True
            except ImportError:
                has_numpy = False

            if has_numpy:
                merge_with_numpy_zip64(
                    args.base_model,
                    adapter_path,
                    args.output,
                    args.scale,
                    skip_base_sha256=args.skip_sha256,
                )
            else:
                merge_with_torch(
                    args.base_model,
                    adapter_path,
                    args.output,
                    args.scale,
                    skip_base_sha256=args.skip_sha256,
                )

        if args.cleanup_base and downloaded_base and args.output.exists():
            print(f"[cleanup] Removing intermediate base checkpoint: {args.base_model}")
            args.base_model.unlink(missing_ok=True)

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
