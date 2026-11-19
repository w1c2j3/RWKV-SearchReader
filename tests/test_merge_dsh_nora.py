from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import merge_dsh_nora  # noqa: E402


class _FakeStorageClass:
    def __init__(self, name: str) -> None:
        self._name = name


class _SyntheticPthPickler(pickle.Pickler):
    def reducer_override(self, obj: object):
        if isinstance(obj, _FakeStorageClass):
            return self._load_global(obj._name)
        return NotImplemented

    @staticmethod
    def _load_global(name: str):
        raise AssertionError(name)


def _float32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    return ((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype("<u2")


def _bf16_u16_to_float32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _build_pth_bytes(
    path: Path,
    tensors: list[tuple[str, str, str, int, int, tuple[int, ...], tuple[int, ...]]],
    storages: dict[str, bytes],
    extra_entries: dict[str, bytes] | None = None,
) -> None:
    """Build a valid uncompressed PyTorch-style ZIP64 .pth archive without importing torch."""
    storage_cls_map = {
        "bfloat16": ("torch", "BFloat16Storage"),
        "float16": ("torch", "HalfStorage"),
        "float32": ("torch", "FloatStorage"),
    }

    # Manually construct pickle stream with persistent_id tuples + torch._utils._rebuild_tensor_v2
    buf = io.BytesIO()

    class _BuilderPickler(pickle.Pickler):
        def persistent_id(self, obj: object) -> object:
            if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "__storage_pid__":
                return obj[1]
            return None

    # Create stub globals in fake modules temporarily so pickle writes exact GLOBAL opcodes
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch_utils = types.ModuleType("torch._utils")

    class BFloat16Storage:
        pass

    class HalfStorage:
        pass

    class FloatStorage:
        pass

    def _rebuild_tensor_v2(*args: object) -> object:
        return args

    BFloat16Storage.__module__ = "torch"
    BFloat16Storage.__qualname__ = "BFloat16Storage"
    HalfStorage.__module__ = "torch"
    HalfStorage.__qualname__ = "HalfStorage"
    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"
    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"

    fake_torch.BFloat16Storage = BFloat16Storage  # type: ignore[attr-defined]
    fake_torch.HalfStorage = HalfStorage  # type: ignore[attr-defined]
    fake_torch.FloatStorage = FloatStorage  # type: ignore[attr-defined]
    fake_torch_utils._rebuild_tensor_v2 = _rebuild_tensor_v2  # type: ignore[attr-defined]

    cls_lookup = {
        "bfloat16": BFloat16Storage,
        "float16": HalfStorage,
        "float32": FloatStorage,
    }

    class _TensorProxy:
        def __init__(self, pid: tuple[object, ...], offset: int, shape: tuple[int, ...], stride: tuple[int, ...]) -> None:
            self.pid = pid
            self.offset = offset
            self.shape = shape
            self.stride = stride

        def __reduce__(self):
            return (
                _rebuild_tensor_v2,
                (("__storage_pid__", self.pid), self.offset, self.shape, self.stride, False, OrderedDict()),
            )

    saved_torch = sys.modules.get("torch")
    saved_utils = sys.modules.get("torch._utils")
    sys.modules["torch"] = fake_torch
    sys.modules["torch._utils"] = fake_torch_utils
    try:
        state = OrderedDict()
        for name, dtype, skey, ssize, soffset, shape, stride in tensors:
            _ = storage_cls_map[dtype]
            pid = ("storage", cls_lookup[dtype], skey, "cpu", ssize)
            state[name] = _TensorProxy(pid, soffset, shape, stride)
        p = _BuilderPickler(buf, protocol=2)
        p.dump(state)
    finally:
        if saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved_torch
        if saved_utils is None:
            sys.modules.pop("torch._utils", None)
        else:
            sys.modules["torch._utils"] = saved_utils

    pkl_bytes = buf.getvalue()
    writer = merge_dsh_nora._StoredZip64Writer(path)
    try:
        writer.add_entry("archive/byteorder", 6, [b"little"])
        writer.add_entry("archive/data.pkl", len(pkl_bytes), [pkl_bytes])
        for skey, raw in storages.items():
            writer.add_entry(f"archive/data/{skey}", len(raw), [raw])
        for ename, raw in (extra_entries or {}).items():
            writer.add_entry(f"archive/{ename}", len(raw), [raw])
        writer.finish()
    finally:
        writer.close()


def test_merge_with_numpy_zip64_bf16_packed_storage(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    # Two tensors sharing storage "0" (non-overlapping slices) + one untouched tensor in storage "1"
    w0_f32 = rng.standard_normal((8, 6), dtype=np.float32) * 0.5
    w1_f32 = rng.standard_normal((4, 6), dtype=np.float32) * 0.5
    emb_f32 = rng.standard_normal((5, 6), dtype=np.float32)

    w0_u16 = _float32_to_bf16_u16(w0_f32)
    w1_u16 = _float32_to_bf16_u16(w1_f32)
    emb_u16 = _float32_to_bf16_u16(emb_f32)

    storage0 = w0_u16.tobytes() + w1_u16.tobytes()
    storage1 = emb_u16.tobytes()

    base_path = tmp_path / "base.pth"
    _build_pth_bytes(
        base_path,
        tensors=[
            ("blocks.0.att.receptance.weight", "bfloat16", "0", 48 + 24, 0, (8, 6), (6, 1)),
            ("blocks.0.ffn.value.weight", "bfloat16", "0", 48 + 24, 48, (4, 6), (6, 1)),
            ("emb.weight", "bfloat16", "1", 30, 0, (5, 6), (6, 1)),
        ],
        storages={"0": storage0, "1": storage1},
    )
    base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()

    rank = 2
    b0_u16 = _float32_to_bf16_u16(rng.standard_normal((8, rank), dtype=np.float32) * 0.25)
    a0_u16 = _float32_to_bf16_u16(rng.standard_normal((rank, 6), dtype=np.float32) * 0.25)
    b1_u16 = _float32_to_bf16_u16(rng.standard_normal((4, rank), dtype=np.float32) * 0.25)
    a1_u16 = _float32_to_bf16_u16(rng.standard_normal((rank, 6), dtype=np.float32) * 0.25)

    nora_manifest = {
        "format_version": 1,
        "method": "nora",
        "kind": "inference_adapter",
        "dtype": "bfloat16",
        "layout": "low_rank_ba",
        "rank": rank,
        "alpha": float(rank),
        "scale": 1.0,
        "base_fingerprint": base_sha,
        "targets": [
            {
                "layer": 0,
                "target": "att.receptance.weight",
                "name": "blocks.0.att.receptance.weight.B",
                "a_name": "blocks.0.att.receptance.weight.A",
                "in_features": 6,
                "shape": [8, rank],
                "a_shape": [rank, 6],
            },
            {
                "layer": 0,
                "target": "ffn.value.weight",
                "name": "blocks.0.ffn.value.weight.B",
                "a_name": "blocks.0.ffn.value.weight.A",
                "in_features": 6,
                "shape": [4, rank],
                "a_shape": [rank, 6],
            },
        ],
    }

    adapter_path = tmp_path / "adapter.pth"
    _build_pth_bytes(
        adapter_path,
        tensors=[
            ("blocks.0.att.receptance.weight.B", "bfloat16", "0", 16, 0, (8, rank), (rank, 1)),
            ("blocks.0.att.receptance.weight.A", "bfloat16", "1", 12, 0, (rank, 6), (6, 1)),
            ("blocks.0.ffn.value.weight.B", "bfloat16", "2", 8, 0, (4, rank), (rank, 1)),
            ("blocks.0.ffn.value.weight.A", "bfloat16", "3", 12, 0, (rank, 6), (6, 1)),
        ],
        storages={
            "0": b0_u16.tobytes(),
            "1": a0_u16.tobytes(),
            "2": b1_u16.tobytes(),
            "3": a1_u16.tobytes(),
        },
        extra_entries={"nora.json": json.dumps(nora_manifest).encode("utf-8")},
    )

    out_path = tmp_path / "merged.pth"
    merge_dsh_nora.merge_with_numpy_zip64(base_path, adapter_path, out_path, scale_override=None)
    assert out_path.exists()

    import zipfile

    with zipfile.ZipFile(out_path, "r") as zf:
        root, recs = merge_dsh_nora._parse_pth_archive(zf)
        assert set(recs.keys()) == {
            "blocks.0.att.receptance.weight",
            "blocks.0.ffn.value.weight",
            "emb.weight",
        }
        merged_s0 = zf.read(f"{root}/data/0")
        merged_s1 = zf.read(f"{root}/data/1")

    expected_w0 = _float32_to_bf16_u16(
        _bf16_u16_to_float32(w0_u16)
        + (_bf16_u16_to_float32(b0_u16) @ _bf16_u16_to_float32(a0_u16))
    )
    expected_w1 = _float32_to_bf16_u16(
        _bf16_u16_to_float32(w1_u16)
        + (_bf16_u16_to_float32(b1_u16) @ _bf16_u16_to_float32(a1_u16))
    )
    assert merged_s0[:96] == expected_w0.tobytes()
    assert merged_s0[96:] == expected_w1.tobytes()
    assert merged_s1 == storage1


def test_merge_rejects_mismatched_base_sha256(tmp_path: Path) -> None:
    w = _float32_to_bf16_u16(np.ones((2, 2), dtype=np.float32))
    base_path = tmp_path / "base.pth"
    _build_pth_bytes(
        base_path,
        tensors=[("blocks.0.att.key.weight", "bfloat16", "0", 4, 0, (2, 2), (2, 1))],
        storages={"0": w.tobytes()},
    )
    adapter_path = tmp_path / "adapter.pth"
    nora_manifest = {
        "method": "nora",
        "dtype": "bfloat16",
        "rank": 1,
        "scale": 1.0,
        "base_fingerprint": "0" * 64,
        "targets": [
            {
                "layer": 0,
                "target": "att.key.weight",
                "name": "blocks.0.att.key.weight.B",
                "a_name": "blocks.0.att.key.weight.A",
                "in_features": 2,
                "shape": [2, 1],
                "a_shape": [1, 2],
            }
        ],
    }
    b = _float32_to_bf16_u16(np.ones((2, 1), dtype=np.float32))
    a = _float32_to_bf16_u16(np.ones((1, 2), dtype=np.float32))
    _build_pth_bytes(
        adapter_path,
        tensors=[
            ("blocks.0.att.key.weight.B", "bfloat16", "0", 2, 0, (2, 1), (1, 1)),
            ("blocks.0.att.key.weight.A", "bfloat16", "1", 2, 0, (1, 2), (2, 1)),
        ],
        storages={"0": b.tobytes(), "1": a.tobytes()},
        extra_entries={"nora.json": json.dumps(nora_manifest).encode("utf-8")},
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        merge_dsh_nora.merge_with_numpy_zip64(
            base_path, adapter_path, tmp_path / "out.pth", scale_override=None
        )
