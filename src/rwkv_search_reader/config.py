from __future__ import annotations

import collections.abc
import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_local_env(
    path: Path, allowed_keys: collections.abc.Set[str] | None = None
) -> None:
    """Load an ignored .env.local without overriding the process environment."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            continue
        key = key.strip()
        if allowed_keys is not None and key not in allowed_keys:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(_text(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _number(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(_text(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _boolean(name: str, default: bool) -> bool:
    value = _text(name, "1" if default else "0").casefold()
    return value in {"1", "true", "yes", "on"}


def _key_pool() -> tuple[str, ...]:
    values: list[str] = []

    def add(value: object) -> None:
        candidate = str(value or "").strip().strip("'\"")
        if candidate and candidate not in values:
            values.append(candidate)

    add(_text("TAVILY_API_KEY"))
    raw = _text("TAVILY_API_KEYS")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw.replace("\n", ",").split(",")
        if not isinstance(parsed, (list, tuple)):
            parsed = [parsed]
        for item in parsed:
            add(item)
    return tuple(values)


_load_local_env(PROJECT_ROOT / ".env.local")
if not (_text("TAVILY_API_KEY") or _text("TAVILY_API_KEYS")):
    # Reuse only the local Tavily credentials already kept by RWKV-ECRA.
    _load_local_env(
        PROJECT_ROOT.parent / "RWKV-ECRA" / ".env.local",
        allowed_keys={"TAVILY_API_KEY", "TAVILY_API_KEYS"},
    )


@dataclass(frozen=True, slots=True)
class Settings:
    search_provider: str = "tavily"
    tavily_api_key: str = ""
    tavily_api_keys: tuple[str, ...] = ()
    wigolo_base_url: str = "http://127.0.0.1:3333"
    wigolo_api_token: str = ""
    searxng_base_url: str = ""
    search_max_results: int = 10
    fetch_pages: int = 10
    request_timeout_seconds: int = 45

    rwkv_base_url: str = "http://127.0.0.1:29613/v1"
    rwkv_api_key: str = "local"
    rwkv_model: str = ""
    rwkv_protocol: str = "completions"
    rwkv_summary_workers: int = 10
    rwkv_reduce_orders: int = 5
    rwkv_summary_max_tokens: int = 384
    rwkv_max_tokens: int = 1536
    rwkv_temperature: float = 0.1
    model_timeout_seconds: int = 180

    chunk_chars: int = 2600
    chunk_overlap_chars: int = 260
    max_evidence_chars: int = 18000
    api_host: str = "0.0.0.0"
    api_port: int = 8790
    trace_enabled: bool = True
    trace_directory: Path = PROJECT_ROOT / "logs"

    @property
    def rwkv_configured(self) -> bool:
        return bool(self.rwkv_base_url and self.rwkv_model)

    def readiness(self) -> dict[str, object]:
        provider_ready = {
            "tavily": bool(self.tavily_api_keys or self.tavily_api_key),
            "wigolo": bool(self.wigolo_base_url),
            "searxng": bool(self.searxng_base_url),
        }.get(self.search_provider, False)
        return {
            "ready": bool(provider_ready and self.rwkv_configured),
            "retrieval_mode": "forced",
            "search_provider": self.search_provider,
            "search_configured": provider_ready,
            "reader_mode": "rwkv-parallel-map-reduce",
            "summary_workers": self.rwkv_summary_workers,
            "reduce_orders": self.rwkv_reduce_orders,
            "rwkv_configured": self.rwkv_configured,
        }


def get_settings() -> Settings:
    provider = _text("SEARCH_READER_SEARCH_PROVIDER", "tavily").casefold()
    if provider not in {"tavily", "wigolo", "searxng"}:
        provider = "tavily"
    protocol = _text("SEARCH_READER_RWKV_PROTOCOL", "completions").casefold()
    if protocol not in {"completions", "chat"}:
        protocol = "completions"
    chunk_chars = _integer("SEARCH_READER_CHUNK_CHARS", 2600, 600, 12000)
    return Settings(
        search_provider=provider,
        tavily_api_key=_text("TAVILY_API_KEY"),
        tavily_api_keys=_key_pool(),
        wigolo_base_url=_text("SEARCH_READER_WIGOLO_BASE_URL", "http://127.0.0.1:3333"),
        wigolo_api_token=_text("SEARCH_READER_WIGOLO_API_TOKEN"),
        searxng_base_url=_text("SEARCH_READER_SEARXNG_BASE_URL"),
        search_max_results=_integer("SEARCH_READER_SEARCH_MAX_RESULTS", 10, 1, 10),
        fetch_pages=_integer("SEARCH_READER_FETCH_PAGES", 10, 1, 10),
        request_timeout_seconds=_integer(
            "SEARCH_READER_REQUEST_TIMEOUT_SECONDS", 45, 5, 180
        ),
        rwkv_base_url=_text("SEARCH_READER_RWKV_BASE_URL", "http://127.0.0.1:29613/v1"),
        rwkv_api_key=_text("SEARCH_READER_RWKV_API_KEY", "local"),
        rwkv_model=_text("SEARCH_READER_RWKV_MODEL"),
        rwkv_protocol=protocol,
        rwkv_summary_workers=_integer(
            "SEARCH_READER_RWKV_SUMMARY_WORKERS", 10, 1, 32
        ),
        rwkv_reduce_orders=_integer("SEARCH_READER_RWKV_REDUCE_ORDERS", 5, 1, 10),
        rwkv_summary_max_tokens=_integer(
            "SEARCH_READER_RWKV_SUMMARY_MAX_TOKENS", 384, 96, 2048
        ),
        rwkv_max_tokens=_integer("SEARCH_READER_RWKV_MAX_TOKENS", 1536, 128, 8192),
        rwkv_temperature=_number("SEARCH_READER_RWKV_TEMPERATURE", 0.1, 0.0, 2.0),
        model_timeout_seconds=_integer(
            "SEARCH_READER_MODEL_TIMEOUT_SECONDS", 180, 10, 900
        ),
        chunk_chars=chunk_chars,
        chunk_overlap_chars=_integer(
            "SEARCH_READER_CHUNK_OVERLAP_CHARS", 260, 0, max(1, chunk_chars // 2)
        ),
        max_evidence_chars=_integer(
            "SEARCH_READER_MAX_EVIDENCE_CHARS", 18000, 2000, 80000
        ),
        api_host=_text("SEARCH_READER_API_HOST", "0.0.0.0"),
        api_port=_integer("SEARCH_READER_API_PORT", 8790, 1, 65535),
        trace_enabled=_boolean("SEARCH_READER_TRACE_ENABLED", True),
    )
