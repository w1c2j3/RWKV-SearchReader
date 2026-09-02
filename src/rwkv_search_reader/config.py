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
    for filename in (".env.local", ".env"):
        _load_local_env(
            PROJECT_ROOT.parent / "RWKV-ECRA" / filename,
            allowed_keys={"TAVILY_API_KEY", "TAVILY_API_KEYS"},
        )
        if _text("TAVILY_API_KEY") or _text("TAVILY_API_KEYS"):
            break


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
    research_plan_enabled: bool = False
    research_plan_max_queries: int = 4
    review_max_rounds: int = 2
    review_max_queries: int = 3
    conversation_turns: int = 10
    request_timeout_seconds: int = 45
    api_max_concurrent_requests: int = 4
    api_queue_timeout_seconds: int = 20
    api_rate_limit_per_minute: int = 20

    rwkv_base_url: str = "http://127.0.0.1:29613/v1"
    rwkv_api_key: str = "local"
    rwkv_model: str = ""
    rwkv_protocol: str = "completions"
    rwkv_summary_workers: int = 10
    rwkv_summary_max_tokens: int = 1024
    rwkv_reduce_chunk_chars: int = 6000
    rwkv_reduce_fan_in: int = 4
    rwkv_final_material_chars: int = 9000
    rwkv_max_tokens: int = 4090
    rwkv_temperature: float = 0.1
    rwkv_final_base_url: str = ""
    rwkv_final_api_key: str = ""
    rwkv_final_model: str = ""
    rwkv_final_protocol: str = ""
    model_timeout_seconds: int = 180
    rwkv_max_calls_per_request: int = 600

    chunk_chars: int = 2600
    chunk_overlap_chars: int = 260
    max_evidence_chars: int = 12000
    document_max_bytes: int = 2_000_000
    document_chunk_chars: int = 2400
    document_max_chunks: int = 12
    api_host: str = "0.0.0.0"
    api_port: int = 8790
    trace_enabled: bool = True
    trace_directory: Path = PROJECT_ROOT / "logs"
    trace_retention_days: int = 30
    trace_max_total_bytes: int = 1_000_000_000

    @property
    def rwkv_configured(self) -> bool:
        return bool(self.rwkv_base_url and self.rwkv_model)

    @property
    def final_model(self) -> str:
        return self.rwkv_final_model or self.rwkv_model

    @property
    def final_base_url(self) -> str:
        return self.rwkv_final_base_url or self.rwkv_base_url

    @property
    def final_api_key(self) -> str:
        return self.rwkv_final_api_key or self.rwkv_api_key

    @property
    def final_protocol(self) -> str:
        return self.rwkv_final_protocol or self.rwkv_protocol

    def readiness(self) -> dict[str, object]:
        provider_ready = {
            "tavily": bool(self.tavily_api_keys or self.tavily_api_key),
            "wigolo": bool(self.wigolo_base_url),
            "searxng": bool(self.searxng_base_url),
        }.get(self.search_provider, False)
        return {
            "ready": bool(provider_ready and self.rwkv_configured),
            "retrieval_mode": "web-or-attached-documents",
            "search_provider": self.search_provider,
            "search_configured": provider_ready,
            "reader_mode": "rwkv-reviewed-parallel-funnel-final-summary",
            "summary_workers": self.rwkv_summary_workers,
            "summary_max_tokens": self.rwkv_summary_max_tokens,
            "reduce_chunk_chars": self.rwkv_reduce_chunk_chars,
            "reduce_fan_in": self.rwkv_reduce_fan_in,
            "final_material_chars": self.rwkv_final_material_chars,
            "review_max_rounds": self.review_max_rounds,
            "max_concurrent_requests": self.api_max_concurrent_requests,
            "max_model_calls_per_request": self.rwkv_max_calls_per_request,
            "research_plan_enabled": self.research_plan_enabled,
            "research_plan_max_queries": self.research_plan_max_queries,
            "rwkv_configured": self.rwkv_configured,
            "final_model": self.final_model,
            "final_model_configured": bool(self.final_base_url and self.final_model),
            "answer_max_tokens": self.rwkv_max_tokens,
        }


def get_settings() -> Settings:
    provider = _text("SEARCH_READER_SEARCH_PROVIDER", "tavily").casefold()
    if provider not in {"tavily", "wigolo", "searxng"}:
        provider = "tavily"
    protocol = _text("SEARCH_READER_RWKV_PROTOCOL", "completions").casefold()
    if protocol not in {"completions", "chat"}:
        protocol = "completions"
    final_protocol = _text("SEARCH_READER_RWKV_FINAL_PROTOCOL").casefold()
    if final_protocol not in {"", "completions", "chat"}:
        final_protocol = ""
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
        research_plan_enabled=_boolean("SEARCH_READER_RESEARCH_PLAN_ENABLED", True),
        research_plan_max_queries=_integer(
            "SEARCH_READER_RESEARCH_PLAN_MAX_QUERIES", 4, 1, 6
        ),
        review_max_rounds=_integer("SEARCH_READER_REVIEW_MAX_ROUNDS", 2, 0, 3),
        review_max_queries=_integer("SEARCH_READER_REVIEW_MAX_QUERIES", 3, 1, 4),
        conversation_turns=_integer("SEARCH_READER_CONVERSATION_TURNS", 10, 1, 20),
        request_timeout_seconds=_integer(
            "SEARCH_READER_REQUEST_TIMEOUT_SECONDS", 45, 5, 180
        ),
        api_max_concurrent_requests=_integer(
            "SEARCH_READER_API_MAX_CONCURRENT_REQUESTS", 4, 1, 32
        ),
        api_queue_timeout_seconds=_integer(
            "SEARCH_READER_API_QUEUE_TIMEOUT_SECONDS", 20, 1, 300
        ),
        api_rate_limit_per_minute=_integer(
            "SEARCH_READER_API_RATE_LIMIT_PER_MINUTE", 20, 1, 600
        ),
        rwkv_base_url=_text("SEARCH_READER_RWKV_BASE_URL", "http://127.0.0.1:29613/v1"),
        rwkv_api_key=_text("SEARCH_READER_RWKV_API_KEY", "local"),
        rwkv_model=_text("SEARCH_READER_RWKV_MODEL"),
        rwkv_protocol=protocol,
        rwkv_summary_workers=_integer(
            "SEARCH_READER_RWKV_SUMMARY_WORKERS", 10, 1, 32
        ),
        rwkv_summary_max_tokens=_integer(
            "SEARCH_READER_RWKV_SUMMARY_MAX_TOKENS", 1024, 96, 2048
        ),
        rwkv_reduce_chunk_chars=_integer(
            "SEARCH_READER_RWKV_REDUCE_CHUNK_CHARS", 6000, 1200, 12000
        ),
        rwkv_reduce_fan_in=_integer(
            "SEARCH_READER_RWKV_REDUCE_FAN_IN", 4, 2, 8
        ),
        rwkv_final_material_chars=_integer(
            "SEARCH_READER_RWKV_FINAL_MATERIAL_CHARS", 9000, 3000, 12000
        ),
        rwkv_max_tokens=_integer("SEARCH_READER_RWKV_MAX_TOKENS", 4090, 128, 4090),
        rwkv_temperature=_number("SEARCH_READER_RWKV_TEMPERATURE", 0.1, 0.0, 2.0),
        rwkv_final_base_url=_text("SEARCH_READER_RWKV_FINAL_BASE_URL"),
        rwkv_final_api_key=_text("SEARCH_READER_RWKV_FINAL_API_KEY"),
        rwkv_final_model=_text("SEARCH_READER_RWKV_FINAL_MODEL"),
        rwkv_final_protocol=final_protocol,
        model_timeout_seconds=_integer(
            "SEARCH_READER_MODEL_TIMEOUT_SECONDS", 180, 10, 900
        ),
        rwkv_max_calls_per_request=_integer(
            "SEARCH_READER_RWKV_MAX_CALLS_PER_REQUEST", 600, 16, 1000
        ),
        chunk_chars=chunk_chars,
        chunk_overlap_chars=_integer(
            "SEARCH_READER_CHUNK_OVERLAP_CHARS", 260, 0, max(1, chunk_chars // 2)
        ),
        max_evidence_chars=_integer(
            "SEARCH_READER_MAX_EVIDENCE_CHARS", 12000, 2000, 12000
        ),
        document_max_bytes=_integer(
            "SEARCH_READER_DOCUMENT_MAX_BYTES", 2_000_000, 10_000, 10_000_000
        ),
        document_chunk_chars=_integer(
            "SEARCH_READER_DOCUMENT_CHUNK_CHARS", 2400, 600, 8000
        ),
        document_max_chunks=_integer(
            "SEARCH_READER_DOCUMENT_MAX_CHUNKS", 12, 1, 30
        ),
        api_host=_text("SEARCH_READER_API_HOST", "0.0.0.0"),
        api_port=_integer("SEARCH_READER_API_PORT", 8790, 1, 65535),
        trace_enabled=_boolean("SEARCH_READER_TRACE_ENABLED", True),
        trace_directory=Path(
            _text("SEARCH_READER_TRACE_DIRECTORY", str(PROJECT_ROOT / "logs"))
        ).expanduser(),
        trace_retention_days=_integer(
            "SEARCH_READER_TRACE_RETENTION_DAYS", 30, 1, 3650
        ),
        trace_max_total_bytes=_integer(
            "SEARCH_READER_TRACE_MAX_TOTAL_BYTES",
            1_000_000_000,
            10_000_000,
            20_000_000_000,
        ),
    )
