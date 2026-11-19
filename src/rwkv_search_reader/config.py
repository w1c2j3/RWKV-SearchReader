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


def _cf_access_headers(client_id: str, client_secret: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if client_id:
        headers["CF-Access-Client-Id"] = client_id
    if client_secret:
        headers["CF-Access-Client-Secret"] = client_secret
    return headers


def _split_urls(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip().rstrip("/")
            for item in str(value or "").replace("\n", ",").split(",")
            if item.strip()
        )
    )


def _first_deepseek_key() -> str:
    """Return one active DeepSeek API key for /v1/tools/search and /v1/tools/fetch."""

    for name in ("DEEPSEEK_API_KEY", "SEARCH_READER_DEEPSEEK_API_KEY"):
        direct = _text(name).strip("'\"")
        if direct:
            return direct
    return ""


def _first_tavily_key() -> str:
    """Return one active Tavily key; never activate a credential pool."""

    direct = _text("TAVILY_API_KEY").strip("'\"")
    if direct:
        return direct
    raw = _text("TAVILY_API_KEYS")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw.replace("\n", ",").split(",")
        if not isinstance(parsed, (list, tuple)):
            parsed = [parsed]
        for item in parsed:
            candidate = str(item or "").strip().strip("'\"")
            if candidate:
                return candidate
    return ""


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
    search_provider: str = "deepseek"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_need_content: bool = True
    tavily_api_key: str = ""
    tavily_api_keys: tuple[str, ...] = ()
    wigolo_base_url: str = "http://127.0.0.1:3333"
    wigolo_api_token: str = ""
    searxng_base_url: str = ""
    search_max_results: int = 10
    fetch_pages: int = 10
    research_plan_enabled: bool = False
    research_plan_max_queries: int = 4
    review_max_rounds: int = 1
    review_max_queries: int = 3
    conversation_turns: int = 100
    request_timeout_seconds: int = 45
    api_max_concurrent_requests: int = 256
    api_queue_timeout_seconds: int = 120
    api_rate_limit_per_minute: int = 1200

    router_base_url: str = ""
    router_api_key: str = "local"
    router_model: str = ""
    router_protocol: str = "completions"
    router_timeout_seconds: int = 20
    router_cf_access_client_id: str = ""
    router_cf_access_client_secret: str = ""
    router_adapter_id: str = ""
    router_state_id: str = ""

    execution_mode: str = "pipeline"
    pi_agent_max_turns: int = 12
    pi_epoch_budget_tokens: int = 9600

    rwkv_base_url: str = "http://127.0.0.1:18430/v1"
    rwkv_api_key: str = "local"
    rwkv_model: str = ""
    rwkv_protocol: str = "completions"
    rwkv_context_window: int = 34304
    rwkv_sweet_spot_tokens: int = 9600
    rwkv_thinking_mode: str = "auto"
    rwkv_adapter_id: str = ""
    rwkv_adapter_version: str = ""
    rwkv_adapter_scale: float = 1.0
    rwkv_state_id: str = ""
    rwkv_cf_access_client_id: str = ""
    rwkv_cf_access_client_secret: str = ""
    rwkv_summary_workers: int = 12
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
    rwkv_final_cf_access_client_id: str = ""
    rwkv_final_cf_access_client_secret: str = ""
    model_timeout_seconds: int = 180
    rwkv_max_calls_per_request: int = 600

    chunk_chars: int = 2600
    chunk_overlap_chars: int = 260
    max_evidence_chars: int = 12000
    document_max_bytes: int = 10_000_000
    document_pdf_max_bytes: int = 20_000_000
    document_pdf_max_pages: int = 200
    document_chunk_chars: int = 2400
    document_max_chunks: int = 12
    document_full_read_max_chunks: int = 400
    api_host: str = "0.0.0.0"
    api_port: int = 8790
    trace_enabled: bool = True
    trace_directory: Path = PROJECT_ROOT / "logs"
    trace_retention_days: int = 30
    trace_max_total_bytes: int = 1_000_000_000

    @property
    def rwkv_base_urls(self) -> tuple[str, ...]:
        return _split_urls(self.rwkv_base_url)

    @property
    def final_base_urls(self) -> tuple[str, ...]:
        return _split_urls(self.final_base_url)

    @property
    def router_base_urls(self) -> tuple[str, ...]:
        return _split_urls(self.router_base_url)

    @property
    def rwkv_configured(self) -> bool:
        return bool(self.rwkv_base_urls and self.rwkv_model)

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

    @property
    def rwkv_access_headers(self) -> dict[str, str]:
        return _cf_access_headers(
            self.rwkv_cf_access_client_id,
            self.rwkv_cf_access_client_secret,
        )

    @property
    def final_access_headers(self) -> dict[str, str]:
        if self.rwkv_final_cf_access_client_id or self.rwkv_final_cf_access_client_secret:
            return _cf_access_headers(
                self.rwkv_final_cf_access_client_id,
                self.rwkv_final_cf_access_client_secret,
            )
        return self.rwkv_access_headers

    @property
    def router_access_headers(self) -> dict[str, str]:
        return _cf_access_headers(
            self.router_cf_access_client_id,
            self.router_cf_access_client_secret,
        )

    def readiness(self) -> dict[str, object]:
        provider_ready = {
            "deepseek": bool(self.deepseek_api_key),
            "tavily": bool(self.tavily_api_keys or self.tavily_api_key),
            "wigolo": bool(self.wigolo_base_url),
            "searxng": bool(self.searxng_base_url),
        }.get(self.search_provider, False)
        return {
            "ready": bool(provider_ready and self.rwkv_configured),
            "retrieval_mode": "web-or-attached-documents",
            "search_provider": self.search_provider,
            "search_configured": provider_ready,
            "reader_mode": (
                "dsh-rwkv-wire-think-v2.0"
                if self.execution_mode == "dsh_agent"
                else "rwkv-pi-xml-continuation-v3"
                if self.execution_mode == "pi_agent"
                else "rwkv-pure-continuation-parallel-reader"
            ),
            "execution_mode": self.execution_mode,
            "thinking_mode": self.rwkv_thinking_mode,
            "context_window": self.rwkv_context_window,
            "sweet_spot_tokens": self.rwkv_sweet_spot_tokens,
            "protocol": self.rwkv_protocol,
            "endpoint_count": len(self.rwkv_base_urls),
            "summary_workers": self.rwkv_summary_workers,
            "summary_max_tokens": self.rwkv_summary_max_tokens,
            "reduce_chunk_chars": self.rwkv_reduce_chunk_chars,
            "reduce_fan_in": self.rwkv_reduce_fan_in,
            "final_material_chars": self.rwkv_final_material_chars,
            "document_full_read_max_chunks": self.document_full_read_max_chunks,
            "review_max_rounds": self.review_max_rounds,
            "max_concurrent_requests": self.api_max_concurrent_requests,
            "max_model_calls_per_request": self.rwkv_max_calls_per_request,
            "research_plan_enabled": self.research_plan_enabled,
            "research_plan_max_queries": self.research_plan_max_queries,
            "rwkv_configured": self.rwkv_configured,
            "final_model": self.final_model,
            "final_model_configured": bool(self.final_base_urls and self.final_model),
            "answer_max_tokens": self.rwkv_max_tokens,
        }


def _continuation_protocol(raw: str, *, allow_empty: bool = False) -> str:
    value = raw.casefold()
    if allow_empty and not value:
        return ""
    if value in {"lightning", "batch_completions", "batch"}:
        return "lightning"
    return "completions"


def get_settings() -> Settings:
    provider = _text("SEARCH_READER_SEARCH_PROVIDER", "deepseek").casefold()
    if provider not in {"deepseek", "tavily", "wigolo", "searxng"}:
        provider = "deepseek"
    protocol = _continuation_protocol(_text("SEARCH_READER_RWKV_PROTOCOL", "completions"))
    final_protocol = _continuation_protocol(
        _text("SEARCH_READER_RWKV_FINAL_PROTOCOL"), allow_empty=True
    )
    router_protocol = _continuation_protocol(
        _text("SEARCH_READER_ROUTER_PROTOCOL", "completions")
    )
    execution_mode = _text("SEARCH_READER_EXECUTION_MODE", "pipeline").casefold()
    if execution_mode == "dsh":
        execution_mode = "dsh_agent"
    if execution_mode not in {"pipeline", "pi_agent", "dsh_agent"}:
        execution_mode = "pipeline"
    thinking_mode = _text("SEARCH_READER_RWKV_THINKING_MODE", "auto").casefold()
    if thinking_mode not in {"off", "auto", "on"}:
        thinking_mode = "auto"
    chunk_chars = _integer("SEARCH_READER_CHUNK_CHARS", 2600, 600, 12000)
    deepseek_api_key = _first_deepseek_key()
    tavily_api_key = _first_tavily_key()
    return Settings(
        search_provider=provider,
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=_text(
            "SEARCH_READER_DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ).rstrip("/"),
        deepseek_need_content=_boolean("SEARCH_READER_DEEPSEEK_NEED_CONTENT", True),
        tavily_api_key=tavily_api_key,
        tavily_api_keys=(tavily_api_key,) if tavily_api_key else (),
        wigolo_base_url=_text("SEARCH_READER_WIGOLO_BASE_URL", "http://127.0.0.1:3333"),
        wigolo_api_token=_text("SEARCH_READER_WIGOLO_API_TOKEN"),
        searxng_base_url=_text("SEARCH_READER_SEARXNG_BASE_URL"),
        search_max_results=_integer("SEARCH_READER_SEARCH_MAX_RESULTS", 8, 1, 10),
        fetch_pages=_integer("SEARCH_READER_FETCH_PAGES", 8, 1, 10),
        research_plan_enabled=_boolean("SEARCH_READER_RESEARCH_PLAN_ENABLED", True),
        research_plan_max_queries=_integer(
            "SEARCH_READER_RESEARCH_PLAN_MAX_QUERIES", 4, 1, 6
        ),
        review_max_rounds=_integer("SEARCH_READER_REVIEW_MAX_ROUNDS", 1, 0, 10),
        review_max_queries=_integer("SEARCH_READER_REVIEW_MAX_QUERIES", 3, 1, 4),
        conversation_turns=_integer("SEARCH_READER_CONVERSATION_TURNS", 100, 1, 100),
        request_timeout_seconds=_integer(
            "SEARCH_READER_REQUEST_TIMEOUT_SECONDS", 45, 5, 180
        ),
        api_max_concurrent_requests=_integer(
            "SEARCH_READER_API_MAX_CONCURRENT_REQUESTS", 256, 1, 2048
        ),
        api_queue_timeout_seconds=_integer(
            "SEARCH_READER_API_QUEUE_TIMEOUT_SECONDS", 120, 1, 600
        ),
        api_rate_limit_per_minute=_integer(
            "SEARCH_READER_API_RATE_LIMIT_PER_MINUTE", 1200, 1, 100000
        ),
        router_base_url=_text("SEARCH_READER_ROUTER_BASE_URL"),
        router_api_key=_text("SEARCH_READER_ROUTER_API_KEY", "local"),
        router_model=_text("SEARCH_READER_ROUTER_MODEL"),
        router_protocol=router_protocol,
        router_timeout_seconds=_integer("SEARCH_READER_ROUTER_TIMEOUT_SECONDS", 20, 1, 120),
        router_cf_access_client_id=_text("SEARCH_READER_ROUTER_CF_ACCESS_CLIENT_ID"),
        router_cf_access_client_secret=_text("SEARCH_READER_ROUTER_CF_ACCESS_CLIENT_SECRET"),
        router_adapter_id=_text("SEARCH_READER_ROUTER_ADAPTER_ID"),
        router_state_id=_text("SEARCH_READER_ROUTER_STATE_ID"),
        execution_mode=execution_mode,
        pi_agent_max_turns=_integer("SEARCH_READER_PI_AGENT_MAX_TURNS", 12, 2, 48),
        pi_epoch_budget_tokens=_integer(
            "SEARCH_READER_PI_EPOCH_BUDGET_TOKENS", 9600, 2000, 65536
        ),
        rwkv_base_url=_text("SEARCH_READER_RWKV_BASE_URL", "http://127.0.0.1:18430/v1"),
        rwkv_api_key=_text("SEARCH_READER_RWKV_API_KEY", "local"),
        rwkv_model=_text("SEARCH_READER_RWKV_MODEL"),
        rwkv_protocol=protocol,
        rwkv_context_window=_integer(
            "SEARCH_READER_RWKV_CONTEXT_WINDOW", 34304, 4096, 262144
        ),
        rwkv_sweet_spot_tokens=_integer(
            "SEARCH_READER_RWKV_SWEET_SPOT_TOKENS", 9600, 2000, 32768
        ),
        rwkv_thinking_mode=thinking_mode,
        rwkv_adapter_id=_text("SEARCH_READER_RWKV_ADAPTER_ID"),
        rwkv_adapter_version=_text("SEARCH_READER_RWKV_ADAPTER_VERSION"),
        rwkv_adapter_scale=_number("SEARCH_READER_RWKV_ADAPTER_SCALE", 1.0, 0.0, 4.0),
        rwkv_state_id=_text("SEARCH_READER_RWKV_STATE_ID"),
        rwkv_cf_access_client_id=_text("SEARCH_READER_RWKV_CF_ACCESS_CLIENT_ID"),
        rwkv_cf_access_client_secret=_text("SEARCH_READER_RWKV_CF_ACCESS_CLIENT_SECRET"),
        rwkv_summary_workers=_integer("SEARCH_READER_RWKV_SUMMARY_WORKERS", 12, 1, 32),
        rwkv_summary_max_tokens=_integer(
            "SEARCH_READER_RWKV_SUMMARY_MAX_TOKENS", 1024, 96, 2048
        ),
        rwkv_reduce_chunk_chars=_integer(
            "SEARCH_READER_RWKV_REDUCE_CHUNK_CHARS", 6000, 1200, 16000
        ),
        rwkv_reduce_fan_in=_integer("SEARCH_READER_RWKV_REDUCE_FAN_IN", 4, 2, 8),
        rwkv_final_material_chars=_integer(
            "SEARCH_READER_RWKV_FINAL_MATERIAL_CHARS", 9000, 3000, 24000
        ),
        rwkv_max_tokens=_integer("SEARCH_READER_RWKV_MAX_TOKENS", 4090, 128, 4090),
        rwkv_temperature=_number("SEARCH_READER_RWKV_TEMPERATURE", 0.1, 0.0, 2.0),
        rwkv_final_base_url=_text("SEARCH_READER_RWKV_FINAL_BASE_URL"),
        rwkv_final_api_key=_text("SEARCH_READER_RWKV_FINAL_API_KEY"),
        rwkv_final_model=_text("SEARCH_READER_RWKV_FINAL_MODEL"),
        rwkv_final_protocol=final_protocol,
        rwkv_final_cf_access_client_id=_text("SEARCH_READER_RWKV_FINAL_CF_ACCESS_CLIENT_ID"),
        rwkv_final_cf_access_client_secret=_text("SEARCH_READER_RWKV_FINAL_CF_ACCESS_CLIENT_SECRET"),
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
            "SEARCH_READER_DOCUMENT_MAX_BYTES", 10_000_000, 10_000, 30_000_000
        ),
        document_pdf_max_bytes=_integer(
            "SEARCH_READER_DOCUMENT_PDF_MAX_BYTES",
            20_000_000,
            100_000,
            50_000_000,
        ),
        document_pdf_max_pages=_integer(
            "SEARCH_READER_DOCUMENT_PDF_MAX_PAGES", 200, 1, 1000
        ),
        document_chunk_chars=_integer(
            "SEARCH_READER_DOCUMENT_CHUNK_CHARS", 2400, 600, 8000
        ),
        document_max_chunks=_integer("SEARCH_READER_DOCUMENT_MAX_CHUNKS", 12, 1, 30),
        document_full_read_max_chunks=_integer(
            "SEARCH_READER_DOCUMENT_FULL_READ_MAX_CHUNKS", 400, 12, 440
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
