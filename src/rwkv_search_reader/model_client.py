from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from .models import SearchReaderError

LIGHTNING_STOP_SEQUENCES: tuple[str, ...] = (
    "<EOD>",
    "✿",
    "\nUser:",
    "\n### User",
    "\nSystem:",
    "\nAssistant:",
)
PI_STOP_SEQUENCES: tuple[str, ...] = (
    "<EOD>",
    "✿",
    "\nUser:",
    "\n### User",
)
STANDARD_STOP_SEQUENCES: list[str] = [
    "<EOD>",
    "✿",
    "\nUser:",
    "\n### User",
    "\nSystem:",
    "\nAssistant:",
]

_LIGHTNING_LOCK = threading.Lock()
_LIGHTNING_BATCH_URLS: set[str] = set()
_NO_TOKEN_COUNT_URLS: set[str] = set()
_POOL_LOCK = threading.Lock()
_POOLED_SESSIONS: dict[tuple[str, int], requests.Session] = {}


class _PooledSession(requests.Session):
    """Keep-alive session whose close() preserves pooled TCP sockets."""

    def close(self) -> None:
        # Sockets remain pooled for subsequent pure-continuation requests.
        pass

    def force_close(self) -> None:
        super().close()


def _shared_session(base_url: str) -> requests.Session:
    key = (base_url, threading.get_ident())
    with _POOL_LOCK:
        session = _POOLED_SESSIONS.get(key)
        if session is None:
            session = _PooledSession()
            session.trust_env = False
            adapter = HTTPAdapter(pool_connections=16, pool_maxsize=32)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            _POOLED_SESSIONS[key] = session
        return session


def _strip_stop_suffix(value: str, matched_stop: str | None = None) -> str:
    text = str(value or "")
    if matched_stop and text.endswith(matched_stop):
        text = text[: -len(matched_stop)]
    changed = True
    while changed:
        changed = False
        for stop in LIGHTNING_STOP_SEQUENCES:
            if text.endswith(stop):
                text = text[: -len(stop)]
                changed = True
    return text


def _content_from_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise SearchReaderError("model", "model response contains no choices")
    choice = choices[0]
    matched_stop = (
        choice.get("matched_stop")
        if isinstance(choice.get("matched_stop"), str)
        else None
    )
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content: Any = message.get("content")
    if content is None or content == "":
        content = choice.get("text") or ""
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in content
        )
    value = _strip_stop_suffix(str(content), matched_stop).strip()
    if not value:
        raise SearchReaderError("model", "model returned an empty answer")
    return value


def _lightning_wire_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
    thinking: bool = False,
    has_tools: bool = False,
    stop: tuple[str, ...] | list[str] | None = None,
    adapter_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stop_list = (
        list(stop)
        if stop is not None
        else list(PI_STOP_SEQUENCES if (thinking or has_tools) else LIGHTNING_STOP_SEQUENCES)
    )
    wire: dict[str, Any] = {
        "model": model,
        "contents": [prompt],
        "stream": stream,
        "max_tokens": max_tokens,
        "force_reasoning": False,
        "stop": stop_list,
        "chunk_size": 6,
        "metrics": True,
    }
    if has_tools:
        wire.update(
            {
                "temperature": 0.96,
                "top_p": 0.76,
                "top_k": 32,
                "alpha_presence": 0.0,
                "alpha_frequency": 0.0,
                "alpha_decay": 0.996,
            }
        )
    elif thinking:
        wire.update(
            {
                "temperature": 0.96,
                "top_p": 0.76,
                "top_k": 32,
                "alpha_presence": 1.0,
                "alpha_frequency": 0.1,
                "alpha_decay": 0.988,
            }
        )
    elif temperature <= 0.0:
        wire.update({"temperature": 1.0, "top_k": 1, "top_p": 1.0})
        if max_tokens >= 256:
            wire.update(
                {
                    "alpha_presence": 0.35,
                    "alpha_frequency": 0.35,
                    "alpha_decay": 0.996,
                }
            )
    else:
        wire.update(
            {
                "temperature": temperature,
                "top_p": 0.28,
                "top_k": 32,
                "alpha_presence": 0.5,
                "alpha_frequency": 0.5,
                "alpha_decay": 0.996,
            }
        )
    if adapter_config:
        for key in ("adapter_id", "adapter_version", "adapter_scale", "state_id"):
            val = adapter_config.get(key)
            if val is not None and val != "":
                wire[key] = val
    return wire


class OpenAICompatibleClient:
    """Pure-continuation client supporting OpenAI /completions and RWKV Lightning /batch/completions."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int,
        extra_headers: dict[str, str] | None = None,
        context_window: int | None = None,
        adapter_config: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.session = _shared_session(self.base_url)
        self.extra_headers = dict(extra_headers or {})
        self.context_window = context_window
        self.adapter_config = dict(adapter_config or {})
        self.on_request = None

    def _notify_request(self, **event: Any) -> None:
        if self.on_request is not None:
            try:
                self.on_request(event)
            except Exception:
                pass  # A visualization must not change model request behavior.

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key or 'local'}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _is_lightning_endpoint(self) -> bool:
        with _LIGHTNING_LOCK:
            return self.base_url in _LIGHTNING_BATCH_URLS

    def _mark_lightning_endpoint(self) -> None:
        with _LIGHTNING_LOCK:
            _LIGHTNING_BATCH_URLS.add(self.base_url)

    def count_tokens(self, text: str) -> int | None:
        """Measure prompt length with Lightning's native /v1/tokens/count endpoint."""

        with _LIGHTNING_LOCK:
            if self.base_url in _NO_TOKEN_COUNT_URLS:
                return None
        try:
            response = self.session.post(
                f"{self.base_url}/tokens/count",
                headers=self.headers,
                json={"text": text},
                timeout=(5, 15),
            )
        except requests.RequestException:
            return None
        if response.status_code == 404:
            with _LIGHTNING_LOCK:
                _NO_TOKEN_COUNT_URLS.add(self.base_url)
            return None
        if response.status_code >= 400:
            return None
        try:
            data = json.loads(response.content.decode("utf-8"), strict=False)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        for key in ("count", "token_count", "tokens"):
            val = data.get(key)
            if isinstance(val, int) and val >= 1:
                return val
        return None

    def stop_request(self, request_id: str) -> bool:
        """Cancel a single in-flight Lightning request ID without global server stop."""

        if not request_id or request_id == "rwkv-continuation":
            return False
        try:
            resp = self.session.post(
                f"{self.base_url}/server/stop",
                headers=self.headers,
                json={"request_id": request_id},
                timeout=(3, 5),
            )
            return resp.status_code < 400
        except Exception:  # noqa: BLE001 - best-effort cancellation
            return False

    def _bound_max_tokens(self, prompt: str, max_tokens: int, stage: str) -> int:
        if self.context_window is None:
            return max_tokens
        if len(prompt) < 4000 and max_tokens < 1024:
            return min(max_tokens, self.context_window - 64)
        count = self.count_tokens(prompt)
        if count is None:
            return max_tokens
        available = self.context_window - count
        if available < 1:
            raise SearchReaderError(
                stage,
                f"RWKV context exhausted ({count} >= {self.context_window}); input truncation is disabled",
            )
        return min(max_tokens, available)

    def _post_raw(self, path: str, payload: dict[str, Any], stage: str) -> Any:
        response = None
        for attempt in range(3):
            self._notify_request(state="running", attempt=attempt + 1)
            try:
                response = self.session.post(
                    f"{self.base_url}{path}",
                    headers=self.headers,
                    json=payload,
                    timeout=(10, self.timeout_seconds),
                )
            except requests.ConnectionError as exc:
                if attempt == 2:
                    raise SearchReaderError(
                        stage, f"model request failed after reconnect retries: {exc}"
                    ) from exc
                self._notify_request(state="retrying", attempt=attempt + 1)
                time.sleep(0.5 * (2**attempt))
                continue
            except requests.RequestException as exc:
                raise SearchReaderError(stage, f"model request failed: {exc}") from exc
            if response.status_code not in {502, 503, 504} or attempt == 2:
                break
            self._notify_request(state="retrying", attempt=attempt + 1)
            time.sleep(0.5 * (2**attempt))
        if response is None:  # pragma: no cover - loop always returns or raises
            raise SearchReaderError(stage, "model request did not return a response")
        return response

    def _parse_response(self, response: Any, stage: str) -> str:
        if response.status_code >= 400:
            detail = response.text[:800].replace("\n", " ")
            raise SearchReaderError(
                stage, f"model HTTP {response.status_code}: {detail}"
            )
        try:
            data = json.loads(response.content.decode("utf-8"), strict=False)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SearchReaderError(stage, "model returned invalid UTF-8 JSON") from exc
        if not isinstance(data, dict):
            raise SearchReaderError(stage, "model returned a non-object JSON response")
        try:
            return _content_from_response(data)
        except SearchReaderError as exc:
            raise SearchReaderError(stage, str(exc)) from exc

    def _post(self, path: str, payload: dict[str, Any], stage: str) -> str:
        response = self._post_raw(path, payload, stage)
        return self._parse_response(response, stage)

    def stream_answer(
        self,
        *,
        user,
        max_tokens,
        temperature,
        stage,
        protocol="completions",
        thinking: bool = False,
    ):
        """Yield actual upstream text deltas. Never retry a partially emitted answer."""

        use_lightning = protocol in {"lightning", "batch_completions"} or (
            protocol == "completions" and self._is_lightning_endpoint()
        )
        effective_max = (
            min(8192, max_tokens + 1536) if thinking and use_lightning else max_tokens
        )
        if protocol == "chat":
            path = "/chat/completions"
            payload: dict[str, Any] = {
                "model": self.model,
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": user}],
            }
        elif use_lightning:
            path = "/batch/completions"
            prompt_text = build_lightning_prompt("", user, thinking=thinking)
            bounded_max = self._bound_max_tokens(prompt_text, effective_max, stage)
            payload = _lightning_wire_payload(
                model=self.model,
                prompt=prompt_text,
                max_tokens=bounded_max,
                temperature=temperature,
                stream=True,
                thinking=thinking,
                adapter_config=self.adapter_config,
            )
        else:
            path = "/completions"
            payload = {
                "model": self.model,
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "prompt": build_completion_prompt("", user),
                "stop": list(STANDARD_STOP_SEQUENCES),
            }

        reasoning_filter = _ReasoningFilter(initially_hidden=thinking and use_lightning)
        stop_filter = _StopSequenceFilter()
        stream_meta: dict[str, Any] = {"request_id": None, "finished": False}
        try:
            cm = self.session.post(
                self.base_url + path,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=(10, self.timeout_seconds),
            )
            with cm as response:
                if (
                    response.status_code == 404
                    and path == "/completions"
                    and protocol == "completions"
                ):
                    self._mark_lightning_endpoint()
                    path = "/batch/completions"
                    prompt_text = build_lightning_prompt("", user, thinking=thinking)
                    bounded_max = self._bound_max_tokens(
                        prompt_text,
                        min(8192, max_tokens + 1536) if thinking else max_tokens,
                        stage,
                    )
                    payload = _lightning_wire_payload(
                        model=self.model,
                        prompt=prompt_text,
                        max_tokens=bounded_max,
                        temperature=temperature,
                        stream=True,
                        thinking=thinking,
                        adapter_config=self.adapter_config,
                    )
                    reasoning_filter = _ReasoningFilter(initially_hidden=thinking)
                    with self.session.post(
                        self.base_url + path,
                        headers=self.headers,
                        json=payload,
                        stream=True,
                        timeout=(10, self.timeout_seconds),
                    ) as retry_resp:
                        yield from self._consume_stream(
                            retry_resp,
                            protocol=protocol,
                            stage=stage,
                            reasoning_filter=reasoning_filter,
                            stop_filter=stop_filter,
                            stream_meta=stream_meta,
                        )
                    return

                yield from self._consume_stream(
                    response,
                    protocol=protocol,
                    stage=stage,
                    reasoning_filter=reasoning_filter,
                    stop_filter=stop_filter,
                    stream_meta=stream_meta,
                )
        except SearchReaderError:
            raise
        except (requests.RequestException, OSError, ValueError, UnicodeError) as exc:
            raise SearchReaderError(stage, "model stream failed") from exc
        finally:
            if (
                not stream_meta.get("finished")
                and stream_meta.get("request_id")
                and path == "/batch/completions"
            ):
                self.stop_request(str(stream_meta["request_id"]))

    @staticmethod
    def _consume_stream(
        response: Any,
        *,
        protocol: str,
        stage: str,
        reasoning_filter: "_ReasoningFilter",
        stop_filter: "_StopSequenceFilter",
        allow_length: bool = False,
        stream_meta: dict[str, Any] | None = None,
    ):
        if response.status_code >= 400:
            raise SearchReaderError(stage, f"model HTTP {response.status_code}")
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise SearchReaderError(stage, "model did not return an event stream")
        finished = False
        matched_stop: str | None = None
        for raw in response.iter_lines(chunk_size=65536):
            if not raw or not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip().decode("utf-8")
            if data == "[DONE]":
                finished = True
                if stream_meta is not None:
                    stream_meta["finished"] = True
                break
            item = json.loads(data, strict=False)
            if stream_meta is not None and isinstance(item.get("id"), str):
                stream_meta["request_id"] = item["id"]
            if item.get("error"):
                raise SearchReaderError(stage, "model stream returned an error")
            for choice in item.get("choices", []):
                if isinstance(choice.get("matched_stop"), str):
                    matched_stop = choice["matched_stop"]
                delta_obj = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                value = (
                    delta_obj.get("content", "")
                    if protocol == "chat" or "delta" in choice
                    else choice.get("text", "")
                )
                if value:
                    clean = reasoning_filter.feed(value)
                    if clean:
                        safe = stop_filter.feed(clean)
                        if safe:
                            yield safe
                reason = choice.get("finish_reason")
                if reason and reason != "stop" and not (allow_length and reason == "length"):
                    raise SearchReaderError(
                        stage, "model stream ended before a complete answer"
                    )
        if not finished:
            raise SearchReaderError(stage, "model stream disconnected before DONE")
        tail_reason = reasoning_filter.finish()
        if tail_reason:
            tail_safe = stop_filter.feed(tail_reason)
            if tail_safe:
                yield tail_safe
        final_tail = stop_filter.finish(matched_stop)
        if final_tail:
            yield final_tail

    def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        stage: str,
    ) -> str:
        # Deliberately no tools/function-calling fields in this payload.
        messages = [{"role": "user", "content": user}]
        if system.strip():
            messages.insert(0, {"role": "system", "content": system})
        return self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            stage,
        )

    def raw_continuation(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: tuple[str, ...] | list[str] | None = None,
        thinking: bool = True,
        has_tools: bool = True,
        stage: str = "pi_agent",
        on_raw_delta: Callable[[str], None] | None = None,
    ) -> tuple[str, str, str | None, int | None]:
        """Execute a raw rwkv-pi / DSH continuation over SSE, returning (raw_text, reason, matched_stop, prompt_tokens)."""

        prompt_utf8_len = len(prompt.encode("utf-8"))
        if (
            self.context_window is not None
            and prompt_utf8_len + max_tokens < self.context_window - 512
        ):
            # 1 UTF-8 byte is an upper bound on RWKV BPE token count; skip synchronous /tokens/count RTT.
            prompt_tokens = max(1, prompt_utf8_len // 3)
            bounded_max = max_tokens
        else:
            prompt_tokens = self.count_tokens(prompt)
            if self.context_window is not None and prompt_tokens is not None:
                available = self.context_window - prompt_tokens
                if available < 1:
                    raise SearchReaderError(
                        stage,
                        f"RWKV context exhausted ({prompt_tokens} >= {self.context_window}); input truncation is disabled",
                    )
                bounded_max = min(max_tokens, available)
            else:
                bounded_max = max_tokens
        payload = _lightning_wire_payload(
            model=self.model,
            prompt=prompt,
            max_tokens=bounded_max,
            temperature=0.96,
            stream=True,
            thinking=thinking,
            has_tools=has_tools,
            stop=stop,
            adapter_config=self.adapter_config,
        )
        request_id: str | None = None
        finished = False
        finish_reason = "stop"
        matched_stop: str | None = None
        parts: list[str] = []
        try:
            with self.session.post(
                f"{self.base_url}/batch/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=(10, min(90, self.timeout_seconds)),
            ) as response:
                if response.status_code >= 400:
                    raise SearchReaderError(
                        stage, f"model HTTP {response.status_code}: {response.text[:400]}"
                    )
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    data = json.loads(response.content.decode("utf-8"), strict=False)
                    choice = (data.get("choices") or [{}])[0]
                    msg = choice.get("message") or {}
                    raw_text = str(msg.get("content") or choice.get("text") or "")
                    finish_reason = str(choice.get("finish_reason") or "stop")
                    ms = choice.get("matched_stop")
                    matched_stop = ms if isinstance(ms, str) else None
                    finished = True
                    if raw_text and on_raw_delta is not None:
                        on_raw_delta(raw_text)
                    return raw_text, finish_reason, matched_stop, prompt_tokens
                for raw in response.iter_lines(chunk_size=65536):
                    if not raw or not raw.startswith(b"data:"):
                        continue
                    data_str = raw[5:].strip().decode("utf-8")
                    if data_str == "[DONE]":
                        finished = True
                        break
                    item = json.loads(data_str, strict=False)
                    if isinstance(item.get("id"), str):
                        request_id = item["id"]
                    if item.get("error"):
                        raise SearchReaderError(stage, f"model stream error: {item['error']}")
                    for choice in item.get("choices", []):
                        if isinstance(choice.get("matched_stop"), str):
                            matched_stop = choice["matched_stop"]
                        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                        piece = delta.get("content")
                        if piece:
                            piece_str = str(piece)
                            parts.append(piece_str)
                            if on_raw_delta is not None:
                                on_raw_delta(piece_str)
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                if not finished:
                    raise SearchReaderError(stage, "model stream disconnected before DONE")
        except SearchReaderError:
            raise
        except (requests.RequestException, OSError, ValueError, UnicodeError) as exc:
            raise SearchReaderError(stage, f"raw continuation failed: {exc}") from exc
        finally:
            if not finished and request_id:
                self.stop_request(request_id)
        return "".join(parts), finish_reason, matched_stop, prompt_tokens

    def lightning_completion(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        stage: str,
        thinking: bool = False,
    ) -> str:
        prompt = build_lightning_prompt(system, user, thinking=thinking)
        effective_max = min(8192, max_tokens + 1536) if thinking else max_tokens
        bounded_max = self._bound_max_tokens(prompt, effective_max, stage)
        if bounded_max > 512 or thinking:
            stream_payload = _lightning_wire_payload(
                model=self.model,
                prompt=prompt,
                max_tokens=bounded_max,
                temperature=temperature,
                stream=True,
                thinking=thinking,
                adapter_config=self.adapter_config,
            )
            last_exc: Exception | None = None
            for attempt in range(3):
                self._notify_request(state="running", attempt=attempt + 1)
                stream_meta: dict[str, Any] = {"request_id": None, "finished": False}
                try:
                    cm = self.session.post(
                        f"{self.base_url}/batch/completions",
                        headers=self.headers,
                        json=stream_payload,
                        stream=True,
                        timeout=(10, min(75, self.timeout_seconds)),
                    )
                    with cm as response:
                        if "text/event-stream" not in response.headers.get(
                            "Content-Type", ""
                        ):
                            stream_meta["finished"] = True
                            answer = self._parse_response(response, stage)
                            if thinking:
                                answer = _strip_reasoning_markup("<think" + answer)
                            return _strip_reasoning_markup(answer)
                        chunks = list(
                            self._consume_stream(
                                response,
                                protocol="lightning",
                                stage=stage,
                                reasoning_filter=_ReasoningFilter(
                                    initially_hidden=thinking
                                ),
                                stop_filter=_StopSequenceFilter(),
                                allow_length=True,
                                stream_meta=stream_meta,
                            )
                        )
                    value = _strip_stop_suffix("".join(chunks)).strip()
                    if not value:
                        raise SearchReaderError(stage, "model returned an empty answer")
                    return value
                except (requests.ConnectionError, requests.Timeout) as exc:
                    last_exc = exc
                    if attempt == 2:
                        raise SearchReaderError(
                            stage,
                            f"model request failed after reconnect retries: {exc}",
                        ) from exc
                    self._notify_request(state="retrying", attempt=attempt + 1)
                    time.sleep(0.5 * (2**attempt))
                except requests.RequestException as exc:
                    raise SearchReaderError(
                        stage, f"model request failed: {exc}"
                    ) from exc
                finally:
                    if not stream_meta.get("finished") and stream_meta.get("request_id"):
                        self.stop_request(str(stream_meta["request_id"]))
            if last_exc is not None:  # pragma: no cover
                raise SearchReaderError(stage, str(last_exc)) from last_exc
        payload = _lightning_wire_payload(
            model=self.model,
            prompt=prompt,
            max_tokens=bounded_max,
            temperature=temperature,
            stream=False,
            thinking=False,
            adapter_config=self.adapter_config,
        )
        answer = self._post("/batch/completions", payload, stage)
        return _strip_reasoning_markup(answer)

    def completion(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        stage: str,
        thinking: bool = False,
    ) -> str:
        if self._is_lightning_endpoint():
            return self.lightning_completion(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                stage=stage,
                thinking=thinking,
            )
        prompt = build_completion_prompt(system, user)
        response = self._post_raw(
            "/completions",
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["\nUser:", "\nSystem:", "\nAssistant:"],
            },
            stage,
        )
        if response.status_code == 404:
            self._mark_lightning_endpoint()
            return self.lightning_completion(
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                stage=stage,
                thinking=thinking,
            )
        answer = self._parse_response(response, stage)
        return _strip_reasoning_markup(answer)


def build_completion_prompt(system: str, user: str) -> str:
    """Canonical RWKV completion bytes, shared with state training."""
    safe_system = _escape_role_headers(system)
    safe_user = _escape_role_headers(user)
    prompt = f"User: {safe_user}\n\nAssistant: <think></think>"
    return f"System: {safe_system}\n\n{prompt}" if safe_system.strip() else prompt


def build_lightning_prompt(system: str, user: str, *, thinking: bool = False) -> str:
    """Canonical RWKV-7 G1L OpenThink / FakeThink pure-continuation prompt."""
    if not thinking:
        return build_completion_prompt(system, user) + "\n"
    safe_system = _escape_role_headers(system)
    safe_user = _escape_role_headers(user)
    prompt = f"User: {safe_user}\n\nAssistant: <think"
    return f"System: {safe_system}\n\n{prompt}" if safe_system.strip() else prompt


def _escape_role_headers(value: str) -> str:
    lines = []
    for line in str(value or "").splitlines():
        if line.lstrip().startswith(("System:", "User:", "Assistant:")):
            line = "[external text] " + line
        lines.append(line)
    return "\n".join(lines)


def _strip_reasoning_markup(value: str) -> str:
    """Remove reasoning markup without exposing hidden prefixes."""
    answer = _strip_stop_suffix(str(value or "")).strip()
    if "</think>" in answer.casefold():
        answer = re.split(r"</think\s*>", answer, flags=re.IGNORECASE)[-1].strip()
    answer = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    return _strip_stop_suffix(answer).strip()


class _StopSequenceFilter:
    """Suppress trailing stop sequences even when split across SSE chunks."""

    def __init__(self, stops: tuple[str, ...] = LIGHTNING_STOP_SEQUENCES) -> None:
        self.stops = stops
        self.buffer = ""
        self.stopped = False

    def feed(self, text: str) -> str:
        if self.stopped or not text:
            return ""
        self.buffer += text
        earliest = -1
        for stop in self.stops:
            idx = self.buffer.find(stop)
            if idx >= 0 and (earliest < 0 or idx < earliest):
                earliest = idx
        if earliest >= 0:
            out = self.buffer[:earliest]
            self.buffer = ""
            self.stopped = True
            return out
        keep = 0
        for stop in self.stops:
            max_k = min(len(stop) - 1, len(self.buffer))
            for size in range(max_k, keep, -1):
                if self.buffer.endswith(stop[:size]):
                    keep = size
                    break
        if keep:
            out = self.buffer[:-keep]
            self.buffer = self.buffer[-keep:]
            return out
        out = self.buffer
        self.buffer = ""
        return out

    def finish(self, matched_stop: str | None = None) -> str:
        if self.stopped:
            self.buffer = ""
            return ""
        out = _strip_stop_suffix(self.buffer, matched_stop)
        self.buffer = ""
        return out


class _ReasoningFilter:
    """Suppress think blocks even when tags cross network chunks (FakeThink & OpenThink)."""

    def __init__(self, initially_hidden: bool = False):
        self.buffer = ""
        self.hidden = initially_hidden
        self._trim_post_think_newline = False

    def feed(self, text):
        self.buffer += text
        output = ""
        while self.buffer:
            if self._trim_post_think_newline:
                if self.buffer.startswith("\r\n"):
                    self.buffer = self.buffer[2:]
                    self._trim_post_think_newline = False
                    continue
                if self.buffer.startswith("\n"):
                    self.buffer = self.buffer[1:]
                    self._trim_post_think_newline = False
                    continue
                if self.buffer == "\r":
                    break
                self._trim_post_think_newline = False
            lower = self.buffer.lower()
            tag = "</think>" if self.hidden else "<think>"
            index = lower.find(tag)
            if index >= 0:
                if not self.hidden:
                    output += self.buffer[:index]
                else:
                    self._trim_post_think_newline = True
                self.buffer = self.buffer[index + len(tag) :]
                self.hidden = not self.hidden
                continue
            keep = 0
            for size in range(1, min(len(tag), len(lower) + 1)):
                if lower.endswith(tag[:size]):
                    keep = size
            safe = self.buffer[:-keep] if keep else self.buffer
            if not self.hidden:
                output += safe
            self.buffer = self.buffer[-keep:] if keep else ""
            break
        return output

    def finish(self):
        if self.hidden or self.buffer.lower().startswith("<think"):
            raise SearchReaderError("finalize", "model ended inside reasoning markup")
        if self._trim_post_think_newline:
            self.buffer = self.buffer.lstrip("\r\n")
            self._trim_post_think_newline = False
        return self.buffer
