from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from .models import SearchReaderError


def _content_from_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise SearchReaderError("model", "model response contains no choices")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content: Any = message.get("content") or choice.get("text") or ""
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in content
        )
    value = str(content).strip()
    if not value:
        raise SearchReaderError("model", "model returned an empty answer")
    return value


class OpenAICompatibleClient:
    """Small text-only client. It never sends tools or a tool_choice field."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.trust_env = False
        self.extra_headers = dict(extra_headers or {})
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

    def _post(self, path: str, payload: dict[str, Any], stage: str) -> str:
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
        if response.status_code >= 400:
            detail = response.text[:800].replace("\n", " ")
            raise SearchReaderError(
                stage, f"model HTTP {response.status_code}: {detail}"
            )
        try:
            data = json.loads(response.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SearchReaderError(stage, "model returned invalid UTF-8 JSON") from exc
        if not isinstance(data, dict):
            raise SearchReaderError(stage, "model returned a non-object JSON response")
        try:
            return _content_from_response(data)
        except SearchReaderError as exc:
            raise SearchReaderError(stage, str(exc)) from exc

    def stream_answer(self, *, user, max_tokens, temperature, stage, protocol="completions"):
        """Yield actual upstream text deltas. Never retry a partially emitted answer."""
        payload = {"model": self.model, "stream": True, "max_tokens": max_tokens, "temperature": temperature}
        if protocol == "chat":
            path = "/chat/completions"
            payload["messages"] = [{"role": "user", "content": user}]
        else:
            path = "/completions"
            payload["prompt"] = build_completion_prompt("", user)
            payload["stop"] = ["\nUser:", "\nSystem:", "\nAssistant:"]
        filter_ = _ReasoningFilter()
        finished = False
        try:
            with self.session.post(self.base_url + path, headers=self.headers, json=payload,
                                   stream=True, timeout=(10, self.timeout_seconds)) as response:
                if response.status_code >= 400:
                    raise SearchReaderError(stage, f"model HTTP {response.status_code}")
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    raise SearchReaderError(stage, "model did not return an event stream")
                for raw in response.iter_lines(chunk_size=65536):
                    if not raw or not raw.startswith(b"data:"):
                        continue
                    data = raw[5:].strip().decode("utf-8")
                    if data == "[DONE]":
                        finished = True
                        break
                    item = json.loads(data)
                    if item.get("error"):
                        raise SearchReaderError(stage, "model stream returned an error")
                    for choice in item.get("choices", []):
                        reason = choice.get("finish_reason")
                        if reason and reason != "stop":
                            raise SearchReaderError(stage, "model stream ended before a complete answer")
                        value = (choice.get("delta") or {}).get("content", "") if protocol == "chat" else choice.get("text", "")
                        if value:
                            text = filter_.feed(value)
                            if text:
                                yield text
                if not finished:
                    raise SearchReaderError(stage, "model stream disconnected before DONE")
                tail = filter_.finish()
                if tail:
                    yield tail
        except (requests.RequestException, ValueError, UnicodeError) as exc:
            raise SearchReaderError(stage, "model stream failed") from exc


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

    def completion(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        stage: str,
    ) -> str:
        prompt = build_completion_prompt(system, user)
        answer = self._post(
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
        return _strip_reasoning_markup(answer)


def build_completion_prompt(system: str, user: str) -> str:
    """Canonical RWKV completion bytes, shared with state training."""
    safe_system = _escape_role_headers(system)
    safe_user = _escape_role_headers(user)
    prompt = f"User: {safe_user}\n\nAssistant: <think></think>"
    return f"System: {safe_system}\n\n{prompt}" if safe_system.strip() else prompt


def _escape_role_headers(value: str) -> str:
    lines = []
    for line in str(value or "").splitlines():
        if line.lstrip().startswith(("System:", "User:", "Assistant:")):
            line = "[external text] " + line
        lines.append(line)
    return "\n".join(lines)


def _strip_reasoning_markup(value: str) -> str:
    """Remove G1I reasoning markup without exposing the hidden prefix."""
    answer = str(value or "").strip()
    if "</think>" in answer.casefold():
        answer = re.split(r"</think\s*>", answer, flags=re.IGNORECASE)[-1].strip()
    answer = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    return answer


class _ReasoningFilter:
    """Suppress think blocks even when tags cross network chunks."""
    def __init__(self):
        self.buffer = ""
        self.hidden = False

    def feed(self, text):
        self.buffer += text
        output = ""
        while self.buffer:
            lower = self.buffer.lower()
            tag = "</think>" if self.hidden else "<think>"
            index = lower.find(tag)
            if index >= 0:
                if not self.hidden:
                    output += self.buffer[:index]
                self.buffer = self.buffer[index + len(tag):]
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
        return self.buffer
