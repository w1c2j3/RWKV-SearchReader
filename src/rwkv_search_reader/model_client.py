from __future__ import annotations

import json
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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.trust_env = False

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key or 'local'}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict[str, Any], stage: str) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}{path}",
                headers=self.headers,
                json=payload,
                timeout=(10, self.timeout_seconds),
            )
        except requests.RequestException as exc:
            raise SearchReaderError(stage, f"model request failed: {exc}") from exc
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
        return self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
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
        safe_system = _escape_role_headers(system)
        safe_user = _escape_role_headers(user)
        prompt = f"System: {safe_system}\n\nUser: {safe_user}\n\nAssistant:"
        return self._post(
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


def _escape_role_headers(value: str) -> str:
    lines = []
    for line in str(value or "").splitlines():
        if line.lstrip().startswith(("System:", "User:", "Assistant:")):
            line = "[external text] " + line
        lines.append(line)
    return "\n".join(lines)
