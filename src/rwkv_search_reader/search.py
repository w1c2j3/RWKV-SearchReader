from __future__ import annotations

import hashlib
import json
import re
import threading
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

import requests

from .config import Settings
from .models import SearchHit, SearchReaderError

_TAVILY_HEALTH_LOCK = threading.Lock()
_TAVILY_PERMANENT_FAILURES: set[str] = set()
_TAVILY_PREFERRED_KEY = ""
_COMMON_GENERIC_TLDS = {
    "ai",
    "app",
    "biz",
    "cloud",
    "com",
    "dev",
    "edu",
    "gov",
    "info",
    "io",
    "me",
    "mil",
    "net",
    "online",
    "org",
    "site",
    "tech",
    "xyz",
}


def _key_identity(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _ordered_tavily_keys(keys: tuple[str, ...]) -> list[str]:
    with _TAVILY_HEALTH_LOCK:
        available = [
            key for key in keys if _key_identity(key) not in _TAVILY_PERMANENT_FAILURES
        ]
        return sorted(
            available,
            key=lambda key: 0 if _key_identity(key) == _TAVILY_PREFERRED_KEY else 1,
        )


def _mark_tavily_failure(api_key: str) -> None:
    with _TAVILY_HEALTH_LOCK:
        _TAVILY_PERMANENT_FAILURES.add(_key_identity(api_key))


def _mark_tavily_success(api_key: str) -> None:
    global _TAVILY_PREFERRED_KEY
    with _TAVILY_HEALTH_LOCK:
        _TAVILY_PREFERRED_KEY = _key_identity(api_key)


def _reset_tavily_health_for_tests() -> None:
    global _TAVILY_PREFERRED_KEY
    with _TAVILY_HEALTH_LOCK:
        _TAVILY_PERMANENT_FAILURES.clear()
        _TAVILY_PREFERRED_KEY = ""


def _clean(value: Any, limit: int = 0) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:limit] if limit else text


def _explicit_domains(query: str) -> list[str]:
    values = re.findall(
        r"(?<![@\w])((?:[a-z0-9-]+\.)+[a-z]{2,})(?![\w])",
        str(query or "").casefold(),
    )
    excluded = {"example.com"}
    return list(
        dict.fromkeys(
            value.removeprefix("www.")
            for value in values
            if value not in excluded
            and (
                len(value.rsplit(".", 1)[-1]) == 2
                or value.rsplit(".", 1)[-1] in _COMMON_GENERIC_TLDS
            )
        )
    )[:5]


def _tavily_topic(query: str) -> str:
    """Use Tavily news mode when publication timestamps are part of the evidence."""

    if re.search(
        r"新闻|今日新闻|今天新闻|刚刚发生|本周新闻|本月新闻",
        str(query or ""),
        flags=re.IGNORECASE,
    ):
        return "news"
    return "general"


def _json_object(response: requests.Response, stage: str) -> dict[str, Any]:
    if response.status_code >= 400:
        detail = response.text[:500].replace("\n", " ")
        raise SearchReaderError(stage, f"HTTP {response.status_code}: {detail}")
    try:
        value = json.loads(response.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SearchReaderError(stage, "provider returned invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SearchReaderError(stage, "provider returned a non-object JSON response")
    return value


class SearchProvider(ABC):
    name: str
    results_are_material = False

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SearchHit]:
        raise NotImplementedError

    def extract(self, hits: list[SearchHit], query: str) -> dict[str, str]:
        del hits, query
        return {}


class TavilyProvider(SearchProvider):
    name = "tavily"
    # Advanced Search already returns query-ranked, source-bounded material in
    # results[].content. Do not fetch or re-chunk those pages locally.
    results_are_material = True
    endpoint = "https://api.tavily.com"

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _post(
        self, path: str, payload: dict[str, Any], *, timeout: tuple[int, int]
    ) -> requests.Response:
        configured_keys = self.settings.tavily_api_keys or (
            (self.settings.tavily_api_key,) if self.settings.tavily_api_key else ()
        )
        if not configured_keys:
            raise SearchReaderError(
                "search", "TAVILY_API_KEY or TAVILY_API_KEYS is not configured"
            )
        keys = _ordered_tavily_keys(tuple(configured_keys))
        if not keys:
            raise SearchReaderError(
                "search", "all configured Tavily credentials are unavailable"
            )
        last_response: requests.Response | None = None
        last_error: requests.RequestException | None = None
        for api_key in keys:
            try:
                response = self.session.post(
                    f"{self.endpoint}/{path.lstrip('/')}",
                    headers=self._headers(api_key),
                    json=payload,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                continue
            last_response = response
            if response.status_code < 400:
                _mark_tavily_success(api_key)
                return response
            if response.status_code in {401, 402, 403, 432}:
                _mark_tavily_failure(api_key)
            if response.status_code not in {401, 402, 403, 429, 432}:
                return response
        if last_response is not None:
            return last_response
        raise SearchReaderError(
            "search",
            f"Tavily request failed: {last_error or 'no credential succeeded'}",
        )

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        request: dict[str, Any] = {
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "topic": _tavily_topic(query),
        }
        domains = _explicit_domains(query)
        if domains:
            request["include_domains"] = domains
        response = self._post(
            "search",
            request,
            timeout=(10, self.settings.request_timeout_seconds),
        )
        payload = _json_object(response, "search")
        hits: list[SearchHit] = []
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = _clean(item.get("url"))
            if not _valid_public_url_shape(url):
                continue
            material = str(item.get("content") or item.get("snippet") or "").strip()
            hits.append(
                SearchHit(
                    title=_clean(item.get("title") or url, 300),
                    url=url,
                    snippet=_clean(material, 1200),
                    content=material,
                    provider=self.name,
                    score=float(item.get("score") or 0.0),
                    published_date=_clean(
                        item.get("published_date") or item.get("publishedDate"), 80
                    ),
                )
            )
        return hits

class WigoloProvider(SearchProvider):
    name = "wigolo"

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.settings.wigolo_api_token:
            headers["Authorization"] = f"Bearer {self.settings.wigolo_api_token}"
        return headers

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.wigolo_base_url:
            raise SearchReaderError(
                "search", "SEARCH_READER_WIGOLO_BASE_URL is not configured"
            )
        try:
            response = self.session.post(
                f"{self.settings.wigolo_base_url.rstrip('/')}/v1/{path.lstrip('/')}",
                headers=self.headers,
                json=payload,
                timeout=(5, self.settings.request_timeout_seconds),
            )
        except requests.RequestException as exc:
            raise SearchReaderError("search", f"wigolo request failed: {exc}") from exc
        return _json_object(response, "search")

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        payload = self._post(
            "search",
            {"query": query, "max_results": max_results, "search_depth": "advanced"},
        )
        hits: list[SearchHit] = []
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = _clean(item.get("url") or item.get("link"))
            if not _valid_public_url_shape(url):
                continue
            hits.append(
                SearchHit(
                    title=_clean(item.get("title") or item.get("name") or url, 300),
                    url=url,
                    snippet=_clean(
                        item.get("excerpt")
                        or item.get("snippet")
                        or item.get("description")
                        or item.get("summary"),
                        1200,
                    ),
                    content=str(
                        item.get("content")
                        or item.get("markdown")
                        or item.get("text")
                        or item.get("body")
                        or ""
                    ).strip(),
                    provider=self.name,
                )
            )
        return hits

    def extract(self, hits: list[SearchHit], query: str) -> dict[str, str]:
        del query
        extracted: dict[str, str] = {}
        for hit in hits:
            if hit.content:
                extracted[hit.url] = hit.content
                continue
            try:
                payload = self._post("fetch", {"url": hit.url})
            except SearchReaderError:
                continue
            content = next(
                (
                    str(payload.get(key) or "").strip()
                    for key in (
                        "content",
                        "markdown",
                        "text",
                        "body",
                        "excerpt",
                        "snippet",
                    )
                    if str(payload.get(key) or "").strip()
                ),
                "",
            )
            if content:
                extracted[hit.url] = content
        return extracted


class SearxngProvider(SearchProvider):
    name = "searxng"

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        if not self.settings.searxng_base_url:
            raise SearchReaderError(
                "search", "SEARCH_READER_SEARXNG_BASE_URL is not configured"
            )
        try:
            response = self.session.get(
                f"{self.settings.searxng_base_url.rstrip('/')}/search",
                params={"q": query, "format": "json", "language": "auto"},
                headers={
                    "Accept": "application/json",
                    "User-Agent": "RWKV-SearchReader/0.1",
                },
                timeout=(5, self.settings.request_timeout_seconds),
            )
        except requests.RequestException as exc:
            raise SearchReaderError("search", f"SearXNG request failed: {exc}") from exc
        payload = _json_object(response, "search")
        hits: list[SearchHit] = []
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = _clean(item.get("url"))
            if not _valid_public_url_shape(url):
                continue
            hits.append(
                SearchHit(
                    title=_clean(item.get("title") or url, 300),
                    url=url,
                    snippet=_clean(item.get("content"), 1200),
                    provider=self.name,
                    score=float(item.get("score") or 0.0),
                    published_date=_clean(item.get("publishedDate"), 80),
                )
            )
            if len(hits) >= max_results:
                break
        return hits


def _valid_public_url_shape(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme.casefold() in {"http", "https"} and bool(parsed.hostname)


def create_search_provider(settings: Settings) -> SearchProvider:
    providers: dict[str, type[SearchProvider]] = {
        "tavily": TavilyProvider,
        "wigolo": WigoloProvider,
        "searxng": SearxngProvider,
    }
    return providers[settings.search_provider](settings)
