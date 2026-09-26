from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

import requests

from .config import Settings
from .models import SearchHit, SearchReaderError

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
_AMBIGUOUS_CODE_SUFFIXES = {
    "js",
    "jsx",
    "py",
    "rb",
    "ts",
    "tsx",
}


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
            and value.rsplit(".", 1)[-1] not in _AMBIGUOUS_CODE_SUFFIXES
            and (
                len(value.rsplit(".", 1)[-1]) == 2
                or value.rsplit(".", 1)[-1] in _COMMON_GENERIC_TLDS
            )
        )
    )[:5]


def _tavily_topic(query: str) -> str:
    """Use Tavily news mode when publication timestamps are part of the evidence."""

    if re.search(
        r"新闻|今日|今天|刚刚|本周|这周|近一周|最近一周|本月|这个月",
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
    # Advanced Search returns query-ranked snippets plus cleaned page Markdown.
    # Do not fetch those pages again locally.
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
        api_key = self.settings.tavily_api_key or next(
            iter(self.settings.tavily_api_keys), ""
        )
        if not api_key:
            raise SearchReaderError(
                "search", "TAVILY_API_KEY is not configured"
            )
        try:
            return self.session.post(
                f"{self.endpoint}/{path.lstrip('/')}",
                headers=self._headers(api_key),
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise SearchReaderError(
                "search", f"Tavily request failed: {type(exc).__name__}"
            ) from exc

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        request: dict[str, Any] = {
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": "markdown",
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
        if (
            response.status_code == 400
            and "include_domains" in response.text.casefold()
            and request.pop("include_domains", None) is not None
        ):
            # Product names such as Node.js can resemble a bare domain. If Tavily
            # rejects an inferred filter, preserve the query and search broadly.
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
            snippet = str(item.get("content") or item.get("snippet") or "").strip()
            raw_content = str(item.get("raw_content") or "").strip()
            material = (raw_content or snippet)[: self.settings.max_evidence_chars]
            hits.append(
                SearchHit(
                    title=_clean(item.get("title") or url, 300),
                    url=url,
                    snippet=_clean(snippet or material, 1200),
                    content=material,
                    content_markdown=raw_content,
                    content_status="fetched" if raw_content else "snippet_only",
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
