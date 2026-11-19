from __future__ import annotations

import json
import re
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

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


def _bound_deepseek_text(value: str, limit: int = 128) -> str:
    """Clamp text to DeepSeek's 1..128 Unicode character schema requirement."""

    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(
        cut.rfind(sep)
        for sep in (" ", "，", ",", "；", ";", "。", "？", "?")
    )
    if boundary >= int(limit * 0.65):
        cut = cut[:boundary]
    return cut.strip(" ，,；;。") or text[:limit].strip()


class SearchProvider(ABC):
    name: str
    results_are_material = False
    concurrent_queries = False

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SearchHit]:
        raise NotImplementedError

    def search_with_context(
        self, query: str, max_results: int, *, ori_question: str = ""
    ) -> list[SearchHit]:
        del ori_question
        return self.search(query, max_results)

    def fetch_url(self, url: str) -> tuple[str, str] | None:
        del url
        return None

    def extract(self, hits: list[SearchHit], query: str) -> dict[str, str]:
        del hits, query
        return {}


class DeepSeekSearchProvider(SearchProvider):
    """Official DeepSeek Search (/v1/tools/search) and Fetch (/v1/tools/fetch)."""

    name = "deepseek"
    concurrent_queries = True

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.results_are_material = bool(settings.deepseek_need_content)
        self._local = threading.local()

    def _thread_session(self) -> requests.Session:
        # Allow unit tests to monkeypatch self.session.post directly.
        if "post" in self.session.__dict__:
            return self.session
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
            self._local.session = sess
        return sess

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @property
    def _endpoint(self) -> str:
        return (self.settings.deepseek_base_url or "https://api.deepseek.com").rstrip("/")

    def _post(
        self, path: str, payload: dict[str, Any], *, stage: str = "search"
    ) -> dict[str, Any]:
        api_key = self.settings.deepseek_api_key.strip()
        if not api_key:
            raise SearchReaderError(stage, "DEEPSEEK_API_KEY is not configured")
        try:
            response = self._thread_session().post(
                f"{self._endpoint}/{path.lstrip('/')}",
                headers=self._headers(api_key),
                json=payload,
                timeout=(8, self.settings.request_timeout_seconds),
            )
        except requests.RequestException as exc:
            raise SearchReaderError(
                stage, f"DeepSeek {stage} request failed: {type(exc).__name__}"
            ) from exc
        return _json_object(response, stage)

    def search(
        self, query: str, max_results: int, *, ori_question: str = ""
    ) -> list[SearchHit]:
        return self.search_with_context(query, max_results, ori_question=ori_question)

    def search_with_context(
        self, query: str, max_results: int, *, ori_question: str = ""
    ) -> list[SearchHit]:
        clean_query = _bound_deepseek_text(query, 128)
        if not clean_query:
            raise SearchReaderError("input", "search query is empty")
        domains = _explicit_domains(query)
        effective_query = clean_query
        added_site_filter = False
        if (
            len(domains) == 1
            and "site:" not in clean_query.casefold()
            and len(f"{clean_query} site:{domains[0]}") <= 128
        ):
            effective_query = f"{clean_query} site:{domains[0]}"
            added_site_filter = True

        count = max(1, min(50, int(max_results)))
        request: dict[str, Any] = {
            "query": effective_query,
            "count": count,
            "need_content": bool(self.settings.deepseek_need_content),
        }
        bounded_question = _bound_deepseek_text(ori_question, 128)
        if bounded_question:
            request["ori_question"] = bounded_question

        payload = self._post("v1/tools/search", request, stage="search")
        items = payload.get("results") or []
        if not items and added_site_filter:
            request["query"] = clean_query
            payload = self._post("v1/tools/search", request, stage="search")
            items = payload.get("results") or []

        hits: list[SearchHit] = []
        limit = self.settings.max_evidence_chars
        for item in items:
            if not isinstance(item, dict):
                continue
            url = _clean(item.get("url"))
            if not _valid_public_url_shape(url):
                continue
            snippet = str(item.get("snippet") or "").strip()
            summary = str(item.get("summary") or "").strip()
            raw_content = str(item.get("content") or "").strip()
            if raw_content:
                if (
                    summary
                    and len(raw_content) > limit
                    and summary[:120] not in raw_content[:limit]
                ):
                    prefix = f"【检索要点】{summary}\n\n"
                    material = prefix + raw_content[: max(0, limit - len(prefix))]
                else:
                    material = raw_content[:limit]
            else:
                material = (summary or snippet)[:limit]
            authority_raw = item.get("authority")
            try:
                authority = int(authority_raw) if authority_raw is not None else 0
            except (TypeError, ValueError):
                authority = 0
            hits.append(
                SearchHit(
                    title=_clean(item.get("title") or url, 300),
                    url=url,
                    snippet=_clean(snippet or summary or material, 1200),
                    content=material,
                    content_markdown=raw_content or summary,
                    content_status="fetched" if raw_content else "snippet_only",
                    provider=self.name,
                    score=float(item.get("rerank_score") or item.get("score") or 0.0),
                    published_date=_clean(
                        item.get("date")
                        or item.get("published_date")
                        or item.get("publishedDate"),
                        80,
                    ),
                    authority=authority,
                    site_name=_clean(item.get("site_name"), 120),
                )
            )
            if len(hits) >= max_results:
                break
        return hits

    def fetch_url(self, url: str) -> tuple[str, str] | None:
        if not self.settings.deepseek_api_key.strip():
            return None
        try:
            payload = self._post("v1/tools/fetch", {"url": url}, stage="fetch")
        except SearchReaderError:
            return None
        content = str(payload.get("content") or "").strip()
        if not content:
            return None
        title = _clean(payload.get("title") or url, 300)
        return title, content

    def extract(self, hits: list[SearchHit], query: str) -> dict[str, str]:
        del query
        extracted: dict[str, str] = {}
        pending = [hit for hit in hits if not hit.content.strip()]
        if not pending:
            return {hit.url: hit.content for hit in hits if hit.content.strip()}
        with ThreadPoolExecutor(max_workers=min(6, len(pending))) as pool:
            futures = {
                pool.submit(self.fetch_url, hit.url): hit.url for hit in pending
            }
            for future in as_completed(futures):
                url = futures[future]
                fetched = future.result()
                if fetched and fetched[1]:
                    extracted[url] = fetched[1]
        for hit in hits:
            if hit.content.strip():
                extracted.setdefault(hit.url, hit.content)
        return extracted


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
        "deepseek": DeepSeekSearchProvider,
        "tavily": TavilyProvider,
        "wigolo": WigoloProvider,
        "searxng": SearxngProvider,
    }
    return providers[settings.search_provider](settings)
