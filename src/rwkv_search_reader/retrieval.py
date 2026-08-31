from __future__ import annotations

import ipaddress
import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse

import requests

from .config import Settings
from .models import DocumentChunk, SearchHit, SearchReaderError
from .search import SearchProvider

_SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "template",
    "nav",
    "footer",
}


class _ReadableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._title_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        name = tag.casefold()
        if name in _SKIP_TAGS:
            self._skip_depth += 1
        if name == "title":
            self._title_depth += 1
        if not self._skip_depth and name in {
            "p",
            "div",
            "article",
            "main",
            "section",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
        }:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        name = tag.casefold()
        if name in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if name == "title" and self._title_depth:
            self._title_depth -= 1
        if not self._skip_depth and name in {
            "p",
            "div",
            "article",
            "main",
            "section",
            "li",
            "tr",
        }:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        value = " ".join(data.split())
        if not value:
            return
        if self._title_depth:
            self.title_parts.append(value)
        self.parts.append(value + " ")

    def text(self) -> str:
        paragraphs = [
            " ".join(item.split()) for item in "".join(self.parts).splitlines()
        ]
        return "\n".join(item for item in paragraphs if len(item) >= 20)


def normalize_url(value: str) -> str:
    url, _ = urldefrag(str(value or "").strip())
    parsed = urlparse(url)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return ""
    return parsed._replace(
        scheme=parsed.scheme.casefold(), netloc=parsed.netloc.casefold()
    ).geturl()


def _assert_public_destination(url: str) -> None:
    """Reject direct access to local/private networks when fetching search hits."""

    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host.casefold() in {"localhost", "localhost.localdomain"}:
        raise ValueError("local destinations are not allowed")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                host, parsed.port or 443, type=socket.SOCK_STREAM
            )
        }
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve host: {host}") from exc
    for address in addresses:
        if not ipaddress.ip_address(address).is_global:
            raise ValueError("private or non-global destinations are not allowed")


def fetch_page(hit: SearchHit, settings: Settings) -> SearchHit:
    """Fallback page fetcher for providers that only return short snippets."""

    if hit.content.strip():
        return hit
    try:
        target = hit.url
        response: requests.Response | None = None
        for _ in range(5):
            _assert_public_destination(target)
            response = requests.get(
                target,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) RWKV-SearchReader/0.1",
                    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
                timeout=(5, settings.request_timeout_seconds),
                allow_redirects=False,
                stream=True,
            )
            if response.is_redirect or response.is_permanent_redirect:
                location = str(response.headers.get("Location") or "").strip()
                response.close()
                target = normalize_url(urljoin(target, location))
                if not target:
                    raise ValueError("redirect target is not a valid HTTP URL")
                continue
            break
        else:
            raise ValueError("too many redirects")
        if response is None:
            raise ValueError("page request did not return a response")
        with response:
            response.raise_for_status()
            content_type = str(response.headers.get("Content-Type") or "").casefold()
            if "pdf" in content_type or (
                content_type
                and not any(
                    value in content_type for value in ("text/", "html", "xml", "json")
                )
            ):
                raise ValueError(
                    f"unsupported content type: {content_type or 'unknown'}"
                )
            raw = bytearray()
            for block in response.iter_content(chunk_size=65536):
                raw.extend(block)
                if len(raw) >= 1_500_000:
                    break
            response.encoding = response.encoding or response.apparent_encoding or "utf-8"
            text = bytes(raw).decode(response.encoding, errors="replace")
            if "html" in content_type or "<html" in text[:1000].casefold():
                parser = _ReadableTextParser()
                parser.feed(text)
                hit.content = parser.text()[:120_000]
                if not hit.title and parser.title_parts:
                    hit.title = " ".join(parser.title_parts)[:300]
            else:
                hit.content = text[:120_000]
    except (requests.RequestException, ValueError, OSError) as exc:
        hit.error = f"{type(exc).__name__}: {exc}"[:400]
    return hit


def _terms(value: str) -> set[str]:
    lowered = str(value or "").casefold()
    terms = set(re.findall(r"[a-z][a-z0-9_+.-]{1,}", lowered))
    for segment in re.findall(r"[\u3400-\u9fff]{2,}", lowered):
        if len(segment) <= 4:
            terms.add(segment)
        terms.update(segment[index : index + 2] for index in range(len(segment) - 1))
    ignored = {
        "the",
        "and",
        "for",
        "with",
        "what",
        "which",
        "怎么",
        "什么",
        "是否",
        "如何",
    }
    return {term for term in terms if term not in ignored}


def _source_adjustment(question: str, url: str) -> float:
    host = (urlparse(url).hostname or "").casefold().removeprefix("www.")
    explicit_domains = re.findall(
        r"(?<![@\w])((?:[a-z0-9-]+\.)+[a-z]{2,})(?![\w])",
        question.casefold(),
    )
    if any(host == domain or host.endswith(f".{domain}") for domain in explicit_domains):
        return 18.0
    official_requested = any(
        marker in question.casefold()
        for marker in ("官方", "官网", "official", "原始论文", "政府")
    )
    low_authority = (
        "csdn.net",
        "juejin.cn",
        "zhihu.com",
        "medium.com",
        "reddit.com",
        "quora.com",
    )
    if official_requested and any(
        host == domain or host.endswith(f".{domain}") for domain in low_authority
    ):
        return -7.0
    if host.endswith((".gov", ".gov.cn", ".edu")):
        return 3.0
    return 0.0


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    clean = re.sub(r"\n{3,}", "\n\n", str(text or "")).strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        if end < len(clean):
            boundary_floor = start + int(size * 0.6)
            boundaries = [
                clean.rfind(marker, boundary_floor, end)
                for marker in ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ")
            ]
            boundary = max(boundaries)
            if boundary >= boundary_floor:
                end = boundary + 1
        value = clean[start:end].strip()
        if value:
            chunks.append(value)
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


def rank_chunks(
    question: str, hits: list[SearchHit], settings: Settings
) -> list[DocumentChunk]:
    """Fallback ranking for providers whose result snippets are not reader-ready."""

    query_terms = _terms(question)
    chunks: list[DocumentChunk] = []
    for source_index, hit in enumerate(hits, start=1):
        body = hit.content.strip() or hit.snippet.strip()
        for chunk_index, text in enumerate(
            chunk_text(body, settings.chunk_chars, settings.chunk_overlap_chars)
        ):
            terms = _terms(text)
            overlap = len(query_terms & terms)
            coverage = overlap / max(1, len(query_terms))
            title_overlap = len(query_terms & _terms(hit.title))
            score = (
                coverage * 10.0
                + title_overlap * 1.5
                + float(hit.score or 0.0)
                + _source_adjustment(question, hit.url)
            )
            chunks.append(
                DocumentChunk(
                    source_id=hit.source_id or f"S{source_index}",
                    title=hit.title,
                    url=hit.url,
                    text=text,
                    score=round(score, 4),
                    chunk_index=chunk_index,
                    published_date=hit.published_date,
                    retrieval_point=hit.retrieval_point,
                    retrieval_query=hit.retrieval_query,
                )
            )
    chunks.sort(
        key=lambda item: (-item.score, int(item.source_id[1:]), item.chunk_index)
    )
    return chunks


def provider_materials(hits: list[SearchHit]) -> list[DocumentChunk]:
    """Preserve a provider's query-ranked result order and material boundaries."""

    return [
        DocumentChunk(
            source_id=hit.source_id or f"S{source_index}",
            title=hit.title,
            url=hit.url,
            text=hit.content.strip() or hit.snippet.strip(),
            score=float(hit.score or 0.0),
            chunk_index=0,
            published_date=hit.published_date,
            retrieval_point=hit.retrieval_point,
            retrieval_query=hit.retrieval_query,
        )
        for source_index, hit in enumerate(hits, start=1)
        if hit.content.strip() or hit.snippet.strip()
    ]


def _search_queries(question: str) -> list[str]:
    """Split only explicit multi-question utterances; never ask a model to route."""

    normalized = " ".join(str(question or "").split()).strip()
    if not normalized:
        return []
    explicit_questions = [
        match.group(0).strip()
        for match in re.finditer(r"[^?？!！]+[?？]", normalized)
        if match.group(0).strip(" ?？!！")
    ]
    if len(explicit_questions) < 2:
        return [normalized]
    return [item.strip() for item in explicit_questions[:3]]


def _balanced_hits(
    groups: list[list[SearchHit]],
    limit: int,
    excluded_urls: set[str] | None = None,
) -> list[SearchHit]:
    """Preserve each query's Tavily rank while giving every subquestion coverage."""

    selected: list[SearchHit] = []
    seen = {_url_identity(url) for url in (excluded_urls or set())}
    seen.discard("")
    depth = 0
    while len(selected) < limit and any(depth < len(group) for group in groups):
        for group in groups:
            if depth >= len(group):
                continue
            hit = group[depth]
            hit.url = normalize_url(hit.url)
            identity = _url_identity(hit.url)
            if hit.url and identity not in seen:
                seen.add(identity)
                selected.append(hit)
                if len(selected) >= limit:
                    break
        depth += 1
    return selected


def _url_identity(url: str) -> str:
    """Treat HTTP/HTTPS variants of the same page as one evidence source."""

    normalized = normalize_url(url)
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    port = parsed.port
    if port and not (
        (parsed.scheme == "http" and port == 80)
        or (parsed.scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/") or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{path}{query}".casefold()


class Retriever:
    def __init__(self, settings: Settings, provider: SearchProvider):
        self.settings = settings
        self.provider = provider

    def run(
        self, question: str
    ) -> tuple[list[str], list[SearchHit], list[DocumentChunk], list[str]]:
        queries = _search_queries(question)
        if not queries:
            raise SearchReaderError("input", "question is empty")
        return self.run_queries(queries, question=question)

    def run_queries(
        self,
        queries: list[str],
        *,
        question: str,
        excluded_urls: set[str] | None = None,
        start_index: int = 1,
        results_per_query: int | None = None,
        max_total_results: int | None = None,
        retrieval_point_by_query: dict[str, str] | None = None,
    ) -> tuple[list[str], list[SearchHit], list[DocumentChunk], list[str]]:
        queries = list(
            dict.fromkeys(
                " ".join(str(query or "").split()).strip()
                for query in queries
                if " ".join(str(query or "").split()).strip()
            )
        )
        if not queries:
            raise SearchReaderError("input", "search query is empty")
        per_query = max(1, min(10, results_per_query or self.settings.search_max_results))
        total_limit = max_total_results or per_query * len(queries)
        total_limit = max(1, min(60, total_limit))
        groups: list[list[SearchHit]] = []
        search_warnings: list[str] = []
        for query in queries:
            try:
                hits = self.provider.search(query, per_query)
                retrieval_point = str(
                    (retrieval_point_by_query or {}).get(query) or query
                ).strip()
                for hit in hits:
                    hit.retrieval_point = retrieval_point
                    hit.retrieval_query = query
                groups.append(hits)
            except SearchReaderError as exc:
                search_warnings.append(f"{query}: {exc}"[:500])
        if not groups:
            raise SearchReaderError("search", "all deterministic search queries failed")
        deduplicated = _balanced_hits(
            groups,
            total_limit,
            excluded_urls=excluded_urls,
        )
        if not deduplicated:
            raise SearchReaderError("search", "search provider returned no usable results")
        for index, hit in enumerate(deduplicated, start=start_index):
            hit.source_id = f"S{index}"

        selected = (
            deduplicated[: self.settings.fetch_pages * len(queries)]
            if self.provider.results_are_material
            else deduplicated[: self.settings.fetch_pages]
        )
        if self.provider.results_are_material:
            chunks = provider_materials(selected)
            if not chunks:
                raise SearchReaderError("search", "search provider returned no readable material")
            return queries, deduplicated, chunks, search_warnings

        extracted = self.provider.extract(selected, question)
        for hit in selected:
            if extracted.get(hit.url):
                hit.content = extracted[hit.url][:120_000]
        missing = [hit for hit in selected if not hit.content.strip()]
        if missing:
            with ThreadPoolExecutor(max_workers=min(5, len(missing))) as pool:
                futures = {
                    pool.submit(fetch_page, hit, self.settings): hit for hit in missing
                }
                for future in as_completed(futures):
                    future.result()

        warnings = [
            *search_warnings,
            *(f"{hit.url}: {hit.error}" for hit in selected if hit.error),
        ]
        chunks = rank_chunks(question, selected, self.settings)
        if not chunks:
            raise SearchReaderError(
                "fetch", "no readable page text or search snippets were available"
            )
        return queries, deduplicated, chunks, warnings
