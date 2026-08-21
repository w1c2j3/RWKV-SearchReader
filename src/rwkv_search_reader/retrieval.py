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
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("private or non-global destinations are not allowed")


def fetch_page(hit: SearchHit, settings: Settings) -> SearchHit:
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
            response.encoding = (
                response.encoding or response.apparent_encoding or "utf-8"
            )
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
            score = coverage * 10.0 + title_overlap * 1.5 + float(hit.score or 0.0)
            chunks.append(
                DocumentChunk(
                    source_id=f"S{source_index}",
                    title=hit.title,
                    url=hit.url,
                    text=text,
                    score=round(score, 4),
                    chunk_index=chunk_index,
                )
            )
    chunks.sort(
        key=lambda item: (-item.score, int(item.source_id[1:]), item.chunk_index)
    )
    return chunks


class Retriever:
    def __init__(self, settings: Settings, provider: SearchProvider):
        self.settings = settings
        self.provider = provider

    def run(
        self, question: str
    ) -> tuple[list[str], list[SearchHit], list[DocumentChunk], list[str]]:
        # V1 deliberately uses the user's complete utterance as the sole query.
        query = " ".join(question.split()).strip()
        if not query:
            raise SearchReaderError("input", "question is empty")
        queries = [query]
        hits = self.provider.search(query, self.settings.search_max_results)
        deduplicated: list[SearchHit] = []
        seen: set[str] = set()
        for hit in hits:
            hit.url = normalize_url(hit.url)
            identity = hit.url.rstrip("/").casefold()
            if not hit.url or identity in seen:
                continue
            seen.add(identity)
            deduplicated.append(hit)
        if not deduplicated:
            raise SearchReaderError(
                "search", "search provider returned no usable results"
            )

        selected = deduplicated[: self.settings.fetch_pages]
        extracted = self.provider.extract(selected, query)
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

        warnings = [f"{hit.url}: {hit.error}" for hit in selected if hit.error]
        chunks = rank_chunks(query, deduplicated, self.settings)
        if not chunks:
            raise SearchReaderError(
                "fetch", "no readable page text or search snippets were available"
            )
        return queries, deduplicated, chunks, warnings
