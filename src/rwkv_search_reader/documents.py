from __future__ import annotations

import math
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BROAD_DOCUMENT_RE = re.compile(
    r"总结|概括|综述|全文|整篇|整体|主要内容|核心内容|要点|大意|梳理"
)


@dataclass(frozen=True, slots=True)
class MarkdownPassage:
    document_id: str
    filename: str
    ordinal: int
    heading: str
    content: str

    @property
    def title(self) -> str:
        return f"{self.filename} · {self.heading}" if self.heading else self.filename

    @property
    def url(self) -> str:
        return f"document://{self.document_id}#chunk-{self.ordinal + 1}"


def normalize_markdown(filename: str, content: str) -> tuple[str, str]:
    clean_name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not clean_name or len(clean_name) > 180:
        raise ValueError("invalid document filename")
    if not clean_name.casefold().endswith((".md", ".markdown")):
        raise ValueError("only Markdown documents are supported")
    clean_content = str(content or "").replace("\x00", "").replace("\r\n", "\n")
    clean_content = clean_content.replace("\r", "\n").strip()
    if not clean_content:
        raise ValueError("document is empty")
    return clean_name, clean_content


def _split_text(value: str, target_chars: int) -> list[str]:
    text = value.strip()
    if not text:
        return []
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > target_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            overlap = min(180, target_chars // 8)
            while start < len(paragraph):
                end = min(len(paragraph), start + target_chars)
                if end < len(paragraph):
                    boundary = max(
                        paragraph.rfind(mark, start + target_chars // 2, end)
                        for mark in ("。", "！", "？", ".", "!", "?", "\n")
                    )
                    if boundary > start:
                        end = boundary + 1
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start = max(start + 1, end - overlap)
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > target_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_markdown(content: str, target_chars: int = 2400) -> list[tuple[str, str]]:
    """Split Markdown by heading path, then by paragraphs within each section."""

    heading_path: list[str] = []
    section_heading = ""
    section_lines: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        body = "\n".join(section_lines).strip()
        if body:
            sections.append((section_heading, body))

    for line in str(content or "").splitlines():
        match = _HEADING_RE.match(line.strip())
        if not match:
            section_lines.append(line)
            continue
        flush()
        level = len(match.group(1))
        title = match.group(2).strip()
        heading_path[level - 1 :] = [title]
        section_heading = " / ".join(heading_path)
        section_lines = []
    flush()

    if not sections and str(content or "").strip():
        sections = [("", str(content).strip())]

    chunks: list[tuple[str, str]] = []
    for heading, body in sections:
        for part in _split_text(body, target_chars):
            chunks.append((heading, part))
    return chunks


def _terms(value: str) -> set[str]:
    text = str(value or "").casefold()
    terms = set(re.findall(r"[a-z][a-z0-9_.+-]{1,}", text))
    terms.update(re.findall(r"\d+(?:[.,]\d+)*", text))
    for segment in re.findall(r"[\u3400-\u9fff]{2,}", text):
        terms.add(segment)
        terms.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return terms


def _even_sample(rows: list[MarkdownPassage], count: int) -> list[MarkdownPassage]:
    if len(rows) <= count:
        return rows
    if count <= 1:
        return [rows[0]]
    indexes = {
        min(len(rows) - 1, round(index * (len(rows) - 1) / (count - 1)))
        for index in range(count)
    }
    return [rows[index] for index in sorted(indexes)]


def rank_markdown_passages(
    query: str, passages: list[MarkdownPassage], limit: int
) -> list[MarkdownPassage]:
    if not passages or limit <= 0:
        return []
    if _BROAD_DOCUMENT_RE.search(str(query or "")):
        by_document: dict[str, list[MarkdownPassage]] = {}
        for passage in passages:
            by_document.setdefault(passage.document_id, []).append(passage)
        allocation = max(1, math.ceil(limit / max(1, len(by_document))))
        samples = [
            _even_sample(rows, allocation) for rows in by_document.values()
        ]
        selected: list[MarkdownPassage] = []
        for position in range(allocation):
            for rows in samples:
                if position < len(rows):
                    selected.append(rows[position])
                if len(selected) >= limit:
                    return selected
        return selected

    query_terms = _terms(query)

    def score(passage: MarkdownPassage) -> tuple[float, float, int]:
        heading_terms = _terms(passage.heading)
        content_terms = _terms(passage.content)
        heading_overlap = len(query_terms & heading_terms) / max(1, len(query_terms))
        content_overlap = len(query_terms & content_terms) / max(1, len(query_terms))
        phrase_bonus = 1.0 if str(query).casefold() in passage.content.casefold() else 0.0
        return (heading_overlap * 1.8 + content_overlap + phrase_bonus, -passage.ordinal, -len(passage.content))

    ranked = sorted(passages, key=score, reverse=True)
    return ranked[:limit]
