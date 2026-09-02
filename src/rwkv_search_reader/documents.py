from __future__ import annotations

import base64
import binascii
import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BROAD_DOCUMENT_RE = re.compile(
    r"总结|概括|综述|全文|整篇|整体|主要内容|核心内容|要点|大意|梳理|"
    r"对比|比较|异同|区别|两篇|多篇|各篇|两份|多份|各份|全部文档|所有文档|"
    r"summari[sz]e|overview|compare|comparison|whole\s+(?:paper|document)|"
    r"entire\s+(?:paper|document)|both\s+papers|all\s+documents",
    flags=re.IGNORECASE,
)

ProgressCallback = Callable[[str, dict[str, Any]], None]


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


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    filename: str
    markdown: str
    source_bytes: bytes
    source_mime: str


_PDF_CONVERSION_LOCK = threading.Lock()


def _clean_filename(filename: str) -> str:
    clean_name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not clean_name or len(clean_name) > 180:
        raise ValueError("invalid document filename")
    return clean_name


def normalize_markdown(filename: str, content: str) -> tuple[str, str]:
    clean_name = _clean_filename(filename)
    if not clean_name.casefold().endswith((".md", ".markdown")):
        raise ValueError("only Markdown documents are supported")
    clean_content = str(content or "").replace("\x00", "").replace("\r\n", "\n")
    clean_content = clean_content.replace("\r", "\n").strip()
    if not clean_content:
        raise ValueError("document is empty")
    return clean_name, clean_content


@lru_cache(maxsize=2)
def _docling_converter(do_ocr: bool = True):
    try:
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
        )
    except (
        ImportError
    ) as exc:  # pragma: no cover - depends on optional deployment extra
        raise RuntimeError(
            "PDF support is unavailable; install the project with the pdf extra"
        ) from exc
    pipeline_options = PdfPipelineOptions(
        document_timeout=120,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU,
            num_threads=4,
        ),
        do_ocr=do_ocr,
        do_table_structure=True,
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )


def _convert_pdf_bytes(
    filename: str,
    content: bytes,
    *,
    max_pages: int,
    max_bytes: int,
    progress: ProgressCallback | None = None,
) -> str:
    try:
        from docling.datamodel.base_models import DocumentStream
    except (
        ImportError
    ) as exc:  # pragma: no cover - depends on optional deployment extra
        raise RuntimeError(
            "PDF support is unavailable; install the project with the pdf extra"
        ) from exc

    needs_ocr = _pdf_needs_ocr(content)
    if progress:
        progress(
            "pdf_analyzed",
            {
                "ocr": needs_ocr,
                "message": (
                    "正在识别扫描页文字"
                    if needs_ocr
                    else "已检测到文本层，跳过 OCR"
                ),
            },
        )
    stream = DocumentStream(name=filename, stream=BytesIO(content))
    with _PDF_CONVERSION_LOCK:
        if progress:
            progress(
                "pdf_converting",
                {"ocr": needs_ocr, "message": "正在还原标题、段落和表格"},
            )
        result = _docling_converter(needs_ocr).convert(
            stream,
            max_num_pages=max_pages,
            max_file_size=max_bytes,
        )
    return str(result.document.export_to_markdown() or "")


def _pdf_needs_ocr(content: bytes) -> bool:
    """Use OCR only when sampled pages do not already expose a useful text layer."""

    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(BytesIO(content))
        page_count = len(document)
        if not page_count:
            return True
        sample_count = min(8, page_count)
        indexes = sorted(
            {
                min(page_count - 1, round(index * (page_count - 1) / max(1, sample_count - 1)))
                for index in range(sample_count)
            }
        )
        visible_chars = 0
        readable_pages = 0
        for index in indexes:
            page = document[index]
            text_page = page.get_textpage()
            text = re.sub(r"\s+", "", text_page.get_text_range() or "")
            visible_chars += len(text)
            if len(text) >= 80:
                readable_pages += 1
        return visible_chars < 300 or readable_pages < max(1, len(indexes) // 2)
    except Exception:  # noqa: BLE001  # pragma: no cover - safe OCR fallback
        return True


def prepare_document(
    filename: str,
    content: str = "",
    content_base64: str = "",
    *,
    max_markdown_bytes: int,
    max_pdf_bytes: int,
    max_pdf_pages: int,
    progress: ProgressCallback | None = None,
) -> PreparedDocument:
    """Validate an uploaded document and normalize it to Markdown for retrieval."""

    clean_name = _clean_filename(filename)
    suffix = clean_name.casefold().rsplit(".", 1)[-1] if "." in clean_name else ""
    if suffix in {"md", "markdown"}:
        clean_name, clean_content = normalize_markdown(clean_name, content)
        encoded = clean_content.encode("utf-8")
        if len(encoded) > max_markdown_bytes:
            raise ValueError(f"document exceeds {max_markdown_bytes} bytes")
        return PreparedDocument(clean_name, clean_content, encoded, "text/markdown")

    if suffix != "pdf":
        raise ValueError("only Markdown and PDF documents are supported")
    if not content_base64:
        raise ValueError("PDF content is missing")
    try:
        raw = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("PDF content is not valid base64") from exc
    if not raw:
        raise ValueError("document is empty")
    if len(raw) > max_pdf_bytes:
        raise ValueError(f"PDF exceeds {max_pdf_bytes} bytes")
    if not raw.lstrip().startswith(b"%PDF-"):
        raise ValueError("uploaded file is not a PDF")
    try:
        if progress:
            progress("pdf_received", {"message": "PDF 已接收，正在检查文本层"})
        markdown = _convert_pdf_bytes(
            clean_name,
            raw,
            max_pages=max_pdf_pages,
            max_bytes=max_pdf_bytes,
            progress=progress,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise ValueError(f"PDF could not be parsed: {exc}") from exc
    markdown = markdown.replace("\x00", "").replace("\r\n", "\n")
    markdown = markdown.replace("\r", "\n").strip()
    if not markdown:
        raise ValueError("PDF has no readable content")
    if len(markdown.encode("utf-8")) > max_markdown_bytes:
        raise ValueError(f"converted document exceeds {max_markdown_bytes} bytes")
    if progress:
        progress(
            "pdf_converted",
            {"char_count": len(markdown), "message": "PDF 已转换，正在切分章节"},
        )
    return PreparedDocument(clean_name, markdown, raw, "application/pdf")


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
        raw_content = str(content).strip()
        heading_only = next(
            (
                match.group(2).strip()
                for line in raw_content.splitlines()
                if (match := _HEADING_RE.match(line.strip()))
            ),
            "",
        )
        sections = [(heading_only, raw_content)]

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
        samples = [_even_sample(rows, allocation) for rows in by_document.values()]
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
        phrase_bonus = (
            1.0 if str(query).casefold() in passage.content.casefold() else 0.0
        )
        return (
            heading_overlap * 1.8 + content_overlap + phrase_bonus,
            -passage.ordinal,
            -len(passage.content),
        )

    ranked = sorted(passages, key=score, reverse=True)
    return ranked[:limit]


def full_document_read_requested(query: str) -> bool:
    """Return whether a question needs coverage across the attached documents."""

    return bool(_BROAD_DOCUMENT_RE.search(str(query or "")))


def limit_full_document_passages(
    passages: list[MarkdownPassage], limit: int
) -> list[MarkdownPassage]:
    """Keep every passage when possible, otherwise sample every document evenly."""

    if limit <= 0 or len(passages) <= limit:
        return passages
    by_document: dict[str, list[MarkdownPassage]] = {}
    for passage in passages:
        by_document.setdefault(passage.document_id, []).append(passage)
    allocation = max(1, math.ceil(limit / max(1, len(by_document))))
    samples = [_even_sample(rows, allocation) for rows in by_document.values()]
    selected: list[MarkdownPassage] = []
    for position in range(allocation):
        for rows in samples:
            if position < len(rows):
                selected.append(rows[position])
            if len(selected) >= limit:
                return selected
    return selected
