from __future__ import annotations

import base64

import pytest

from rwkv_search_reader import documents
from rwkv_search_reader.documents import (
    MarkdownPassage,
    normalize_markdown,
    prepare_document,
    rank_markdown_passages,
    split_markdown,
)


def test_markdown_is_split_by_heading_and_paragraph() -> None:
    content = """# 项目背景

这是背景信息。

## 用户偏好

用户喜欢回甘明显的茶叶。

## 结论

建议优先尝试铁观音和生普。
"""

    chunks = split_markdown(content, target_chars=600)

    assert chunks == [
        ("项目背景", "这是背景信息。"),
        ("项目背景 / 用户偏好", "用户喜欢回甘明显的茶叶。"),
        ("项目背景 / 结论", "建议优先尝试铁观音和生普。"),
    ]


def test_heading_only_document_keeps_heading_as_passage_title() -> None:
    assert split_markdown("## Dummy PDF file", target_chars=600) == [
        ("Dummy PDF file", "## Dummy PDF file")
    ]


def test_document_validation_accepts_only_nonempty_markdown() -> None:
    assert normalize_markdown("folder/notes.md", "\r\n# Notes\r\nText") == (
        "notes.md",
        "# Notes\nText",
    )


def test_pdf_is_converted_to_markdown_for_existing_rag_flow(monkeypatch) -> None:
    raw_pdf = b"%PDF-1.7\nminimal-test-payload"
    calls = []

    def convert(filename, content, *, max_pages, max_bytes):
        calls.append((filename, content, max_pages, max_bytes))
        return "# 报告\n\n| 指标 | 数值 |\n| --- | --- |\n| 准确率 | 91% |"

    monkeypatch.setattr(documents, "_convert_pdf_bytes", convert)

    prepared = prepare_document(
        "folder/report.pdf",
        content_base64=base64.b64encode(raw_pdf).decode("ascii"),
        max_markdown_bytes=10_000,
        max_pdf_bytes=20_000,
        max_pdf_pages=80,
    )

    assert prepared.filename == "report.pdf"
    assert prepared.source_bytes == raw_pdf
    assert prepared.source_mime == "application/pdf"
    assert "准确率" in prepared.markdown
    assert calls == [("report.pdf", raw_pdf, 80, 20_000)]


def test_pdf_upload_rejects_non_pdf_payload() -> None:
    with pytest.raises(ValueError, match="not a PDF"):
        prepare_document(
            "report.pdf",
            content_base64=base64.b64encode(b"not really a pdf").decode("ascii"),
            max_markdown_bytes=10_000,
            max_pdf_bytes=20_000,
            max_pdf_pages=80,
        )


def test_specific_document_query_ranks_matching_section_first() -> None:
    passages = [
        MarkdownPassage("doc_test123", "notes.md", 0, "背景", "一般背景信息"),
        MarkdownPassage(
            "doc_test123", "notes.md", 1, "部署", "公网部署使用临时转发地址"
        ),
        MarkdownPassage("doc_test123", "notes.md", 2, "测试", "运行单元测试"),
    ]

    ranked = rank_markdown_passages("如何公网部署？", passages, 2)

    assert ranked[0].heading == "部署"


def test_broad_document_summary_samples_the_whole_document() -> None:
    passages = [
        MarkdownPassage("doc_test123", "long.md", index, f"第{index}章", f"内容{index}")
        for index in range(20)
    ]

    ranked = rank_markdown_passages("总结全文", passages, 5)

    assert [item.ordinal for item in ranked] == [0, 5, 10, 14, 19]


def test_broad_summary_allocates_passages_across_all_documents() -> None:
    passages = [
        MarkdownPassage(f"doc_{document}123", f"{document}.md", index, "", "内容")
        for document in range(5)
        for index in range(4)
    ]

    ranked = rank_markdown_passages("概括全部文档", passages, 7)

    assert {item.document_id for item in ranked} == {
        "doc_0123",
        "doc_1123",
        "doc_2123",
        "doc_3123",
        "doc_4123",
    }
