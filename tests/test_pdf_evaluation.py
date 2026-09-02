from __future__ import annotations

from io import BytesIO

import pypdfium2 as pdfium

from rwkv_search_reader.pdf_evaluation import (
    DEFAULT_DATASET,
    _conversion_grade,
    _load_suite,
    _transform_pdf,
)


def _blank_pdf() -> bytes:
    document = pdfium.PdfDocument.new()
    document.new_page(320, 240)
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def test_pdf_reference_suite_covers_real_document_categories() -> None:
    documents, cases = _load_suite(DEFAULT_DATASET)

    assert len(documents) == 8
    assert len(cases) == 9
    assert sum(len(case["turns"]) for case in cases) == 12
    assert {
        "ocr",
        "table",
        "code",
        "long_document",
        "chinese_ocr",
        "multiple_pdf",
    } <= {case["category"] for case in cases}


def test_pdf_transforms_can_make_scan_and_long_document_variants() -> None:
    source = _blank_pdf()

    scanned = _transform_pdf(source, {"scan_first_page": True})
    repeated = _transform_pdf(source, {"repeat_pages": 5})

    assert scanned.startswith(b"%PDF-")
    assert len(pdfium.PdfDocument(BytesIO(scanned))) == 1
    assert len(pdfium.PdfDocument(BytesIO(repeated))) == 5


def test_conversion_grade_normalizes_markdown_formatting() -> None:
    grade = _conversion_grade(
        {
            "conversion_must_include_all": ["5%", "1200 万人"],
            "conversion_must_include_any": ["GDP", "国内生产总值"],
        },
        "## 国内生产总值\n\n增长 **5%**，新增就业 1200 万人。",
    )

    assert grade["pass"] is True
