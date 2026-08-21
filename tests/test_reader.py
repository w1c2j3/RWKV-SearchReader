from rwkv_search_reader.models import Evidence
from rwkv_search_reader.reader import citation_warnings


def _evidence() -> list[Evidence]:
    return [
        Evidence(
            source_id="S1", title="A", url="https://example.com", claim="A", quote="A"
        )
    ]


def test_citation_audit_accepts_known_source() -> None:
    assert citation_warnings("Answer [S1]", _evidence()) == []


def test_citation_audit_reports_missing_and_unknown_sources() -> None:
    assert "no [S1]-style" in citation_warnings("Answer", _evidence())[0]
    assert "S9" in citation_warnings("Answer [S9]", _evidence())[0]
