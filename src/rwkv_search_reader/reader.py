from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, DecimalException, InvalidOperation
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from .config import Settings
from .model_client import OpenAICompatibleClient
from .models import (
    DocumentChunk,
    Evidence,
    EvidenceAudit,
    FieldEvidenceSelection,
    ResearchPlan,
    ResearchRecord,
    SearchHit,
    SearchReaderError,
)

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SCIENTIFIC_RE = re.compile(
    r"(?P<coefficient>[+-]?\d+(?:\.\d+)?)\s*"
    r"(?:[eE]\s*|(?:\\(?:times|cdot)|[×*])\s*10\s*\^\s*\{?\s*)"
    r"(?P<exponent>[+-]?\d+)\s*\}?"
)
_LATEX_FRACTION_RE = re.compile(
    r"\\(?:d?frac)\s*\{\s*(?P<numerator>[+-]?\d+(?:\.\d+)?)\s*\}"
    r"\s*\{\s*(?P<denominator>[+-]?\d+(?:\.\d+)?)\s*\}"
)
_NUMBER_SUFFIX_RE = re.compile(
    r"\s*(?P<suffix>"
    r"%|‰|万亿|千万|百万|十万|亿|万(?!分之)|千(?!分之)|百(?!分)|兆|"
    r"trillions?(?![a-z])|billions?(?![a-z])|"
    r"millions?(?![a-z])|thousands?(?![a-z])"
    r")",
    flags=re.IGNORECASE,
)
_NUMBER_SCALES = {
    "百": Decimal(100),
    "千": Decimal(1000),
    "万": Decimal(10000),
    "十万": Decimal(100000),
    "百万": Decimal(1000000),
    "千万": Decimal(10000000),
    "亿": Decimal(100000000),
    "兆": Decimal(1000000000000),
    "万亿": Decimal(1000000000000),
    "thousand": Decimal(1000),
    "thousands": Decimal(1000),
    "million": Decimal(1000000),
    "millions": Decimal(1000000),
    "billion": Decimal(1000000000),
    "billions": Decimal(1000000000),
    "trillion": Decimal(1000000000000),
    "trillions": Decimal(1000000000000),
}
_CONTEXT_REFERENCE_RE = re.compile(
    r"它|它们|他们|她们|这个|那个|这些|那些|这两个|那两个|二者|两者|上述|前者|后者|"
    r"该(?:模型|版本|公司|产品|事件|人物|项目)|"
    r"^(?:那|那么|然后|为什么|什么时候|哪天|具体|价格呢|日期呢|版本呢)"
)
_KNOWN_SECONDARY_HOSTS = (
    "wikipedia.org",
    "reddit.com",
    "zhihu.com",
    "ign.com",
    "ign.com.cn",
    "epicgames.com",
    "ourplay.net",
    "fun-game.online",
)
_FINAL_MODEL_STAGES = {
    "aggregate",
    "conversation",
    "document",
    "finalize",
    "memory",
}
_SCOPE_ACCEPTANCE_RE = re.compile(
    r"^(?:都需要|都要|全部|全都要|所有|这些都要|就这些|都可以|都行)[。.!！?？]*$"
)
_UNEXPANDED_COLLECTION_RE = re.compile(
    r"(?:旗下|的)[^，。！？!?]{0,16}(?:游戏|产品|模型|项目|作品|应用)"
    r"[^，。！？!?]{0,20}(?:都|分别|各自|所有)|"
    r"(?:各类|各个|所有|全部)[^，。！？!?]{0,16}"
    r"(?:游戏|产品|模型|项目|作品|应用)"
)


def _today() -> str:
    return datetime.now().astimezone().date().isoformat()


def _normalized_text(value: str) -> str:
    return re.sub(r"\W+", "", str(value or "").casefold())


def _parse_source_summary(value: str) -> str:
    """Remove transport wrappers and exact sentence repetition only."""

    text = str(value or "").strip()
    text = re.sub(r"^\x60{3}(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\x60{3}$", "", text)
    text = re.sub(r"^\s*<(?:材料|摘要|summary)>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*</(?:材料|摘要|summary)>\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^\s*(?:summary|摘要|网页摘要)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*(?:根据(?:提供的|以上|上述)?网页内容[，,]?\s*)?"
        r"(?:以下是(?:对[^：:\n]{0,40})?(?:总结|概括)|总结如下)\s*[:：]\s*",
        "",
        text,
    )
    lines: list[str] = []
    seen_lines: set[str] = set()
    for line in text.splitlines():
        normalized = _normalized_text(line)
        if normalized and normalized in seen_lines:
            continue
        if normalized:
            seen_lines.add(normalized)
        lines.append(line.rstrip())
    text = "\n".join(lines).strip()
    sentences = re.split(r"(?<=[。！？!?])\s*", text.strip())
    unique: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = _normalized_text(sentence)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(sentence.strip())
    return "".join(unique).strip()


def _remove_copied_temporal_qualifiers(summary: str, material: str) -> str:
    """Do not let a per-source reader copy recency claims from the question."""

    cleaned = summary
    for qualifier in ("当前", "最新", "目前", "现阶段"):
        if qualifier not in material:
            cleaned = cleaned.replace(f"{qualifier}版本", "版本")
            cleaned = cleaned.replace(f"{qualifier}福利", "福利")
            cleaned = cleaned.replace(f"{qualifier}活动", "活动")
    return cleaned


def _claim_terms(value: str) -> set[str]:
    lowered = str(value or "").casefold()
    terms = set(re.findall(r"[a-z][a-z0-9_.+-]{1,}", lowered))
    for segment in re.findall(r"[\u3400-\u9fff]{2,}", lowered):
        terms.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return terms


def _is_irrelevant_summary(value: str) -> bool:
    return _normalized_text(value) in {
        "无相关信息",
        "未提及",
        "没有提及",
        "材料中未找到相关信息",
        "norelevantinformation",
    }


def _decimal_key(value: Decimal) -> str:
    return str(value.normalize())


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _normalize_numeric_markup(value: str) -> str:
    """Normalize Unicode and numeric LaTeX without changing prose semantics."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace(r"\%", "%")
    text = re.sub(r"(?<=\d)\s*\{\s*,\s*\}\s*(?=\d{3}(?:\D|$))", ",", text)
    text = re.sub(r"(?<=\d)\\,\s*(?=\d{3}(?:\D|$))", ",", text)

    def replace_fraction(match: re.Match[str]) -> str:
        try:
            denominator = Decimal(match.group("denominator"))
            if not denominator:
                return match.group(0)
            result = Decimal(match.group("numerator")) / denominator
        except DecimalException:
            return match.group(0)
        return _decimal_text(result)

    def replace_scientific(match: re.Match[str]) -> str:
        try:
            exponent = int(match.group("exponent"))
            if abs(exponent) > 1000:
                return match.group(0)
            coefficient = Decimal(match.group("coefficient"))
            result = coefficient * (Decimal(10) ** exponent)
        except (DecimalException, ValueError):
            return match.group(0)
        return _decimal_text(result)

    text = _LATEX_FRACTION_RE.sub(replace_fraction, text)
    return _SCIENTIFIC_RE.sub(replace_scientific, text)


def _normalized_number_pairs(value: str) -> list[tuple[str, str]]:
    value = _normalize_numeric_markup(value)
    value = re.sub(r"(?m)^\s*(?:#{1,6}\s*)?\d{1,3}[.)、]\s+", "", value)
    normalized: list[tuple[str, str]] = []
    for match in _NUMBER_RE.finditer(value):
        raw = match.group(0)
        try:
            number = Decimal(raw.replace(",", "")).normalize()
        except InvalidOperation:
            if raw.count(".") > 1:
                for part in raw.split("."):
                    if part.isdigit():
                        key = _decimal_key(Decimal(part))
                        normalized.append((key, key))
            continue
        suffix_match = _NUMBER_SUFFIX_RE.match(value, match.end())
        suffix = suffix_match.group("suffix").casefold() if suffix_match else ""
        try:
            if suffix in {"%", "‰"}:
                ratio_scale = Decimal("0.01") if suffix == "%" else Decimal("0.001")
                canonical = f"ratio:{_decimal_key(number * ratio_scale)}"
            else:
                canonical = _decimal_key(
                    number * _NUMBER_SCALES.get(suffix, Decimal(1))
                )
        except DecimalException:
            canonical = _decimal_key(number)
        normalized.append((canonical, _decimal_key(number)))
    return normalized


def _normalized_numbers(value: str) -> set[str]:
    return {canonical for canonical, _display in _normalized_number_pairs(value)}


def _unsupported_summary_numbers(summary: str, material: str) -> set[str]:
    material_numbers = _normalized_numbers(material)
    return {
        display
        for canonical, display in _normalized_number_pairs(summary)
        if canonical not in material_numbers
    }


def _remove_unsupported_number_sentences(summary: str, material: str) -> str:
    """Keep supported parts of a summary instead of discarding the whole page."""

    lines: list[str] = []
    for raw_line in str(summary or "").splitlines() or [str(summary or "")]:
        prefix = ""
        content = raw_line.strip()
        marker = re.match(r"^\s*((?:[-*+]\s+)|(?:\d{1,3}[.)、]\s+))(.*)$", raw_line)
        if marker:
            prefix, content = marker.group(1), marker.group(2).strip()
        kept = [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？!?；;])\s*", content)
            if sentence.strip()
            and not _unsupported_summary_numbers(sentence, material)
        ]
        if kept:
            lines.append(f"{prefix}{''.join(kept)}".rstrip())
    return "\n".join(lines).strip()


def _normalized_copy_text(value: str) -> str:
    def normalize_number(match: re.Match[str]) -> str:
        try:
            return str(Decimal(match.group(0).replace(",", "")).normalize())
        except InvalidOperation:
            return match.group(0)

    return _normalized_text(_NUMBER_RE.sub(normalize_number, str(value or "")))


def _copied_fact_supported(fact: str, material: str) -> bool:
    fact_key = _normalized_copy_text(fact)
    material_key = _normalized_copy_text(material)
    if fact_key in material_key:
        return True
    date_pattern = re.compile(
        r"(?<!\d)(\d{4})[年/.-](\d{1,2})[月/.-](\d{1,2})(?:日)?(?!\d)"
    )
    fact_dates = {
        tuple(int(value) for value in match)
        for match in date_pattern.findall(str(fact or ""))
    }
    material_dates = {
        tuple(int(value) for value in match)
        for match in date_pattern.findall(str(material or ""))
    }
    if fact_dates and not fact_dates <= material_dates:
        return False
    fact_terms = _claim_terms(fact)
    material_terms = _claim_terms(material)
    return bool(_normalized_numbers(fact)) and fact_terms <= material_terms


def _eligible_evidence(question: str, evidence: list[Evidence]) -> list[Evidence]:
    del question
    relevant = [item for item in evidence if not _is_irrelevant_summary(item.claim)]
    return relevant or evidence


def _render_research_records(records: list[ResearchRecord] | None) -> str:
    if not records:
        return "[]"
    return json.dumps(
        [
            {
                "subject": record.subject,
                "question": record.question,
                "fields": record.fields,
            }
            for record in records
        ],
        ensure_ascii=False,
    )


def _narrow_selected_fact(field: str, fact: str) -> str:
    """Keep a selector's extract limited to the requested field."""

    value = " ".join(str(fact or "").split()).strip()
    if "版本" in field:
        return re.split(r"[，；。]", value, maxsplit=1)[0].strip()
    if "主题" in field:
        match = re.search(r"主题(?:包括|为|是)?\s*[:：]?\s*(.+)", value)
        if match:
            return match.group(1).strip(" ，；。")
    if re.search(r"福利|奖励|补偿|赠送|活动", field):
        clauses = [
            clause.strip() for clause in re.split(r"[，；。]", value) if clause.strip()
        ]
        selected = [
            clause
            for clause in clauses
            if re.search(
                r"补偿|奖励|赠送|免费|领取|签到|兑换码|折扣|首充|登录奖励",
                clause,
            )
        ]
        if selected:
            return "；".join(selected[:3])
    return value


def _is_explicit_benefit_fact(value: str) -> bool:
    return bool(
        re.search(
            r"补偿|奖励|赠送|免费|领取|签到|兑换码|折扣|首充|登录奖励",
            str(value or ""),
        )
    )


def _bind_subject_citations(
    answer: str,
    selections: list[FieldEvidenceSelection],
) -> str:
    """Replace each subject clause's citations with its selected source set."""

    sources_by_subject: dict[str, list[str]] = {}
    for item in selections:
        if item.status != "supported":
            continue
        sources_by_subject.setdefault(item.subject, [])
        for source_id in _selection_source_ids(item):
            if source_id not in sources_by_subject[item.subject]:
                sources_by_subject[item.subject].append(source_id)
    parts = re.split(r"([。！？!?；;\n]+)", answer)
    for index in range(0, len(parts), 2):
        clause = parts[index]
        matched_ids: list[str] = []
        for subject, source_ids in sources_by_subject.items():
            if subject and subject in clause:
                for source_id in source_ids:
                    if source_id not in matched_ids:
                        matched_ids.append(source_id)
        if not matched_ids:
            continue
        plain = re.sub(r"\s*(?:\[S\d+\])+", "", clause, flags=re.IGNORECASE)
        citations = "".join(f"[{source_id}]" for source_id in matched_ids)
        parts[index] = f"{plain.rstrip()} {citations}"
    return "".join(parts)


def _bind_atomic_citations(
    answer: str,
    selections: list[FieldEvidenceSelection],
) -> str:
    """Bind each answer clause to matching atomic facts, never just the subject."""

    supported = [item for item in selections if item.status == "supported"]
    conflicts = [item for item in selections if item.status == "conflict"]
    parts = re.split(r"([。！？!?；;，,\n]+)", answer)
    for index in range(0, len(parts), 2):
        clause = parts[index]
        plain = re.sub(r"\s*(?:\[S\d+\])+", "", clause, flags=re.IGNORECASE)
        clause_numbers = _normalized_numbers(plain)
        clause_terms = _claim_terms(plain)
        matched_ids: list[str] = []
        for selection in supported:
            fact_numbers = _normalized_numbers(selection.fact)
            fact_terms = _claim_terms(
                f"{selection.subject} {selection.field} {selection.fact}"
            )
            number_match = bool(clause_numbers & fact_numbers)
            term_match = bool(fact_terms) and len(clause_terms & fact_terms) >= max(
                1, min(3, len(fact_terms) // 3)
            )
            if not (number_match if fact_numbers else term_match):
                continue
            for source_id in _selection_source_ids(selection):
                if source_id not in matched_ids:
                    matched_ids.append(source_id)
        if not matched_ids and re.search(r"存在冲突|无法确认|不能确定", plain):
            plain_terms = _claim_terms(plain)
            for selection in conflicts:
                identity_terms = _claim_terms(f"{selection.subject} {selection.field}")
                if identity_terms and not (plain_terms & identity_terms):
                    continue
                for source_id in _selection_source_ids(selection):
                    if source_id not in matched_ids:
                        matched_ids.append(source_id)
        if matched_ids:
            citations = "".join(f"[{source_id}]" for source_id in matched_ids)
            parts[index] = f"{plain.rstrip()} {citations}"
        else:
            # Never retain a model-invented citation.
            parts[index] = plain
    return "".join(parts)


def _filter_unsupported_atomic_clauses(
    answer: str, selections: list[FieldEvidenceSelection]
) -> tuple[str, list[str]]:
    """Remove final clauses that cannot be bound to supported atomic facts."""

    supported = [item for item in selections if item.status == "supported"]
    allowed_numbers = set().union(
        *(_normalized_numbers(item.fact) for item in supported), set()
    )
    parts = re.split(r"([。！？!?；;，,\n]+)", str(answer or ""))
    removed: list[str] = []
    for index in range(0, len(parts), 2):
        plain = re.sub(r"\s*(?:\[S\d+\])+", "", parts[index], flags=re.IGNORECASE)
        unsupported = _normalized_numbers(plain) - allowed_numbers
        clause_numbers = _normalized_numbers(plain)
        clause_terms = _claim_terms(plain)
        matched = False
        for item in supported:
            fact_numbers = _normalized_numbers(item.fact)
            fact_terms = _claim_terms(f"{item.subject} {item.field} {item.fact}")
            number_match = bool(clause_numbers & fact_numbers)
            term_match = bool(fact_terms) and len(clause_terms & fact_terms) >= max(
                1, min(3, len(fact_terms) // 3)
            )
            if number_match if fact_numbers else term_match:
                matched = True
                break
        uncertainty = bool(
            re.search(
                r"无法核实|无法确认|不能确定|未能确认|材料不足|存在冲突|信息缺失",
                plain,
            )
        )
        unsupported_clause = bool(unsupported) or bool(
            _normalized_text(plain)
            and not plain.rstrip().endswith((":", "："))
            and not uncertainty
            and not matched
        )
        if not unsupported_clause:
            continue
        removed.extend(sorted(unsupported) or ["unbound clause"])
        parts[index] = ""
        if index + 1 < len(parts):
            parts[index + 1] = ""
    cleaned = "".join(parts).strip()
    cleaned = re.sub(r"[，,；;：:]\s*$", "。", cleaned)
    warnings = (
        [
            "final atomic answer omitted unsupported or unbound clauses: "
            + ", ".join(dict.fromkeys(removed))
        ]
        if removed
        else []
    )
    return cleaned, warnings


def _atomic_fallback_answer(selections: list[FieldEvidenceSelection]) -> str:
    """Provide a source-bound safety answer only when final output is unusable."""

    lines: list[str] = []
    for item in selections:
        if item.status == "supported" and item.fact:
            citations = "".join(
                f"[{source_id}]" for source_id in _selection_source_ids(item)
            )
            lines.append(f"- {item.subject} · {item.field}：{item.fact} {citations}".rstrip())
        elif item.status == "conflict":
            citations = "".join(
                f"[{source_id}]" for source_id in _selection_source_ids(item)
            )
            lines.append(
                f"- {item.subject} · {item.field}：现有材料存在冲突，无法确认。{citations}"
            )
        else:
            lines.append(f"- {item.subject} · {item.field}：现有材料未能确认。")
    return "\n".join(lines).strip()


def _render_source_pairs(evidence: list[Evidence]) -> str:
    """Keep every RWKV summary inseparable from its source identifier."""

    return "\n".join(
        json.dumps(
            [
                item.source_id,
                {
                    "published_date": item.published_date or "unknown",
                    "summary": item.claim,
                },
            ],
            ensure_ascii=False,
        )
        for item in evidence
    )


def _bounded_audit_material(
    evidence: list[Evidence],
    max_chars: int,
    selections: list[FieldEvidenceSelection] | None = None,
) -> str:
    """Give the reviewer every source/field while enforcing a hard input bound."""

    if selections:
        return _bounded_field_material(selections, max_chars)[0]
    if not evidence:
        return ""
    overhead = sum(len(item.source_id) + len(item.published_date or "unknown") + 40 for item in evidence)
    per_claim = max(120, (max_chars - overhead) // max(1, len(evidence)))
    rows = "\n".join(
        json.dumps(
            [
                item.source_id,
                {
                    "published_date": item.published_date or "unknown",
                    "summary": item.claim[:per_claim],
                },
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for item in evidence
    )
    if len(rows) <= max_chars:
        return rows
    return rows[:max_chars].rsplit("\n", 1)[0]


@dataclass(slots=True)
class _MaterialNode:
    source_ids: tuple[str, ...]
    summary: str
    retrieval_point: str = ""


@dataclass(slots=True)
class _FieldFactCandidate:
    subject: str
    field: str
    source_id: str
    fact: str
    relevance: float
    title: str = ""
    published_date: str = ""


def _render_material_nodes(nodes: list[_MaterialNode]) -> str:
    """Render each reduced summary as an inseparable lineage-summary pair."""

    return "\n".join(
        json.dumps(
            [
                list(node.source_ids),
                (
                    f"检索点：{node.retrieval_point}\n{node.summary}"
                    if node.retrieval_point
                    else node.summary
                ),
            ],
            ensure_ascii=False,
        )
        for node in nodes
    )


def _selection_source_ids(selection: FieldEvidenceSelection) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *selection.source_ids,
                *([selection.source_id] if selection.source_id else []),
            ]
        )
    )


def _render_field_selections(
    selections: list[FieldEvidenceSelection], fact_chars: int | None = None
) -> str:
    """Keep a selected value bound to its subject, field, and source lineage."""

    subject_ids: dict[str, str] = {}
    for selection in selections:
        subject_ids.setdefault(selection.subject, f"R{len(subject_ids) + 1}")
    header = json.dumps(
        {"records": [[record_id, subject] for subject, record_id in subject_ids.items()]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    rows = "\n".join(
        json.dumps(
            [
                _selection_source_ids(selection),
                {
                    "record": subject_ids[selection.subject],
                    "field": selection.field,
                    "status": selection.status,
                    "fact": (
                        selection.fact[:fact_chars]
                        if fact_chars is not None
                        else selection.fact
                    ),
                },
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for selection in selections
    )
    return f"{header}\n{rows}" if rows else header


def _bounded_field_material(
    selections: list[FieldEvidenceSelection], max_chars: int
) -> tuple[str, bool]:
    """Fit every atomic field into the final context without repeating subjects."""

    rendered = _render_field_selections(selections)
    if len(rendered) <= max_chars:
        return rendered, False
    high = max((len(item.fact) for item in selections), default=0)
    low = 0
    best = _render_field_selections(selections, fact_chars=0)
    while low <= high:
        middle = (low + high) // 2
        candidate = _render_field_selections(selections, fact_chars=middle)
        if len(candidate) <= max_chars:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if len(best) > max_chars:
        # The configured budget can only be exceeded by adversarially long field
        # names; keep a syntactically complete prefix rather than a broken JSON row.
        lines = best.splitlines()
        kept: list[str] = []
        for line in lines:
            if kept and len("\n".join([*kept, line])) > max_chars:
                break
            kept.append(line)
        best = "\n".join(kept)
    return best, True


def _field_binding_applicable(
    question: str, records: list[ResearchRecord] | None
) -> bool:
    """Use atomic records for bounded facts, not open-ended or procedural tasks."""

    if not records or not all(record.fields for record in records):
        return False
    if _broad_question(question):
        return False
    return not bool(
        re.search(
            r"怎样|怎么|如何|示例|例子|步骤|教程|配置|代码|写法",
            str(question or ""),
            flags=re.IGNORECASE,
        )
    )


def _subject_aliases(subject: str) -> list[str]:
    aliases = [str(subject or "").strip()]
    titled = re.search(r"《[^》]{1,120}》", subject)
    if titled:
        aliases.append(titled.group(0))
    stripped = re.sub(
        r"正式服|当前|目前|最新|下一个|下一|最近一次|最新稳定版|版本|"
        r"的(?:完整)?信息$",
        " ",
        subject,
    )
    aliases.append(" ".join(stripped.split()))
    return list(
        dict.fromkeys(alias for alias in aliases if len(_normalized_text(alias)) >= 2)
    )


def _is_missing_fact_output(value: str) -> bool:
    return _normalized_text(value) in {
        "none",
        "missing",
        "无相关信息",
        "未提及",
        "没有",
        "没有提及",
    }


def _field_fact_signature(field: str, fact: str) -> str:
    """Cluster equivalent short field values without merging different fields."""

    if re.search(r"日期|时间|上线|发布|上市", field):
        full_date = re.search(
            r"(?<!\d)(\d{4})[年/.-](\d{1,2})[月/.-](\d{1,2})(?:日)?(?!\d)",
            fact,
        )
        if full_date:
            return "date:" + "-".join(
                str(int(value)) for value in full_date.groups()
            )
    numbers = sorted(_normalized_numbers(fact))
    if numbers:
        return "numbers:" + ",".join(numbers)
    return "text:" + _normalized_text(fact)


def _fact_date_specificity(value: str) -> int:
    if re.search(r"(?<!\d)\d{4}[年/.-]\d{1,2}[月/.-]\d{1,2}(?:日)?(?!\d)", value):
        return 2
    if re.search(r"(?<!\d)\d{1,2}月\d{1,2}日(?!\d)", value):
        return 1
    return 0


def _version_title_fact(value: str) -> str:
    match = re.search(
        r"(?<!\d)(\d+(?:\.\d+)+)\s*版本\s*[「“\"]([^」”\"]{1,100})[」”\"]",
        str(value or ""),
    )
    if not match:
        return ""
    return f"{match.group(1)}版本「{match.group(2)}」"


def _version_order(value: str) -> tuple[int, ...] | None:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)+)(?!\d)", str(value or ""))
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


def _chunk_material_nodes(
    nodes: list[_MaterialNode], *, max_chars: int, fan_in: int
) -> list[list[_MaterialNode]]:
    """Bound every reducer input by both lineage count and rendered size."""

    groups: list[list[_MaterialNode]] = []
    current: list[_MaterialNode] = []
    current_chars = 0
    for node in nodes:
        rendered_chars = len(_render_material_nodes([node]))
        exceeds = current and (
            len(current) >= fan_in or current_chars + rendered_chars > max_chars
        )
        if exceeds:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(node)
        current_chars += rendered_chars
    if current:
        groups.append(current)
    return groups


def _chunk_material_nodes_by_retrieval_point(
    nodes: list[_MaterialNode], *, max_chars: int, fan_in: int
) -> list[list[_MaterialNode]]:
    """Finish each search point before any cross-point funnel reduction."""

    point_groups: dict[str, list[_MaterialNode]] = {}
    for node in nodes:
        point_groups.setdefault(node.retrieval_point, []).append(node)
    if not any(point and len(items) > 1 for point, items in point_groups.items()):
        return _chunk_material_nodes(nodes, max_chars=max_chars, fan_in=fan_in)
    groups: list[list[_MaterialNode]] = []
    for items in point_groups.values():
        groups.extend(
            _chunk_material_nodes(items, max_chars=max_chars, fan_in=fan_in)
        )
    return groups


def _combined_retrieval_point(nodes: list[_MaterialNode]) -> str:
    return " | ".join(
        dict.fromkeys(node.retrieval_point for node in nodes if node.retrieval_point)
    )


def _retrieval_points_collapsed(nodes: list[_MaterialNode]) -> bool:
    points = [node.retrieval_point for node in nodes]
    return bool(points) and all(points) and len(set(points)) == len(points)


def _source_relevance(question: str, evidence: Evidence) -> float:
    question_terms = _claim_terms(question)
    source_terms = _claim_terms(f"{evidence.title} {evidence.claim}")
    coverage = len(question_terms & source_terms) / max(1, len(question_terms))
    return coverage * 0.65 + max(0.0, min(1.0, evidence.score)) * 0.35


def _recency_requested(question: str) -> bool:
    return bool(
        re.search(
            r"今日|今天|当前|现在|最新|刚刚|本周|这周|近期|最近|本月|今年",
            question,
        )
    )


def _broad_question(question: str) -> bool:
    return bool(
        re.search(
            r"哪些|有哪些|推荐|比较|对比|总结|综述|盘点|动态|新闻|怎么选|如何选",
            str(question or ""),
            flags=re.IGNORECASE,
        )
    )


def _valid_discovered_subject(question: str, subject: str) -> bool:
    """Reject field values and public events masquerading as set members."""

    value = " ".join(str(subject or "").split()).strip()
    if not value:
        return False
    game_set_question = bool(
        re.search(r"二游|二次元(?:游戏|手游)|游戏(?:本体|版本|主题)", question)
    )
    event_or_field = bool(
        re.search(
            r"(?:动漫|动画|游戏)(?:节|展|展会|大会)$|"
            r"(?:发布会|见面会|赛事|比赛|活动|版本|主题|排行榜)$",
            value,
        )
    )
    return not (game_set_question and event_or_field)


def _published_timestamp(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed.timestamp()


def _final_evidence_order(question: str, evidence: list[Evidence]) -> list[Evidence]:
    """Place the strongest pair nearest the question while retaining all pairs."""

    def order_group(group: list[Evidence]) -> list[Evidence]:
        if not _recency_requested(question):
            return sorted(
                group,
                key=lambda item: (_source_relevance(question, item), item.score),
            )
        return sorted(
            group,
            key=lambda item: (
                _published_timestamp(item.published_date) is not None,
                _published_timestamp(item.published_date) or 0.0,
                _source_relevance(question, item),
                item.score,
            ),
        )

    if not any(item.retrieval_point for item in evidence):
        return order_group(evidence)
    point_groups: dict[str, list[Evidence]] = {}
    for item in evidence:
        point_groups.setdefault(item.retrieval_point, []).append(item)
    return [
        item
        for group in point_groups.values()
        for item in order_group(group)
    ]


def _verified_source_relation(url: str, model_relation: str) -> str:
    host = (urlparse(url).hostname or "").casefold().removeprefix("www.")
    if any(
        host == value or host.endswith(f".{value}") for value in _KNOWN_SECONDARY_HOSTS
    ):
        return "secondary"
    return model_relation if model_relation in {"official", "secondary"} else "unclear"


def _clean_final_answer(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\x60{3}(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\x60{3}$", "", text)
    text = re.sub(
        r"^\s*(?:最终答案|最终回答|回答|分支总结|汇总材料)\s*[:：]\s*",
        "",
        text,
    )
    text = re.sub(r"^\s*根据(?:以上|上述|提供的)?材料[，,:：]\s*", "", text)
    text = re.sub(
        r"(^|[。！？!?]\s*)材料来源\s*[:：][^。！？!?]*(?:[。！？!?]|$)",
        r"\1",
        text,
    )
    text = re.sub(
        r"(?m)^\s*[-*]?\s*[^。\n]{0,40}来源于材料\s*\d[^\n]*$",
        "",
        text,
    )
    text = re.sub(
        r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?(?:材料)?来源(?:\*\*)?\s*[:：]"
        r"(?:[、，,;；\s]*(?:\[\s*S\d+\s*\])*)*$",
        "",
        text,
    )
    text = re.sub(
        r"\[\s*s(\d+)\s*\]",
        lambda match: f"[S{match.group(1)}]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"([。！？!?；;])\s*((?:\[S\d+\])+)",
        r" \2\1",
        text,
    )
    return _deduplicate_final_output(text).strip()


def _deduplicate_final_output(value: str) -> str:
    """Remove exact generation loops without rewriting the model's answer."""

    lines: list[str] = []
    seen: set[str] = set()
    in_code = False
    raw_lines = str(value or "").splitlines()
    last_content = max(
        (index for index, line in enumerate(raw_lines) if line.strip()), default=-1
    )
    for index, raw_line in enumerate(raw_lines):
        line = raw_line.rstrip()
        if line.lstrip().startswith("```"):
            in_code = not in_code
            lines.append(line)
            continue
        if in_code:
            lines.append(line)
            continue

        line = _deduplicate_delimited_fragments(line)
        plain = re.sub(r"\[S\d+\]", "", line, flags=re.IGNORECASE)
        plain = re.sub(r"^\s*(?:[-*+]\s+|\d{1,3}[.)、]\s*)", "", plain)
        normalized = _normalized_text(plain)
        if normalized and (
            normalized in seen
            or (
                index == last_content
                and len(normalized) >= 6
                and any(previous.startswith(normalized) for previous in seen)
            )
        ):
            continue
        if normalized:
            seen.add(normalized)
        if index == last_content and _has_unbalanced_delimiters(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _deduplicate_delimited_fragments(line: str) -> str:
    """Collapse a repeated comma-delimited list emitted on a single line."""

    if line.count("、") < 3:
        return line
    fragments = line.split("、")
    unique: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        if _has_unbalanced_delimiters(fragment):
            continue
        plain = re.sub(r"\[S\d+\]", "", fragment, flags=re.IGNORECASE)
        normalized = _normalized_text(plain)
        if normalized and normalized in seen:
            continue
        if normalized:
            seen.add(normalized)
        unique.append(fragment)
    return "、".join(unique)


def _has_unbalanced_delimiters(value: str) -> bool:
    """Drop only visibly truncated list tails produced at the token boundary."""

    text = re.sub(r"\[S\d+\]", "", value, flags=re.IGNORECASE)
    return (
        text.count("（") != text.count("）")
        or text.count("(") != text.count(")")
        or text.count("**") % 2 != 0
    )


def _clean_standalone_question(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\x60{3}(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\x60{3}$", "", text)
    text = re.sub(
        r"^\s*(?:独立问题|检索问题|改写后问题)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line[:8000] or fallback


def _needs_conversation_subject(value: str) -> bool:
    return bool(_CONTEXT_REFERENCE_RE.search(str(value or "").strip()))


def _parse_audit_output(value: str, max_queries: int) -> EvidenceAudit:
    text = str(value or "").strip()
    text = re.sub(r"^\x60{3}(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\x60{3}$", "", text)
    if re.fullmatch(r"\s*ENOUGH[。.!！]?\s*", text, flags=re.IGNORECASE):
        return EvidenceAudit(sufficient=True)
    queries: list[str] = []
    for line in text.splitlines():
        match = re.match(r"\s*SEARCH\s*[:：]\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if not match:
            continue
        query = " ".join(match.group(1).split()).strip(" -•")[:800]
        if query and query.casefold() not in {item.casefold() for item in queries}:
            queries.append(query)
        if len(queries) >= max_queries:
            break
    if not queries:
        raise SearchReaderError("audit", "RWKV returned an invalid evidence audit")
    return EvidenceAudit(sufficient=False, queries=queries)


def _json_object_from_output(value: str, stage: str) -> dict[str, object]:
    text = str(value or "").strip()
    text = re.sub(r"^\x60{3}(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\x60{3}$", "", text)
    start = text.find("{")
    if start < 0:
        raise SearchReaderError(stage, "RWKV returned no JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise SearchReaderError(stage, "RWKV returned an invalid JSON object") from exc
    if not isinstance(payload, dict):
        raise SearchReaderError(stage, "RWKV returned a non-object plan")
    return payload


def _parse_research_plan_output(
    value: str,
    fallback_question: str,
    max_queries: int,
) -> ResearchPlan:
    payload = _json_object_from_output(value, "plan")
    decision = str(payload.get("decision") or "").strip().casefold()
    if decision not in {"search", "clarify"}:
        raise SearchReaderError("plan", "RWKV returned an unknown plan decision")
    scope_mode = str(payload.get("scope_mode") or "fixed").strip().casefold()
    if scope_mode not in {"fixed", "discover"}:
        scope_mode = "fixed"

    records: list[ResearchRecord] = []
    prompt_placeholders = {
        "明确对象",
        "材料中出现的具体对象",
        "该记录需要回答的问题",
        "字段",
        "用户要求的字段",
    }
    for raw in payload.get("records") or []:
        if not isinstance(raw, dict):
            continue
        subject = " ".join(str(raw.get("subject") or "").split()).strip()[:200]
        question = " ".join(str(raw.get("question") or "").split()).strip()[:800]
        if subject in prompt_placeholders:
            subject = ""
        if question in prompt_placeholders:
            question = ""
        fields = [
            " ".join(str(item).split()).strip()[:80]
            for item in (raw.get("fields") or raw.get("required_fields") or [])
            if " ".join(str(item).split()).strip()
            not in prompt_placeholders
        ][:8]
        if subject or question:
            records.append(
                ResearchRecord(
                    subject=subject or question[:120],
                    question=question or fallback_question,
                    fields=list(dict.fromkeys(fields)),
                )
            )
        if len(records) >= max_queries:
            break

    queries = list(
        dict.fromkeys(
            " ".join(str(item).split()).strip()[:800]
            for item in (payload.get("queries") or [])
            if " ".join(str(item).split()).strip()
        )
    )[:max_queries]
    goal = " ".join(str(payload.get("goal") or fallback_question).split()).strip()
    clarification = " ".join(str(payload.get("clarification") or "").split()).strip()[
        :1200
    ]
    if decision == "clarify":
        if not clarification:
            raise SearchReaderError("plan", "RWKV clarification plan has no question")
        return ResearchPlan(
            decision="clarify",
            goal=goal or fallback_question,
            records=records,
            scope_mode="fixed",
            clarification=clarification,
        )
    return ResearchPlan(
        decision="search",
        goal=goal or fallback_question,
        records=records,
        queries=queries or [fallback_question],
        scope_mode=scope_mode,
    )


def _unexpanded_collection_plan(question: str, plan: ResearchPlan) -> bool:
    """Keep unnamed collective requests on the search-point/funnel path."""

    if plan.scope_mode == "discover":
        return True
    return bool(_UNEXPANDED_COLLECTION_RE.search(str(question or "")))


def _support_score(claim: str, evidence: Evidence) -> float:
    material = f"{evidence.title} {evidence.claim}"
    claim_terms = _claim_terms(claim)
    material_terms = _claim_terms(material)
    term_coverage = len(claim_terms & material_terms) / max(1, len(claim_terms))
    claim_numbers = {value.replace(",", "") for value in _NUMBER_RE.findall(claim)}
    material_numbers = {
        value.replace(",", "") for value in _NUMBER_RE.findall(material)
    }
    number_coverage = (
        len(claim_numbers & material_numbers) / len(claim_numbers)
        if claim_numbers
        else term_coverage
    )
    return term_coverage * 0.35 + number_coverage * 0.6 + evidence.score * 0.05


def _attach_missing_citations(answer: str, evidence: list[Evidence]) -> str:
    """Bind uncited answer clauses to summaries without rewriting model prose."""

    if not answer or not evidence:
        return answer
    segments = re.split(r"(```[\s\S]*?```)", answer)
    if len(segments) > 1:
        return "".join(
            re.sub(r"\s*\[S\d+\]", "", segment, flags=re.IGNORECASE)
            if segment.lstrip().startswith("```")
            else _attach_missing_citations(segment, evidence)
            for segment in segments
        )
    allowed = {item.source_id for item in evidence}
    parts = re.split(r"([。！？!?；;\n]+)", answer)
    for index in range(0, len(parts), 2):
        clause = parts[index]
        if not _normalized_text(clause) or clause.rstrip().endswith((":", "：")):
            continue
        referenced = [
            item.upper()
            for item in re.findall(r"\[(S\d+)\]", clause, flags=re.IGNORECASE)
            if item.upper() in allowed
        ]
        plain_clause = re.sub(r"\s*(?:\[S\d+\])+", "", clause, flags=re.IGNORECASE)
        if re.search(
            r"无法核实|无法确认|不能确定|尚未(?:提供|明确|写出)|"
            r"未(?:提供|明确|写出|找到|核实)|材料不足|存在冲突|信息缺失",
            plain_clause,
        ):
            continue
        clause_terms = _claim_terms(plain_clause)
        selected = [item for item in evidence if item.source_id in set(referenced)]
        covered_terms = set().union(
            *(
                clause_terms & _claim_terms(f"{item.title} {item.claim}")
                for item in selected
            ),
            set(),
        )
        ranked = sorted(
            evidence,
            key=lambda item: (_support_score(plain_clause, item), item.score),
            reverse=True,
        )
        for candidate in ranked:
            if candidate.source_id in {item.source_id for item in selected}:
                continue
            support = _support_score(plain_clause, candidate)
            candidate_terms = clause_terms & _claim_terms(
                f"{candidate.title} {candidate.claim}"
            )
            new_terms = candidate_terms - covered_terms
            clause_numbers = _normalized_numbers(plain_clause)
            candidate_numbers = _normalized_numbers(candidate.claim)
            covered_numbers = set().union(
                *(_normalized_numbers(item.claim) for item in selected), set()
            )
            adds_number = bool((clause_numbers & candidate_numbers) - covered_numbers)
            if not selected:
                if support < 0.08:
                    break
            elif not adds_number and len(new_terms) < max(
                2, round(len(clause_terms) * 0.12)
            ):
                continue
            selected.append(candidate)
            covered_terms.update(candidate_terms)
            if len(selected) >= 2:
                break
        if selected:
            source_ids = sorted(
                {item.source_id for item in selected},
                key=lambda value: int(value[1:]),
            )
            citations = "".join(f"[{source_id}]" for source_id in source_ids)
            parts[index] = f"{plain_clause.rstrip()} {citations}"
    return "".join(parts)


def _context_retry_tokens(error: str, requested: int) -> int | None:
    """Recover when a compatible endpoint rejects input + requested output."""

    match = re.search(
        r"maximum context length is\s+(\d+).*?requested\s+(\d+)\s+output tokens"
        r".*?at least\s+(\d+)\s+input tokens",
        str(error or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    context_tokens = int(match.group(1))
    input_tokens = int(match.group(3))
    available = context_tokens - input_tokens - 256
    conservative = requested // 2
    retry = min(available, conservative)
    if retry < 128 or retry >= requested:
        return None
    return retry


class RWKVReader:
    """Parallel page reading, chunk-bounded funnel reduction, then final summary."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._call_lock = threading.Lock()
        self._call_sequence = 0
        self._call_records: list[dict[str, object]] = []
        self._field_candidate_cache: dict[
            tuple[str, str, str], _FieldFactCandidate | None
        ] = {}

    def reset_call_records(self) -> None:
        with self._call_lock:
            self._call_sequence = 0
            self._call_records = []
            self._field_candidate_cache = {}

    @staticmethod
    def clarification_required(question: str) -> bool:
        return _needs_conversation_subject(question)

    def call_records(self) -> list[dict[str, object]]:
        with self._call_lock:
            return [dict(record) for record in self._call_records]

    def _next_call_id(self) -> int:
        with self._call_lock:
            if self._call_sequence >= self.settings.rwkv_max_calls_per_request:
                raise SearchReaderError(
                    "budget",
                    "request exceeded the configured RWKV call budget",
                )
            self._call_sequence += 1
            return self._call_sequence

    def _record_call(self, record: dict[str, object]) -> None:
        with self._call_lock:
            self._call_records.append(record)

    def _model_profile(self, stage: str) -> tuple[str, str, str]:
        if stage in _FINAL_MODEL_STAGES:
            return (
                self.settings.final_base_url,
                self.settings.final_api_key,
                self.settings.final_model,
            )
        return (
            self.settings.rwkv_base_url,
            self.settings.rwkv_api_key,
            self.settings.rwkv_model,
        )

    def _client(self, stage: str) -> OpenAICompatibleClient:
        base_url, api_key, model = self._model_profile(stage)
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=self.settings.model_timeout_seconds,
        )

    def _complete(self, user: str, *, max_tokens: int, stage: str) -> str:
        call_id = self._next_call_id()
        started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        started = time.perf_counter()
        base_url, _, model = self._model_profile(stage)
        protocol = (
            self.settings.final_protocol
            if stage in _FINAL_MODEL_STAGES
            else self.settings.rwkv_protocol
        )
        client = self._client(stage)
        try:
            if protocol == "chat":
                output = client.chat(
                    system="",
                    user=user,
                    max_tokens=max_tokens,
                    temperature=self.settings.rwkv_temperature,
                    stage=stage,
                )
            else:
                output = client.completion(
                    system="",
                    user=user,
                    max_tokens=max_tokens,
                    temperature=self.settings.rwkv_temperature,
                    stage=stage,
                )
            value = output.strip()
            if not value:
                raise SearchReaderError(stage, "RWKV returned an empty answer")
            self._record_call(
                {
                    "call_id": call_id,
                    "stage": stage,
                    "status": "ok",
                    "started_at": started_at,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    "model": model,
                    "base_url": base_url,
                    "protocol": protocol,
                    "max_tokens": max_tokens,
                    "temperature": self.settings.rwkv_temperature,
                    "system": "",
                    "input": user,
                    "output": value,
                }
            )
            return value
        except Exception as exc:
            self._record_call(
                {
                    "call_id": call_id,
                    "stage": stage,
                    "status": "error",
                    "started_at": started_at,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    "model": model,
                    "base_url": base_url,
                    "protocol": protocol,
                    "max_tokens": max_tokens,
                    "temperature": self.settings.rwkv_temperature,
                    "system": "",
                    "input": user,
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                }
            )
            raise

    def plan_research(self, question: str) -> ResearchPlan:
        """Let RWKV define scope and search records without choosing tools."""

        user = f"""当前日期：{_today()}
问题：{question}

把问题转换为检索计划。decision 只能是 search 或 clarify；公开信息用 search，只有指代无法确定时用 clarify。scope_mode 只能是 fixed 或 discover；明确列出的对象用 fixed，要求某主体旗下或某类别的多个对象但没有逐一列名时用 discover，records 留空。保留问题中的字段，最多生成 {self.settings.research_plan_max_queries} 个检索词。
只输出 JSON：
{{"decision":"","scope_mode":"","goal":"","records":[{{"subject":"","question":"","fields":[]}}],"queries":[],"clarification":""}}
检索计划 JSON："""
        output = self._complete(user, max_tokens=768, stage="plan")
        plan = _parse_research_plan_output(
            output,
            question,
            self.settings.research_plan_max_queries,
        )
        if plan.decision == "clarify" and not self.clarification_required(question):
            plan = ResearchPlan(
                decision="search",
                goal=plan.goal or question,
                records=[],
                queries=[question],
                scope_mode=(
                    "discover"
                    if re.search(
                        r"各个|所有|哪些|有哪些|某类|一类|类别|旗下|都分别|各自",
                        question,
                    )
                    else "fixed"
                ),
                fallback=True,
            )
        if plan.decision == "search" and _unexpanded_collection_plan(question, plan):
            suggested_points = list(
                dict.fromkeys(record.subject for record in plan.records if record.subject)
            )
            suggested_requirements = {
                record.subject: record.question
                for record in plan.records
                if record.subject and record.question
            }
            if len(suggested_points) < 2:
                suggested_points = []
                suggested_requirements = {}
            planned_by_point = {
                point: next(
                    (
                        query
                        for query in plan.queries
                        if _normalized_text(point) in _normalized_text(query)
                    ),
                    "",
                )
                for point in suggested_points
            }
            if suggested_points and all(planned_by_point.values()):
                plan.retrieval_points = suggested_points
                plan.queries = [
                    " ".join(
                        (
                            f"{planned_by_point[point]} "
                            f"{suggested_requirements.get(point, '')}"
                        ).split()
                    )[:800]
                    for point in suggested_points
                ]
            else:
                try:
                    plan.queries, plan.retrieval_points = self.design_search_points(
                        question,
                        suggested_points=suggested_points,
                        suggested_requirements=suggested_requirements,
                    )
                except SearchReaderError:
                    plan.queries = plan.queries or [question]
                    plan.retrieval_points = list(plan.queries)
                    plan.fallback = True
            plan.scope_mode = "discover"
            plan.records = []
        self._anchor_research_queries(question, plan)
        return plan

    def design_search_points(
        self,
        question: str,
        *,
        suggested_points: list[str] | None = None,
        suggested_requirements: dict[str, str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Give one RWKV call the single job of designing retrieval queries."""

        suggested_points = list(
            dict.fromkeys(
                " ".join(str(point).split()).strip()[:120]
                for point in (suggested_points or [])
                if " ".join(str(point).split()).strip()
            )
        )[: self.settings.research_plan_max_queries]
        requirements = {
            point: " ".join(
                str((suggested_requirements or {}).get(point) or "").split()
            ).strip()[:500]
            for point in suggested_points
        }
        point_hint = (
            "\n对象及检索要求："
            + "；".join(
                f"{point}（{requirements[point] or '覆盖用户所问信息'}）"
                for point in suggested_points
            )
            + "。逐个设计，不要增加公司名或信息类型作为检索点。"
            if suggested_points
            else ""
        )
        user = f"""当前日期：{_today()}
问题：{question}

设计最多 {self.settings.research_plan_max_queries} 个彼此不同的联网检索点。每个点给出一个简短对象名称和一个检索词；一个检索点只负责一个对象，检索词覆盖该对象所需信息。不要回答问题。
检索词必须包含该对象名称。涉及当前状态时优先设计能找到官方公告的检索词。{point_hint}
只输出 JSON：
{{"points":[["名称","检索词"]]}}
检索点 JSON："""
        output = self._complete(user, max_tokens=512, stage="search_design")
        payload = _json_object_from_output(output, "search_design")
        designed: list[tuple[str, str]] = []
        for item in payload.get("points") or []:
            if isinstance(item, dict):
                name = " ".join(str(item.get("name") or "").split()).strip()
                query = " ".join(str(item.get("query") or "").split()).strip()
            elif isinstance(item, list) and len(item) >= 2:
                name = " ".join(str(item[0]).split()).strip()
                query = " ".join(str(item[1]).split()).strip()
            else:
                continue
            if name and query:
                if suggested_points:
                    matched = next(
                        (
                            point
                            for point in suggested_points
                            if _normalized_text(point) in _normalized_text(name)
                            or _normalized_text(point) in _normalized_text(query)
                        ),
                        "",
                    )
                    if not matched:
                        continue
                    name = matched
                if _normalized_text(name) not in _normalized_text(query):
                    query = f"{name} {query}"
                requirement = requirements.get(name, "")
                if requirement:
                    query = f"{query} {requirement}"
                designed.append((name[:120], query[:800]))
        if not designed:
            fallback_queries = list(
                dict.fromkeys(
                    " ".join(str(item).split()).strip()[:800]
                    for item in (payload.get("queries") or [])
                    if " ".join(str(item).split()).strip()
                )
            )
            if suggested_points:
                designed = [
                    (
                        point,
                        next(
                            (
                                query
                                for query in fallback_queries
                                if _normalized_text(point)
                                in _normalized_text(query)
                            ),
                            f"{point} {question}"[:800],
                        ),
                    )
                    for point in suggested_points
                ]
            else:
                designed = [(query, query) for query in fallback_queries]
        if suggested_points:
            designed_by_name: dict[str, tuple[str, str]] = {}
            for name, query in designed:
                designed_by_name.setdefault(name, (name, query))
            designed = list(designed_by_name.values())
            present = {name for name, _ in designed}
            designed.extend(
                (
                    point,
                    f"{point} {requirements.get(point) or question}"[:800],
                )
                for point in suggested_points
                if point not in present
            )
        def complete_query(name: str, query: str) -> str:
            value = query
            if _normalized_text(name) not in _normalized_text(value):
                value = f"{name} {value}"
            requirement = requirements.get(name, "")
            if requirement and _normalized_text(requirement) not in _normalized_text(
                value
            ):
                value = f"{value} {requirement}"
            return " ".join(value.split())[:800]

        designed = [
            (name, complete_query(name, query)) for name, query in designed
        ]
        designed = designed[: self.settings.research_plan_max_queries]
        queries = list(dict.fromkeys(query for _, query in designed))
        point_by_query = {query: name for name, query in designed}
        if not queries:
            raise SearchReaderError(
                "search_design", "RWKV returned no usable search points"
            )
        return queries, [point_by_query[query] for query in queries]

    def _anchor_research_queries(
        self, question: str, plan: ResearchPlan
    ) -> ResearchPlan:
        """Add current-time terms without changing the model-selected scope."""

        if plan.decision == "search" and _recency_requested(question):
            current = datetime.now().astimezone()
            anchor = f"{current.year}年{current.month}月"
            anchored: list[str] = []
            for query in plan.queries:
                if str(current.year) not in query:
                    query = f"{query} {anchor} 最新"
                if re.search(r"版本|福利|活动|活動", f"{question} {query}"):
                    query = f"{query} 官方 更新公告 活动"
                anchored.append(" ".join(query.split()))
            plan.queries = anchored
        return plan

    def discover_research_scope(
        self, question: str, sources: list[SearchHit]
    ) -> ResearchPlan:
        """Map each page concurrently, then organize only verified subjects."""

        usable = [
            source
            for source in sources[: self.settings.search_max_results]
            if (source.content or source.snippet).strip()
        ]
        if not usable:
            raise SearchReaderError("scope_discovery", "no discovery material")

        def extract_candidates(source: SearchHit) -> tuple[SearchHit, list[str]]:
            material = (source.content or source.snippet).strip()[:3200]
            user = f"""{source.title}

{material}

问题：{question}
提取网页中属于问题所问类别的具体对象名称。版本、主题、活动、角色和道具不是对象。最多 4 个。
只输出 JSON：
{{"subjects":[]}}
候选对象 JSON："""
            output = self._complete(user, max_tokens=256, stage="scope_candidate")
            payload = _json_object_from_output(output, "scope_candidate")
            material_key = _normalized_text(f"{source.title} {material}")
            subjects: list[str] = []
            for item in payload.get("subjects") or []:
                subject = " ".join(str(item).split()).strip()[:160]
                titled = re.match(r"^(《[^》]{1,80}》)", subject)
                if titled:
                    subject = titled.group(1)
                subject_key = _normalized_text(subject)
                if (
                    not subject_key
                    or subject_key
                    in {
                        "二游",
                        "二次元游戏",
                        "二次元手游",
                        "二次元手机游戏",
                    }
                    or re.search(r"排行榜|推荐列表|游戏列表", subject)
                    or subject_key not in material_key
                    or not _valid_discovered_subject(question, subject)
                ):
                    continue
                if subject not in subjects:
                    subjects.append(subject)
                if len(subjects) >= 4:
                    break
            return source, subjects

        raw_candidate_rows: list[dict[str, object]] = []
        candidate_warnings: list[str] = []
        workers = min(self.settings.rwkv_summary_workers, len(usable))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(extract_candidates, source): source for source in usable
            }
            completed: dict[str, tuple[SearchHit, list[str]]] = {}
            for future in as_completed(futures):
                source = futures[future]
                try:
                    completed[source.source_id] = future.result()
                except SearchReaderError as exc:
                    candidate_warnings.append(
                        f"{source.source_id or source.title}: {exc}"[:500]
                    )
            for source in usable:
                row = completed.get(source.source_id)
                if not row:
                    continue
                for subject in row[1]:
                    material = (source.content or source.snippet).strip()
                    needle = subject.strip("《》")
                    position = material.find(needle)
                    start = max(0, position - 240) if position >= 0 else 0
                    raw_candidate_rows.append(
                        {
                            "subject": subject,
                            "source_id": source.source_id,
                            "source_title": source.title[:240],
                            "source_rank": usable.index(source),
                            "direct_title": _normalized_text(subject)
                            in _normalized_text(source.title),
                            "category_title": bool(
                                re.search(r"二次元|二游", source.title)
                            ),
                            "context": material[start : start + 720],
                        }
                    )
        if not raw_candidate_rows:
            detail = "; ".join(candidate_warnings[:3])
            raise SearchReaderError(
                "scope_discovery",
                f"RWKV found no material-backed subjects{': ' + detail if detail else ''}",
            )

        grouped: dict[str, dict[str, object]] = {}
        for row in raw_candidate_rows:
            subject = str(row["subject"])
            key = _normalized_text(subject)
            entry = grouped.setdefault(
                key,
                {
                    "subject": subject,
                    "source_ids": [],
                    "source_titles": [],
                    "contexts": [],
                    "source_rank": int(row["source_rank"]),
                    "direct_title_count": 0,
                    "category_title_count": 0,
                },
            )
            if len(subject) < len(str(entry["subject"])):
                entry["subject"] = subject
            source_id = str(row["source_id"])
            if source_id not in entry["source_ids"]:
                entry["source_ids"].append(source_id)
                entry["source_titles"].append(str(row["source_title"]))
                entry["contexts"].append(str(row["context"]))
                entry["direct_title_count"] = int(entry["direct_title_count"]) + int(
                    bool(row["direct_title"])
                )
                entry["category_title_count"] = int(
                    entry["category_title_count"]
                ) + int(bool(row["category_title"]))
            entry["source_rank"] = min(
                int(entry["source_rank"]), int(row["source_rank"])
            )
        trusted_category_rows = [
            row
            for row in grouped.values()
            if len(row["source_ids"]) >= 2 or int(row["category_title_count"]) >= 1
        ]
        credible_rows = [
            row
            for row in grouped.values()
            if int(row["direct_title_count"]) >= 1
            or int(row["category_title_count"]) >= 1
        ]
        candidate_rows = sorted(
            trusted_category_rows or credible_rows or grouped.values(),
            key=lambda row: (
                -len(row["source_ids"]),
                -int(row["direct_title_count"]),
                -int(row["category_title_count"]),
                int(row["source_rank"]),
                len(str(row["subject"])),
            ),
        )[:16]

        candidates_json = json.dumps(candidate_rows, ensure_ascii=False)
        user = f"""{candidates_json}

问题：{question}
从候选 subject 中原样选择最多 {self.settings.research_plan_max_queries} 个具体对象，为每个对象生成记录和检索词。保留问题中的字段，不新增对象。
只输出 JSON：
{{"decision":"search","scope_mode":"fixed","goal":"","records":[{{"subject":"","question":"","fields":[]}}],"queries":[],"clarification":""}}
检索计划 JSON："""
        output = self._complete(user, max_tokens=1024, stage="scope_discovery")
        plan = _parse_research_plan_output(
            output,
            question,
            self.settings.research_plan_max_queries,
        )
        canonical_subjects = {
            _normalized_text(row["subject"]): row["subject"] for row in candidate_rows
        }
        records: list[ResearchRecord] = []
        queries: list[str] = []
        seen_subjects: set[str] = set()
        for index, record in enumerate(plan.records):
            key = _normalized_text(record.subject)
            if key not in canonical_subjects or key in seen_subjects:
                continue
            seen_subjects.add(key)
            record.subject = canonical_subjects[key]
            records.append(record)
            query = (
                plan.queries[index]
                if index < len(plan.queries)
                else f"{record.subject} {record.question}"
            )
            queries.append(query)
        if not records:
            raise SearchReaderError(
                "scope_discovery", "RWKV found no material-backed subjects"
            )
        plan.decision = "search"
        plan.scope_mode = "fixed"
        plan.records = records
        plan.queries = queries
        plan.clarification = ""
        return self._anchor_research_queries(question, plan)

    @staticmethod
    def prior_facts(history: list[dict[str, object]]) -> list[dict[str, object]]:
        """Return the newest validated atomic state, not prose from an old answer."""

        for turn in reversed(history):
            state = turn.get("state")
            if not isinstance(state, dict):
                continue
            facts = state.get("facts")
            if isinstance(facts, list):
                return [item for item in facts if isinstance(item, dict)]
        return []

    def contextualize(self, question: str, history: list[dict[str, object]]) -> str:
        """Resolve follow-up references without treating prior answers as evidence."""

        if not history:
            return question
        if _SCOPE_ACCEPTANCE_RE.fullmatch(str(question or "").strip()):
            previous_question = str(history[-1].get("question") or "").strip()
            if previous_question:
                previous_question = previous_question.rstrip("。.!！?？")
                return (
                    f"{previous_question}；用户补充：公开资料能够确认的代表性对象都需要，"
                    "无需再次询问具体名单。"
                )[:8000]
        compact_history = history[-6:]
        material = "\n".join(
            json.dumps(
                {
                    "user": str(turn.get("question", ""))[:500],
                    "assistant": str(turn.get("answer", ""))[:700],
                },
                ensure_ascii=False,
            )
            for turn in compact_history
        )
        prior_facts = self.prior_facts(history)
        compact_facts = [
            {
                "subject": str(item.get("subject") or "")[:200],
                "field": str(item.get("field") or "")[:80],
                "status": str(item.get("status") or "")[:20],
                "fact": str(item.get("fact") or "")[:300],
            }
            for item in prior_facts[:32]
        ]
        user = f"""{material}

verified_state：{json.dumps(compact_facts, ensure_ascii=False)}

当前消息：{question}
结合此前对话和 verified_state，把当前消息改写成可独立理解的问题；“下一个”必须保留此前已确认对象和基准记录。只输出改写后的问题。
独立问题："""
        output = self._complete(user, max_tokens=160, stage="contextualize")
        rewritten = _clean_standalone_question(output, question)
        previous_answer = str(history[-1].get("answer") or "").strip()
        if previous_answer and _normalized_text(rewritten) == _normalized_text(
            previous_answer
        ):
            previous_question = str(history[-1].get("question") or "").strip()
            if previous_question:
                return f"上一问：{previous_question}；当前追问：{question}"[:8000]
        if _needs_conversation_subject(question) and (
            _normalized_text(rewritten) == _normalized_text(question)
            or _needs_conversation_subject(rewritten)
        ):
            if prior_facts:
                anchor = json.dumps(prior_facts, ensure_ascii=False)
                return f"此前已确认：{anchor}；当前追问：{question}"[:8000]
            previous_question = str(history[-1].get("question") or "").strip()
            if previous_question:
                previous_question = previous_question.rstrip("。.!！?？")
                return f"上一问：{previous_question}；当前追问：{question}"[:8000]
        return rewritten

    def answer_from_history(self, question: str, history: list[dict[str, object]]) -> str:
        """Answer a request about this conversation without external retrieval."""

        rows: list[str] = []
        total = 0
        for index, turn in reversed(list(enumerate(history, start=1))):
            row = json.dumps(
                {
                    "turn": index,
                    "user": str(turn.get("question", ""))[:500],
                    "assistant": str(turn.get("answer", ""))[:1000],
                },
                ensure_ascii=False,
            )
            if rows and total + len(row) > self.settings.rwkv_final_material_chars:
                break
            rows.append(row)
            total += len(row)
        material = "\n".join(reversed(rows))
        user = f"""{material}

请求：{question}
根据此前对话完成请求。合并重复内容，不复制来源编号。直接回答。
回答："""
        output = self._complete(
            user,
            max_tokens=self.settings.rwkv_max_tokens,
            stage="conversation",
        )
        answer = _clean_final_answer(output)
        answer = re.sub(
            r"^\s*(?:我已阅读(?:完)?(?:所有|上述|此前)?对话(?:消息|内容)?|"
            r"根据(?:上述|此前|前面)?对话(?:内容|消息)?|"
            r"以下是(?:对当前对话的)?总结)\s*[，,:：。]?\s*",
            "",
            answer,
        )
        answer = re.sub(r"\s*(?:\[S\d+\])+", "", answer, flags=re.IGNORECASE)
        if not answer:
            raise SearchReaderError(
                "conversation", "RWKV returned an empty conversation answer"
            )
        return answer

    def audit(
        self,
        question: str,
        evidence: list[Evidence],
        tried_queries: list[str],
        research_records: list[ResearchRecord] | None = None,
        field_selections: list[FieldEvidenceSelection] | None = None,
    ) -> EvidenceAudit:
        """Ask RWKV whether the accumulated material can answer the question."""

        material = _bounded_audit_material(
            evidence,
            min(
                self.settings.rwkv_reduce_chunk_chars,
                self.settings.rwkv_final_material_chars,
            ),
            field_selections,
        )
        tried = "\n".join(f"- {query}" for query in tried_queries)
        records = _render_research_records(research_records)
        user = f"""{material}

记录：{records}
问题：{question}
已检索：{tried}
当前日期：{_today()}

判断材料是否足以覆盖问题和记录中的字段。足够只输出 ENOUGH；不足输出最多 {self.settings.review_max_queries} 行新检索词，每行格式为 SEARCH: 检索词。
审核结果："""
        output = self._complete(user, max_tokens=192, stage="audit")
        audit = _parse_audit_output(output, self.settings.review_max_queries)
        missing_queries = [
            f"{item.subject} {item.field} 官方资料"
            for item in (field_selections or [])
            if item.status != "supported"
        ][: self.settings.review_max_queries]
        if audit.sufficient and missing_queries:
            return EvidenceAudit(sufficient=False, queries=missing_queries)
        return audit

    def _extract_one_field(
        self,
        record: ResearchRecord,
        field: str,
        evidence: Evidence,
        relevance: float,
    ) -> _FieldFactCandidate | None:
        """Ask one RWKV to copy one field from one webpage summary."""

        material = f"{evidence.title}\n{evidence.claim}".strip()
        if re.search(r"日期|时间|上线|发布|上市", field):
            request = (
                f"复制明确写着{record.subject}{field}的完整原句。"
                "相关活动、申购日期不算；没有则输出 NONE。"
            )
        else:
            request = (
                f"复制{record.subject}的{field}，只保留包含答案的最短原句。"
                "没有则输出 NONE。"
            )
        output = self._complete(
            f"{material}\n\n{request}\n字段原句：",
            max_tokens=min(256, self.settings.rwkv_summary_max_tokens),
            stage="evidence_extract",
        )
        fact = _parse_source_summary(output).strip()
        if _is_missing_fact_output(fact):
            return None

        if re.search(r"版本名称|版本名", field):
            titled = _version_title_fact(evidence.title)
            if titled:
                fact = titled
        if not _copied_fact_supported(fact, material):
            return None
        unsupported = _unsupported_summary_numbers(fact, material)
        if unsupported:
            values = ", ".join(sorted(unsupported))
            raise SearchReaderError(
                "evidence_extract",
                f"RWKV copied unsupported numbers: {values}",
            )
        return _FieldFactCandidate(
            subject=record.subject,
            field=field,
            source_id=evidence.source_id,
            fact=fact[:500],
            relevance=relevance,
            title=evidence.title,
            published_date=evidence.published_date,
        )

    @staticmethod
    def _fallback_field_selection(
        record: ResearchRecord,
        field: str,
        candidates: list[_FieldFactCandidate],
    ) -> FieldEvidenceSelection:
        if not candidates:
            return FieldEvidenceSelection(
                subject=record.subject,
                field=field,
                source_id="",
                status="missing",
                fact="材料未给出这个字段。",
            )
        groups: dict[str, list[_FieldFactCandidate]] = {}
        for candidate in candidates:
            groups.setdefault(
                _field_fact_signature(field, candidate.fact), []
            ).append(candidate)
        if len(groups) > 1:
            source_ids = list(
                dict.fromkeys(candidate.source_id for candidate in candidates)
            )
            return FieldEvidenceSelection(
                subject=record.subject,
                field=field,
                source_id="",
                status="conflict",
                fact="材料对这个字段给出了互相冲突的记录。",
                source_ids=source_ids,
            )
        selected_group = max(
            groups.values(),
            key=lambda group: (
                max(_fact_date_specificity(item.fact) for item in group)
                if re.search(r"日期|时间|上线|发布|上市", field)
                else 0,
                sum(item.relevance for item in group),
                len(group),
            ),
        )
        selected = max(
            selected_group,
            key=lambda item: (
                _fact_date_specificity(item.fact),
                item.relevance,
                -len(item.fact),
            ),
        )
        source_ids = list(
            dict.fromkeys(item.source_id for item in selected_group)
        )
        return FieldEvidenceSelection(
            subject=record.subject,
            field=field,
            source_id=selected.source_id,
            status="supported",
            fact=selected.fact,
            source_ids=source_ids,
        )

    def _resolve_record_fields(
        self,
        question: str,
        record: ResearchRecord,
        candidates_by_field: dict[str, list[_FieldFactCandidate]],
        prior_facts: list[dict[str, object]] | None = None,
    ) -> list[FieldEvidenceSelection]:
        all_candidates = [
            candidate
            for field in record.fields
            for candidate in candidates_by_field.get(field, [])
        ]
        fallback = {
            field: self._fallback_field_selection(
                record, field, candidates_by_field.get(field, [])
            )
            for field in record.fields
        }
        if not all_candidates:
            return [fallback[field] for field in record.fields]

        material = "\n".join(
            json.dumps(
                [
                    [candidate.source_id],
                    {
                        "subject": candidate.subject,
                        "field": candidate.field,
                        "fact": candidate.fact,
                    },
                ],
                ensure_ascii=False,
            )
            for candidate in all_candidates
        )
        temporal_rule = ""
        record_subject = _normalized_text(record.subject)
        relevant_prior = [
            item
            for item in (prior_facts or [])
            if record_subject
            and (
                record_subject in _normalized_text(str(item.get("subject") or ""))
                or _normalized_text(str(item.get("subject") or "")) in record_subject
            )
        ]
        prior_material = json.dumps(relevant_prior[:16], ensure_ascii=False)
        next_version_requested = bool(re.search(r"下一个|下一版|后续版本", question))
        prior_versions = [
            version
            for item in relevant_prior
            if (version := _version_order(str(item.get("fact") or "")))
        ]
        if next_version_requested:
            temporal_rule = (
                "“下一个”必须晚于此前已确认记录；如果没有比此前记录更晚的材料，就标记缺失。"
            )
        elif _recency_requested(question):
            temporal_rule = "“当前/最新”只选择当前日期前已经生效的同一记录。"
        user = f"""{material}

此前已确认：{prior_material}
当前日期：{_today()}
问题：{question}
整理成一个一致的记录。每个字段只使用该字段下的值，不交换字段。{temporal_rule}
只输出 JSON：{{"facts":[{{"field":"","fact":"","source_ids":[]}}]}}
记录 JSON："""
        output = self._complete(user, max_tokens=768, stage="evidence_resolve")
        payload = _json_object_from_output(output, "evidence_resolve")
        raw_facts = payload.get("facts") or []
        if not isinstance(raw_facts, list):
            raise SearchReaderError(
                "evidence_resolve", "RWKV returned a non-list fact record"
            )

        resolved: dict[str, list[FieldEvidenceSelection]] = {}
        field_lookup = {_normalized_text(field): field for field in record.fields}
        for raw in raw_facts:
            if not isinstance(raw, dict):
                continue
            field = field_lookup.get(_normalized_text(str(raw.get("field") or "")))
            if not field:
                continue
            candidates = candidates_by_field.get(field, [])
            allowed = {candidate.source_id: candidate for candidate in candidates}
            fact = " ".join(str(raw.get("fact") or "").split()).strip()[:500]
            if not fact:
                continue
            source_ids = list(
                dict.fromkeys(
                    str(item or "").strip().upper()
                    for item in (raw.get("source_ids") or [])
                    if str(item or "").strip().upper() in allowed
                )
            )
            if not source_ids:
                signature = _field_fact_signature(field, fact)
                source_ids = [
                    candidate.source_id
                    for candidate in candidates
                    if _field_fact_signature(field, candidate.fact) == signature
                ]
            if not source_ids:
                continue
            support_material = " ".join(
                f"{allowed[source_id].title} {allowed[source_id].fact}"
                for source_id in source_ids
            )
            if _unsupported_summary_numbers(fact, support_material):
                continue
            if not _copied_fact_supported(fact, support_material):
                continue
            if re.search(r"版本名称|版本名", field) and not _version_title_fact(fact):
                for source_id in source_ids:
                    titled = _version_title_fact(allowed[source_id].title)
                    if titled:
                        fact = titled
                        break
            if next_version_requested and re.search(r"版本", field):
                candidate_version = _version_order(fact)
                if (
                    prior_versions
                    and candidate_version
                    and candidate_version <= max(prior_versions)
                ):
                    continue
            resolved.setdefault(field, []).append(
                FieldEvidenceSelection(
                    subject=record.subject,
                    field=field,
                    source_id=source_ids[0],
                    status="supported",
                    fact=fact,
                    source_ids=source_ids,
                )
            )

        selections: list[FieldEvidenceSelection] = []
        for field in record.fields:
            options = resolved.get(field, [])
            if not options:
                selected = fallback[field]
                selected_version = _version_order(selected.fact)
                if (
                    next_version_requested
                    and re.search(r"版本", field)
                    and prior_versions
                    and selected_version
                    and selected_version <= max(prior_versions)
                ):
                    selected = FieldEvidenceSelection(
                        subject=record.subject,
                        field=field,
                        source_id="",
                        status="missing",
                        fact="材料未给出晚于此前已确认记录的下一个版本。",
                    )
                selections.append(selected)
                continue
            options.sort(
                key=lambda item: (
                    len(_selection_source_ids(item)),
                    _fact_date_specificity(item.fact),
                ),
                reverse=True,
            )
            selections.append(options[0])
        return selections

    def select_evidence(
        self,
        question: str,
        records: list[ResearchRecord],
        evidence: list[Evidence],
        on_progress: Callable[[str, dict[str, object]], None] | None = None,
        prior_facts: list[dict[str, object]] | None = None,
    ) -> tuple[list[FieldEvidenceSelection], list[str]]:
        """Map one webpage per call, then reduce only within a research record."""

        if not _field_binding_applicable(question, records):
            return [], []
        useful = [item for item in evidence if not _is_irrelevant_summary(item.claim)]
        useful = useful or evidence
        extracted: dict[tuple[int, str], list[_FieldFactCandidate]] = {}
        field_tasks: list[tuple[int, ResearchRecord, str, Evidence, float]] = []
        for record_index, record in enumerate(records):
            alias_keys = [_normalized_text(alias) for alias in _subject_aliases(record.subject)]
            related = [
                item
                for item in useful
                if any(
                    alias in _normalized_text(f"{item.title} {item.quote}")
                    for alias in alias_keys
                )
            ]
            related = related or useful
            for field in record.fields:
                ranked = sorted(
                    related,
                    key=lambda item: _source_relevance(
                        f"{record.subject} {field}", item
                    ),
                    reverse=True,
                )[:4]
                for item in ranked:
                    cache_key = (
                        _normalized_text(record.subject),
                        _normalized_text(field),
                        item.source_id,
                    )
                    if cache_key in self._field_candidate_cache:
                        cached = self._field_candidate_cache[cache_key]
                        if cached:
                            extracted.setdefault((record_index, field), []).append(cached)
                        continue
                    field_tasks.append(
                        (
                            record_index,
                            record,
                            field,
                            item,
                            _source_relevance(f"{record.subject} {field}", item),
                        )
                    )
        if not field_tasks and not extracted:
            return [], []

        warnings: list[str] = []
        workers = max(1, min(self.settings.rwkv_summary_workers, len(field_tasks)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._extract_one_field,
                    record,
                    field,
                    item,
                    relevance,
                ): (record_index, record, field, item)
                for record_index, record, field, item, relevance in field_tasks
            }
            for extracted_completed, future in enumerate(
                as_completed(futures), start=1
            ):
                record_index, record, field, item = futures[future]
                status = "missing"
                try:
                    candidate = future.result()
                except SearchReaderError as exc:
                    status = "degraded"
                    candidate = None
                    warnings.append(
                        f"{record.subject}/{field}/{item.source_id} field extraction degraded: {exc}"[
                            :500
                        ]
                    )
                cache_key = (
                    _normalized_text(record.subject),
                    _normalized_text(field),
                    item.source_id,
                )
                self._field_candidate_cache[cache_key] = candidate
                if candidate:
                    status = "found"
                    extracted.setdefault((record_index, field), []).append(candidate)
                if on_progress:
                    on_progress(
                        "evidence_page_checked",
                        {
                            "completed": extracted_completed,
                            "total": len(field_tasks),
                            "subject": record.subject,
                            "field": field,
                            "source_id": item.source_id,
                            "status": status,
                        },
                    )

        selections_by_record: dict[int, list[FieldEvidenceSelection]] = {}
        resolve_workers = min(self.settings.rwkv_summary_workers, len(records))
        with ThreadPoolExecutor(max_workers=resolve_workers) as pool:
            futures = {
                pool.submit(
                    self._resolve_record_fields,
                    question,
                    record,
                    {
                        field: extracted.get((record_index, field), [])
                        for field in record.fields
                    },
                    prior_facts,
                ): (record_index, record)
                for record_index, record in enumerate(records)
            }
            for resolved_completed, future in enumerate(
                as_completed(futures), start=1
            ):
                record_index, record = futures[future]
                try:
                    selections_by_record[record_index] = future.result()
                except SearchReaderError as exc:
                    warnings.append(
                        f"{record.subject} record resolution degraded: {exc}"[:500]
                    )
                    selections_by_record[record_index] = [
                        self._fallback_field_selection(
                            record,
                            field,
                            extracted.get((record_index, field), []),
                        )
                        for field in record.fields
                    ]
                if on_progress:
                    on_progress(
                        "evidence_record_resolved",
                        {
                            "completed": resolved_completed,
                            "total": len(records),
                            "subject": record.subject,
                        },
                    )

        selections = [
            selection
            for record_index in range(len(records))
            for selection in selections_by_record.get(record_index, [])
        ]
        for completed, selection in enumerate(selections, start=1):
            if on_progress:
                on_progress(
                    "evidence_field_selected",
                    {
                        "completed": completed,
                        "total": len(selections),
                        "subject": selection.subject,
                        "field": selection.field,
                        "source_id": selection.source_id,
                        "source_ids": _selection_source_ids(selection),
                        "status": selection.status,
                        "fact_preview": selection.fact[:180],
                    },
                )
        return selections, warnings

    def _summarize_one(self, _question: str, chunk: DocumentChunk) -> Evidence:
        material = chunk.text[: self.settings.max_evidence_chars].strip()
        focus = chunk.retrieval_query or chunk.retrieval_point
        instruction = (
            f"总结以上网页中与检索词“{focus}”有关的内容。"
            if focus
            else "总结以上网页内容。"
        )
        user = f"""{material}

{instruction}
网页摘要："""
        output = self._complete(
            user,
            max_tokens=self.settings.rwkv_summary_max_tokens,
            stage="summarize",
        )
        summary = _parse_source_summary(output)
        if not summary:
            raise SearchReaderError(
                "summarize", "RWKV returned an empty source summary"
            )
        if chunk.url.startswith("document://"):
            summary = material
        else:
            summary = _remove_copied_temporal_qualifiers(
                summary, f"{chunk.title}\n{material}"
            )
        unsupported = _unsupported_summary_numbers(
            summary, material
        )
        if unsupported:
            summary = _remove_unsupported_number_sentences(summary, material)
            if not summary:
                values = ", ".join(sorted(unsupported))
                raise SearchReaderError(
                    "summarize", f"RWKV introduced unsupported numbers: {values}"
                )
        return Evidence(
            source_id=chunk.source_id,
            title=chunk.title,
            url=chunk.url,
            claim=summary,
            quote=material,
            score=chunk.score,
            source_relation=_verified_source_relation(chunk.url, "unclear"),
            published_date=chunk.published_date,
            retrieval_point=chunk.retrieval_point,
            retrieval_query=chunk.retrieval_query,
        )

    def summarize(
        self,
        question: str,
        chunks: list[DocumentChunk],
        on_progress: Callable[[str, dict[str, object]], None] | None = None,
    ) -> tuple[list[Evidence], list[str]]:
        if not self.settings.rwkv_configured:
            raise SearchReaderError(
                "summarize", "RWKV endpoint or model is not configured"
            )
        if not chunks:
            raise SearchReaderError(
                "summarize", "there is no material for RWKV to read"
            )

        evidence_by_index: dict[int, Evidence] = {}
        warnings: list[str] = []
        workers = min(self.settings.rwkv_summary_workers, len(chunks))
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._summarize_one, question, chunk): (index, chunk)
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                index, chunk = futures[future]
                try:
                    evidence = future.result()
                    evidence_by_index[index] = evidence
                    status = "done"
                except SearchReaderError as exc:
                    material = chunk.text[: self.settings.max_evidence_chars]
                    evidence = Evidence(
                        source_id=chunk.source_id,
                        title=chunk.title,
                        url=chunk.url,
                        claim=material,
                        quote=material,
                        score=chunk.score,
                        source_relation="unclear",
                        published_date=chunk.published_date,
                        retrieval_point=chunk.retrieval_point,
                        retrieval_query=chunk.retrieval_query,
                    )
                    evidence_by_index[index] = evidence
                    status = "degraded"
                    warnings.append(
                        f"{chunk.source_id} summary degraded to original material: {exc}"[
                            :500
                        ]
                    )
                completed += 1
                if on_progress:
                    on_progress(
                        "source_read",
                        {
                            "source_id": chunk.source_id,
                            "title": chunk.title,
                            "url": chunk.url,
                            "completed": completed,
                            "total": len(chunks),
                            "status": status,
                            "source_relation": evidence.source_relation,
                            "retrieval_point": evidence.retrieval_point,
                            "retrieval_query": evidence.retrieval_query,
                            "summary_preview": evidence.claim[:220],
                        },
                    )
        return [evidence_by_index[index] for index in range(len(chunks))], warnings

    def _summarize_group(
        self,
        question: str,
        group: list[_MaterialNode],
        evidence_by_id: dict[str, Evidence],
        target_chars: int | None = None,
    ) -> _MaterialNode:
        material = _render_material_nodes(group)
        retrieval_point = _combined_retrieval_point(group)
        instruction = (
            f"压缩以上同一检索点的材料，不超过 {target_chars} 个字符"
            if target_chars
            else "合并以上材料中能回答问题的内容"
        )
        user = f"""{material}

问题：{question}
当前日期：{_today()}
当前检索点：{retrieval_point or '未指定'}
{instruction}，删除重复，保留具体名称、日期、数字和 [S1] 来源编号。只写当前检索点，不写其他对象；问题询问当前状态时，区分已生效、旧版本和前瞻信息，无法确认就保留冲突。只输出合并后的材料。
分支总结："""
        output = self._complete(
            user,
            max_tokens=(
                min(
                    self.settings.rwkv_summary_max_tokens,
                    max(128, target_chars // 2),
                )
                if target_chars
                else self.settings.rwkv_summary_max_tokens
            ),
            stage="reduce",
        )
        summary = _clean_final_answer(output)
        if not summary:
            raise SearchReaderError("reduce", "RWKV returned an empty group summary")
        if retrieval_point:
            summary_key = _normalized_text(summary)
            foreign_points = {
                item.retrieval_point
                for item in evidence_by_id.values()
                if item.retrieval_point
                and item.retrieval_point != retrieval_point
            }
            crossed = [
                point
                for point in foreign_points
                if _normalized_text(point) in summary_key
            ]
            if crossed:
                raise SearchReaderError(
                    "reduce",
                    "RWKV crossed retrieval points: " + ", ".join(sorted(crossed)),
                )
        unsupported = _unsupported_summary_numbers(summary, material)
        if unsupported:
            values = ", ".join(sorted(unsupported))
            raise SearchReaderError(
                "reduce", f"RWKV introduced unsupported numbers: {values}"
            )
        source_ids = tuple(
            dict.fromkeys(
                source_id for node in group for source_id in node.source_ids
            )
        )
        allowed = set(source_ids)
        summary = re.sub(
            r"\[(S\d+)\]",
            lambda match: match.group(0)
            if match.group(1).upper() in allowed
            else "",
            summary,
            flags=re.IGNORECASE,
        )
        source_evidence = [
            evidence_by_id[source_id]
            for source_id in source_ids
            if source_id in evidence_by_id
        ]
        summary = _attach_missing_citations(summary, source_evidence)
        return _MaterialNode(
            source_ids=source_ids,
            summary=summary,
            retrieval_point=_combined_retrieval_point(group),
        )

    def _finalize_branch(
        self,
        question: str,
        node: _MaterialNode,
        evidence_by_id: dict[str, Evidence],
        target_chars: int,
    ) -> _MaterialNode:
        material = _render_material_nodes([node])
        user = f"""{material}

问题：{question}
当前日期：{_today()}
当前检索点：{node.retrieval_point}
形成这个检索点的最终分支总结，不超过 {target_chars} 个字符。问题询问当前状态时，选择当前日期前已上线且生效时间最晚的信息；旧版本与未上线前瞻不作为当前版本，无法确认就说明冲突。只写当前检索点，不写其他对象；[S1] 只是来源编号，不是版本号。保留事实后的来源编号。
分支最终总结："""
        output = self._complete(
            user,
            max_tokens=min(
                self.settings.rwkv_summary_max_tokens,
                max(256, target_chars // 2),
            ),
            stage="branch_finalize",
        )
        summary = _clean_final_answer(output)
        if not summary:
            raise SearchReaderError(
                "branch_finalize", "RWKV returned an empty branch summary"
            )
        unsupported = _unsupported_summary_numbers(summary, material)
        if unsupported:
            values = ", ".join(sorted(unsupported))
            raise SearchReaderError(
                "branch_finalize",
                f"RWKV introduced unsupported numbers: {values}",
            )
        summary_key = _normalized_text(summary)
        crossed = [
            point
            for point in {
                item.retrieval_point
                for item in evidence_by_id.values()
                if item.retrieval_point
                and item.retrieval_point != node.retrieval_point
            }
            if _normalized_text(point) in summary_key
        ]
        if crossed:
            raise SearchReaderError(
                "branch_finalize",
                "RWKV crossed retrieval points: " + ", ".join(sorted(crossed)),
            )
        allowed = set(node.source_ids)
        summary = re.sub(
            r"\[(S\d+)\]",
            lambda match: match.group(0)
            if match.group(1).upper() in allowed
            else "",
            summary,
            flags=re.IGNORECASE,
        )
        summary = _attach_missing_citations(
            summary,
            [
                evidence_by_id[source_id]
                for source_id in node.source_ids
                if source_id in evidence_by_id
            ],
        )
        return _MaterialNode(
            source_ids=node.source_ids,
            summary=summary,
            retrieval_point=node.retrieval_point,
        )

    def _reduce_evidence(
        self,
        question: str,
        evidence: list[Evidence],
        on_progress: Callable[[str, dict[str, object]], None] | None = None,
    ) -> tuple[str, list[str], int]:
        evidence_by_id = {item.source_id: item for item in evidence}
        nodes = [
            _MaterialNode(
                source_ids=(item.source_id,),
                summary=(
                    f"{item.claim}\n发布时间：{item.published_date}"
                    if item.published_date
                    else item.claim
                ),
                retrieval_point=item.retrieval_point,
            )
            for item in evidence
        ]
        warnings: list[str] = []
        level = 0
        final_node_target = self.settings.rwkv_reduce_fan_in + 1
        while True:
            rendered_chars = len(_render_material_nodes(nodes))
            points_collapsed = _retrieval_points_collapsed(nodes)
            point_bound_nodes = bool(nodes) and all(
                node.retrieval_point for node in nodes
            )
            structure_ready = (
                points_collapsed
                if point_bound_nodes
                else len(nodes) <= final_node_target
            )
            if (
                structure_ready
                and rendered_chars <= self.settings.rwkv_final_material_chars
            ):
                break
            level += 1
            compression_target: int | None = None
            if points_collapsed:
                label_overhead = sum(
                    len(node.retrieval_point)
                    + sum(len(source_id) for source_id in node.source_ids)
                    + 80
                    for node in nodes
                )
                compression_target = max(
                    160,
                    (
                        self.settings.rwkv_final_material_chars - label_overhead
                    )
                    // max(1, len(nodes)),
                )
                groups = [[node] for node in nodes]
            else:
                groups = _chunk_material_nodes_by_retrieval_point(
                    nodes,
                    max_chars=self.settings.rwkv_reduce_chunk_chars,
                    fan_in=self.settings.rwkv_reduce_fan_in,
                )
            if on_progress:
                on_progress(
                    "reviewing",
                    {
                        "level": level,
                        "source_count": len(evidence),
                        "node_count": len(nodes),
                        "group_count": len(groups),
                    },
                )
            next_nodes: dict[int, _MaterialNode] = {}
            singleton_groups = {
                index: group[0]
                for index, group in enumerate(groups)
                if len(group) == 1
                and (
                    compression_target is None
                    or len(group[0].summary) <= compression_target
                )
            }
            next_nodes.update(singleton_groups)
            completed = 0
            for node in singleton_groups.values():
                completed += 1
                if on_progress:
                    on_progress(
                        "evidence_group_read",
                        {
                            "level": level,
                            "completed": completed,
                            "total": len(groups),
                            "source_ids": list(node.source_ids),
                            "retrieval_point": node.retrieval_point,
                            "status": "retained",
                            "summary_preview": node.summary[:220],
                        },
                    )
            reducible_groups = [
                (index, group)
                for index, group in enumerate(groups)
                if index not in singleton_groups
            ]
            workers = min(
                self.settings.rwkv_summary_workers, max(1, len(reducible_groups))
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self._summarize_group,
                        question,
                        group,
                        evidence_by_id,
                        compression_target,
                    ): (index, group)
                    for index, group in reducible_groups
                }
                for future in as_completed(futures):
                    index, group = futures[future]
                    source_ids = tuple(
                        dict.fromkeys(
                            source_id
                            for node in group
                            for source_id in node.source_ids
                        )
                    )
                    try:
                        node = future.result()
                        status = "done"
                    except SearchReaderError as exc:
                        per_node = max(
                            240,
                            self.settings.rwkv_reduce_chunk_chars
                            // max(1, len(group))
                            - 80,
                        )
                        node = _MaterialNode(
                            source_ids=source_ids,
                            summary="\n".join(
                                item.summary[:per_node] for item in group
                            ),
                            retrieval_point=_combined_retrieval_point(group),
                        )
                        status = "degraded"
                        warnings.append(
                            f"{'/'.join(source_ids)} funnel reduction degraded: {exc}"[
                                :500
                            ]
                        )
                    next_nodes[index] = node
                    completed += 1
                    if on_progress:
                        on_progress(
                            "evidence_group_read",
                            {
                                "level": level,
                                "completed": completed,
                                "total": len(groups),
                                "source_ids": list(source_ids),
                                "retrieval_point": node.retrieval_point,
                                "status": status,
                                "summary_preview": node.summary[:220],
                            },
                        )
            nodes = [next_nodes[index] for index in range(len(groups))]
            if level >= 6:
                warnings.append("funnel reduction reached the six-level safety limit")
                break
        if _retrieval_points_collapsed(nodes):
            label_overhead = sum(
                len(node.retrieval_point)
                + sum(len(source_id) for source_id in node.source_ids)
                + 80
                for node in nodes
            )
            target_chars = max(
                320,
                (self.settings.rwkv_final_material_chars - label_overhead)
                // max(1, len(nodes)),
            )
            if on_progress:
                on_progress(
                    "branch_finalizing",
                    {
                        "branch_count": len(nodes),
                        "retrieval_points": [
                            node.retrieval_point for node in nodes
                        ],
                    },
                )
            finalized_nodes: dict[int, _MaterialNode] = {}
            workers = min(self.settings.rwkv_summary_workers, len(nodes))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self._finalize_branch,
                        question,
                        node,
                        evidence_by_id,
                        target_chars,
                    ): (index, node)
                    for index, node in enumerate(nodes)
                }
                completed = 0
                for future in as_completed(futures):
                    index, original = futures[future]
                    try:
                        finalized = future.result()
                        status = "done"
                    except SearchReaderError as exc:
                        finalized = original
                        status = "degraded"
                        warnings.append(
                            f"{original.retrieval_point} branch finalization degraded: {exc}"[
                                :500
                            ]
                        )
                    finalized_nodes[index] = finalized
                    completed += 1
                    if on_progress:
                        on_progress(
                            "branch_finalized",
                            {
                                "completed": completed,
                                "total": len(nodes),
                                "retrieval_point": original.retrieval_point,
                                "status": status,
                                "summary_preview": finalized.summary[:220],
                            },
                        )
            nodes = [finalized_nodes[index] for index in range(len(nodes))]
        return _render_material_nodes(nodes), warnings, len(nodes)

    def _aggregate_retrieval_points(
        self,
        question: str,
        material: str,
        retrieval_points: list[str],
    ) -> tuple[str, list[str]]:
        point_list = "、".join(retrieval_points)
        user = f"""{material}

问题：{question}
将以上分支总结整理成一份汇总材料。只保留这些检索点：{point_list}。每个检索点单独成段，保留名称、事实和 [S1] 来源编号；不新增对象，不回答问题，不改变事实归属。只输出汇总材料。
汇总材料："""
        output = self._complete(
            user,
            max_tokens=min(self.settings.rwkv_max_tokens, 2048),
            stage="aggregate",
        )
        aggregated = _clean_final_answer(output)
        warnings: list[str] = []
        if not aggregated:
            return material, ["retrieval-point aggregation returned empty material"]
        unsupported = _unsupported_summary_numbers(aggregated, material)
        if unsupported:
            values = ", ".join(sorted(unsupported))
            return material, [
                f"retrieval-point aggregation introduced unsupported numbers: {values}"
            ]
        missing_points = [
            point
            for point in retrieval_points
            if _normalized_text(point) not in _normalized_text(aggregated)
        ]
        if missing_points:
            warnings.append(
                "retrieval-point aggregation omitted branches: "
                + ", ".join(missing_points)
            )
            return material, warnings
        headings = re.findall(r"(?m)^#{1,6}\s+(.+?)\s*$", aggregated)
        unknown_headings = [
            heading
            for heading in headings
            if not any(
                _normalized_text(point) in _normalized_text(heading)
                for point in retrieval_points
            )
        ]
        if unknown_headings:
            warnings.append(
                "retrieval-point aggregation added branches: "
                + ", ".join(unknown_headings)
            )
            return material, warnings
        source_points: dict[str, str] = {}
        for line in material.splitlines():
            try:
                source_ids, summary = json.loads(line)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            point_match = re.search(r"检索点：([^\n]+)", str(summary))
            if point_match:
                for source_id in source_ids:
                    source_points[str(source_id).upper()] = point_match.group(1)
        heading_matches = list(
            re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", aggregated)
        )
        for index, heading_match in enumerate(heading_matches):
            heading = heading_match.group(1)
            point = next(
                (
                    item
                    for item in retrieval_points
                    if _normalized_text(item) in _normalized_text(heading)
                ),
                "",
            )
            if not point:
                continue
            end = (
                heading_matches[index + 1].start()
                if index + 1 < len(heading_matches)
                else len(aggregated)
            )
            section = aggregated[heading_match.end() : end]
            crossed = [
                source_id
                for source_id in re.findall(
                    r"\[(S\d+)\]", section, flags=re.IGNORECASE
                )
                if source_points.get(source_id.upper(), point) != point
            ]
            if crossed:
                warnings.append(
                    f"retrieval-point aggregation crossed {point} citations: "
                    + ", ".join(dict.fromkeys(crossed))
                )
                return material, warnings
        allowed_ids = set(re.findall(r"S\d+", material, flags=re.IGNORECASE))
        aggregated = re.sub(
            r"\[(S\d+)\]",
            lambda match: match.group(0)
            if match.group(1).upper() in allowed_ids
            else "",
            aggregated,
            flags=re.IGNORECASE,
        )
        return aggregated, warnings

    def finalize(
        self,
        question: str,
        evidence: list[Evidence],
        on_progress: Callable[[str, dict[str, object]], None] | None = None,
        conversation_history: list[dict[str, object]] | None = None,
        current_message: str = "",
        research_records: list[ResearchRecord] | None = None,
        evidence_sufficient: bool | None = None,
        evidence_selections: list[FieldEvidenceSelection] | None = None,
        representative_scope: bool = False,
    ) -> tuple[str, list[str]]:
        if not evidence:
            raise SearchReaderError("read", "there are no source summaries to combine")
        eligible = _eligible_evidence(question, evidence)
        ordered = _final_evidence_order(question, eligible)
        use_atomic_records = bool(
            evidence_selections
            and _field_binding_applicable(question, research_records)
        )
        if use_atomic_records:
            selected_rows = list(evidence_selections or [])
            reduced_material, material_truncated = _bounded_field_material(
                selected_rows, self.settings.rwkv_final_material_chars
            )
            reduction_warnings: list[str] = (
                ["atomic field material was compacted to fit the final context"]
                if material_truncated
                else []
            )
            group_count = len(
                {selection.subject for selection in selected_rows}
            )
        else:
            reduced_material, reduction_warnings, group_count = self._reduce_evidence(
                question, ordered, on_progress=on_progress
            )
            retrieval_points = list(
                dict.fromkeys(
                    item.retrieval_point
                    for item in ordered
                    if item.retrieval_point
                )
            )
            if len(retrieval_points) > 1:
                if on_progress:
                    on_progress(
                        "aggregating",
                        {
                            "retrieval_points": retrieval_points,
                            "branch_count": len(retrieval_points),
                        },
                    )
                reduced_material, aggregate_warnings = (
                    self._aggregate_retrieval_points(
                        question,
                        reduced_material,
                        retrieval_points,
                    )
                )
                reduction_warnings.extend(aggregate_warnings)
                if on_progress:
                    on_progress(
                        "aggregate_completed",
                        {
                            "retrieval_points": retrieval_points,
                            "branch_count": len(retrieval_points),
                            "status": "degraded" if aggregate_warnings else "done",
                        },
                    )
        del (
            conversation_history,
            current_message,
            evidence_sufficient,
        )
        scope_note = (
            "回答当轮材料能够确认的代表对象，不要声称已经穷举全部。"
            if representative_scope
            else ""
        )
        material_request = (
            "根据以上字段记录直接回答问题，每个字段只使用自己的值"
            if use_atomic_records
            else "根据以上材料直接回答问题"
        )
        final_point_scope = (
            "只回答这些检索点："
            + "、".join(
                dict.fromkeys(
                    item.retrieval_point
                    for item in ordered
                    if item.retrieval_point
                )
            )
            + "；不要增加其他对象。"
            if not use_atomic_records
            and any(item.retrieval_point for item in ordered)
            else ""
        )
        user = f"""{reduced_material}

问题：{question}
{material_request}，只回答所问内容，合并重复内容，不推断材料没有的信息；不同检索点的对象和事实不得混用；材料不足就说明。事实后只用 [S1] 格式标注来源。{final_point_scope}{scope_note}
最终回答："""
        if on_progress:
            on_progress(
                "finalizing",
                {"source_count": len(ordered), "group_count": group_count},
            )
        requested_tokens = self.settings.rwkv_max_tokens
        output_tokens = requested_tokens
        for attempt in range(4):
            try:
                output = self._complete(
                    user,
                    max_tokens=output_tokens,
                    stage="finalize",
                )
                break
            except SearchReaderError as exc:
                retry_tokens = _context_retry_tokens(str(exc), output_tokens)
                if retry_tokens is None or attempt == 3:
                    raise
                output_tokens = retry_tokens
        if output_tokens != requested_tokens:
            reduction_warnings.append(
                f"final output budget reduced to {output_tokens} tokens to fit context"
            )
        answer = _clean_final_answer(output)
        if not answer:
            raise SearchReaderError("read", "RWKV returned an empty final answer")
        if use_atomic_records:
            answer, atomic_warnings = _filter_unsupported_atomic_clauses(
                answer, list(evidence_selections or [])
            )
            reduction_warnings.extend(atomic_warnings)
            if not answer:
                answer = _atomic_fallback_answer(list(evidence_selections or []))
                reduction_warnings.append(
                    "final atomic answer used the deterministic safety renderer"
                )
            selected_ids = {
                source_id
                for selection in evidence_selections or []
                if selection.status == "supported"
                for source_id in _selection_source_ids(selection)
            }
            selected_evidence = [
                item for item in ordered if item.source_id in selected_ids
            ]
            answer = _attach_missing_citations(answer, selected_evidence)
            answer = _bind_atomic_citations(
                answer, list(evidence_selections or [])
            )
            if selected_ids and not re.search(r"\[S\d+\]", answer, flags=re.IGNORECASE):
                answer = _atomic_fallback_answer(list(evidence_selections or []))
                reduction_warnings.append(
                    "final atomic answer had no supportable clause; used safety renderer"
                )
        else:
            answer = _attach_missing_citations(answer, ordered)
        return answer, reduction_warnings


def citation_warnings(answer: str, evidence: list[Evidence]) -> list[str]:
    allowed = {item.source_id for item in evidence}
    referenced = {
        item.upper()
        for item in re.findall(r"\[(S\d+)\]", str(answer or ""), flags=re.IGNORECASE)
    }
    invalid = sorted(referenced - allowed)
    warnings: list[str] = []
    if invalid:
        warnings.append(f"RWKV referenced unknown source ids: {', '.join(invalid)}")
    if allowed and not referenced:
        warnings.append("RWKV answer contains no [S1]-style source citations")
    return warnings
