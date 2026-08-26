from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
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
_CONTEXT_REFERENCE_RE = re.compile(
    r"它|它们|他们|她们|这个|那个|这些|那些|上述|前者|后者|"
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
_FINAL_MODEL_STAGES = {"conversation", "document", "finalize", "memory"}
_SCOPE_ACCEPTANCE_RE = re.compile(
    r"^(?:都需要|都要|全部|全都要|所有|这些都要|就这些|都可以|都行)[。.!！?？]*$"
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
    text = re.sub(r"^\s*(?:summary|摘要)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
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


def _normalized_numbers(value: str) -> set[str]:
    value = re.sub(r"(?m)^\s*(?:#{1,6}\s*)?\d{1,3}[.)、]\s+", "", value)
    normalized: set[str] = set()
    for raw in _NUMBER_RE.findall(value):
        try:
            number = Decimal(raw.replace(",", "")).normalize()
        except InvalidOperation:
            if raw.count(".") > 1:
                for part in raw.split("."):
                    if part.isdigit():
                        normalized.add(str(Decimal(part).normalize()))
            continue
        normalized.add(str(number))
    return normalized


def _unsupported_summary_numbers(summary: str, material: str) -> set[str]:
    return _normalized_numbers(summary) - _normalized_numbers(material)


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
        if item.status != "supported" or not item.source_id:
            continue
        sources_by_subject.setdefault(item.subject, [])
        if item.source_id not in sources_by_subject[item.subject]:
            sources_by_subject[item.subject].append(item.source_id)
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


@dataclass(slots=True)
class _MaterialNode:
    source_ids: tuple[str, ...]
    summary: str


def _render_material_nodes(nodes: list[_MaterialNode]) -> str:
    """Render each reduced summary as an inseparable lineage-summary pair."""

    return "\n".join(
        json.dumps([list(node.source_ids), node.summary], ensure_ascii=False)
        for node in nodes
    )


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

    if _recency_requested(question):
        return sorted(
            evidence,
            key=lambda item: (
                _published_timestamp(item.published_date) is not None,
                _published_timestamp(item.published_date) or 0.0,
                _source_relevance(question, item),
                item.score,
            ),
        )
    return sorted(
        evidence,
        key=lambda item: (_source_relevance(question, item), item.score),
    )


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
    text = re.sub(r"^\s*(?:最终答案|回答)\s*[:：]\s*", "", text)
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
    if scope_mode == "discover":
        queries = queries[:2]
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

    def reset_call_records(self) -> None:
        with self._call_lock:
            self._call_sequence = 0
            self._call_records = []

    def call_records(self) -> list[dict[str, object]]:
        with self._call_lock:
            return [dict(record) for record in self._call_records]

    def _next_call_id(self) -> int:
        with self._call_lock:
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

把问题转换为检索计划。decision 只能是 search 或 clarify；公开信息用 search，只有指代无法确定时用 clarify。scope_mode 只能是 fixed 或 discover；明确对象用 fixed，未点名的公开类别用 discover。保留问题中的字段，最多生成 {self.settings.research_plan_max_queries} 个检索词。
只输出 JSON：
{{"decision":"","scope_mode":"","goal":"","records":[{{"subject":"","question":"","fields":[]}}],"queries":[],"clarification":""}}"""
        output = self._complete(user, max_tokens=768, stage="plan")
        plan = _parse_research_plan_output(
            output,
            question,
            self.settings.research_plan_max_queries,
        )
        self._anchor_research_queries(question, plan)
        return plan

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
                    query = f"{query} 更新公告 活动"
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
{{"subjects":[]}}"""
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
{{"decision":"search","scope_mode":"fixed","goal":"","records":[{{"subject":"","question":"","fields":[]}}],"queries":[],"clarification":""}}"""
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

    def contextualize(self, question: str, history: list[dict[str, str]]) -> str:
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
        material = "\n".join(
            json.dumps(
                {"user": turn.get("question", ""), "assistant": turn.get("answer", "")},
                ensure_ascii=False,
            )
            for turn in history
        )
        user = f"""{material}

当前消息：{question}
结合此前对话，把当前消息改写成可独立理解的问题。只输出改写后的问题。"""
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
            previous_question = str(history[-1].get("question") or "").strip()
            if previous_question:
                previous_question = previous_question.rstrip("。.!！?？")
                return f"上一问：{previous_question}；当前追问：{question}"[:8000]
        return rewritten

    def answer_from_history(self, question: str, history: list[dict[str, str]]) -> str:
        """Answer a request about this conversation without external retrieval."""

        material = "\n".join(
            json.dumps(
                {
                    "turn": index,
                    "user": turn.get("question", "")[:1200],
                    "assistant": turn.get("answer", "")[:1800],
                },
                ensure_ascii=False,
            )
            for index, turn in enumerate(history, start=1)
        )
        user = f"""{material}

请求：{question}
根据此前对话完成请求。合并重复内容，不复制来源编号。直接回答。"""
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
    ) -> EvidenceAudit:
        """Ask RWKV whether the accumulated material can answer the question."""

        material = _render_source_pairs(evidence)
        tried = "\n".join(f"- {query}" for query in tried_queries)
        records = _render_research_records(research_records)
        user = f"""{material}

记录：{records}
问题：{question}
已检索：{tried}
当前日期：{_today()}

判断材料是否足以覆盖问题和记录中的字段。足够只输出 ENOUGH；不足输出最多 {self.settings.review_max_queries} 行新检索词，每行格式为 SEARCH: 检索词。"""
        output = self._complete(user, max_tokens=192, stage="audit")
        return _parse_audit_output(output, self.settings.review_max_queries)

    def _select_one_field(
        self,
        question: str,
        record: ResearchRecord,
        field: str,
        evidence: list[Evidence],
    ) -> FieldEvidenceSelection:
        material = _render_source_pairs(evidence)
        user = f"""{material}

对象：{record.subject}
字段：{field}
当前日期：{_today()}
选择一个直接支持该字段的来源；当前信息选择日期最新的来源。没有材料用 missing，冲突用 conflict。
只输出 JSON：
{{"source_id":"","status":"","fact":""}}"""
        output = self._complete(user, max_tokens=320, stage="evidence_select")
        payload = _json_object_from_output(output, "evidence_select")
        status = str(payload.get("status") or "").strip().casefold()
        if status not in {"supported", "missing", "conflict"}:
            raise SearchReaderError(
                "evidence_select", "RWKV returned an unknown evidence status"
            )
        source_id = str(payload.get("source_id") or "").strip().upper()
        allowed = {item.source_id: item for item in evidence}
        if status == "supported" and source_id not in allowed:
            raise SearchReaderError(
                "evidence_select", "RWKV selected an unknown evidence source"
            )
        if status != "supported":
            source_id = ""
        fact = _narrow_selected_fact(field, str(payload.get("fact") or ""))[:500]
        if status == "supported":
            source = allowed[source_id]
            if _normalized_text(record.subject) not in _normalized_text(
                f"{source.title} {source.quote}"
            ):
                raise SearchReaderError(
                    "evidence_select",
                    "selected source does not explicitly name the research subject",
                )
            if re.search(r"福利|奖励|补偿|赠送|活动", field) and not (
                _is_explicit_benefit_fact(fact)
            ):
                raise SearchReaderError(
                    "evidence_select",
                    "RWKV selected content without an explicit player benefit",
                )
            if re.search(r"时间|日期|上线|发布", field) and not re.search(
                r"(?:更新|维护|上线|发布)(?:的)?(?:时间|日期)\s*(?:为|是|[:：])?\s*\d{4}"
                r"|\d{4}[^。；]{0,30}(?:更新|维护|上线|发布)",
                f"{source.title} {source.claim}",
            ):
                return FieldEvidenceSelection(
                    subject=record.subject,
                    field=field,
                    source_id="",
                    status="missing",
                    fact="来源未明确写出版本更新发生的确切日期。",
                )
            unsupported = _unsupported_summary_numbers(
                fact,
                f"{question} {source.title} {source.published_date} {source.claim}",
            )
            if unsupported:
                values = ", ".join(sorted(unsupported))
                raise SearchReaderError(
                    "evidence_select",
                    f"RWKV selected fact introduced unsupported numbers: {values}",
                )
        return FieldEvidenceSelection(
            subject=record.subject,
            field=field,
            source_id=source_id,
            status=status,
            fact=fact,
        )

    def select_evidence(
        self,
        question: str,
        records: list[ResearchRecord],
        evidence: list[Evidence],
        on_progress: Callable[[str, dict[str, object]], None] | None = None,
    ) -> tuple[list[FieldEvidenceSelection], list[str]]:
        """Select one atomic source per answer field, in parallel."""

        useful = [item for item in evidence if not _is_irrelevant_summary(item.claim)]
        useful = useful or evidence
        tasks: list[tuple[ResearchRecord, str, list[Evidence]]] = []
        for record in records:
            subject_key = _normalized_text(record.subject)
            direct = [
                item
                for item in useful
                if subject_key
                and subject_key in _normalized_text(item.title)
            ]
            contextual = [
                item
                for item in useful
                if subject_key
                and subject_key in _normalized_text(f"{item.title} {item.quote}")
            ]
            candidates = sorted(
                direct or contextual,
                key=lambda item: _source_relevance(record.question, item),
                reverse=True,
            )[:8]
            for field in record.fields or [record.question]:
                tasks.append((record, field, candidates))
        if not tasks:
            return [], []

        selections: dict[int, FieldEvidenceSelection] = {}
        warnings: list[str] = []
        completed = 0
        workers = min(self.settings.rwkv_summary_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._select_one_field,
                    question,
                    record,
                    field,
                    candidates,
                ): (index, record, field)
                for index, (record, field, candidates) in enumerate(tasks)
            }
            for future in as_completed(futures):
                index, record, field = futures[future]
                try:
                    selection = future.result()
                except SearchReaderError as exc:
                    try:
                        selection = self._select_one_field(
                            question,
                            record,
                            field,
                            tasks[index][2],
                        )
                    except SearchReaderError as retry_exc:
                        selection = FieldEvidenceSelection(
                            subject=record.subject,
                            field=field,
                            source_id="",
                            status="missing",
                            fact="证据选择失败，不能安全绑定来源。",
                        )
                        warnings.append(
                            f"{record.subject}/{field} evidence selection degraded: "
                            f"{exc}; retry: {retry_exc}"[:500]
                        )
                selections[index] = selection
                completed += 1
                if on_progress:
                    on_progress(
                        "evidence_field_selected",
                        {
                            "completed": completed,
                            "total": len(tasks),
                            "subject": selection.subject,
                            "field": selection.field,
                            "source_id": selection.source_id,
                            "status": selection.status,
                            "fact_preview": selection.fact[:180],
                        },
                    )
        return [selections[index] for index in range(len(tasks))], warnings

    def _summarize_one(self, _question: str, chunk: DocumentChunk) -> Evidence:
        material = chunk.text[: self.settings.max_evidence_chars].strip()
        user = f"""{material}

总结以上网页内容。"""
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
                    material = chunk.text[: min(self.settings.max_evidence_chars, 1200)]
                    evidence = Evidence(
                        source_id=chunk.source_id,
                        title=chunk.title,
                        url=chunk.url,
                        claim=f"RWKV 摘要失败；原始材料摘录：{material}",
                        quote=material,
                        score=chunk.score,
                        source_relation="unclear",
                        published_date=chunk.published_date,
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
                            "summary_preview": evidence.claim[:220],
                        },
                    )
        return [evidence_by_index[index] for index in range(len(chunks))], warnings

    def _summarize_group(
        self,
        question: str,
        group: list[_MaterialNode],
        evidence_by_id: dict[str, Evidence],
    ) -> _MaterialNode:
        material = _render_material_nodes(group)
        user = f"""{material}

问题：{question}
合并以上材料中能回答问题的内容，删除重复，保留具体名称、日期、数字和 [S1] 来源编号。只输出合并后的材料。"""
        output = self._complete(
            user,
            max_tokens=self.settings.rwkv_summary_max_tokens,
            stage="reduce",
        )
        summary = _clean_final_answer(output)
        if not summary:
            raise SearchReaderError("reduce", "RWKV returned an empty group summary")
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
        return _MaterialNode(source_ids=source_ids, summary=summary)

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
            )
            for item in evidence
        ]
        warnings: list[str] = []
        level = 0
        final_node_target = self.settings.rwkv_reduce_fan_in + 1
        while (
            len(nodes) > final_node_target
            or len(_render_material_nodes(nodes))
            > self.settings.rwkv_final_material_chars
        ):
            level += 1
            groups = _chunk_material_nodes(
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
            completed = 0
            workers = min(self.settings.rwkv_summary_workers, len(groups))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self._summarize_group,
                        question,
                        group,
                        evidence_by_id,
                    ): (index, group)
                    for index, group in enumerate(groups)
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
                                "status": status,
                                "summary_preview": node.summary[:220],
                            },
                        )
            nodes = [next_nodes[index] for index in range(len(groups))]
            if level >= 6:
                warnings.append("funnel reduction reached the six-level safety limit")
                break
        return _render_material_nodes(nodes), warnings, len(nodes)

    def finalize(
        self,
        question: str,
        evidence: list[Evidence],
        on_progress: Callable[[str, dict[str, object]], None] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
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
        reduced_material, reduction_warnings, group_count = self._reduce_evidence(
            question, ordered, on_progress=on_progress
        )
        del (
            conversation_history,
            current_message,
            research_records,
            evidence_sufficient,
            evidence_selections,
        )
        scope_note = (
            "回答当轮材料能够确认的代表对象，不要声称已经穷举全部。"
            if representative_scope
            else ""
        )
        user = f"""{reduced_material}

问题：{question}
根据以上材料直接回答问题，只回答所问内容，合并重复内容，不推断材料没有的信息；材料不足就说明。事实后只用 [S1] 格式标注来源。{scope_note}"""
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
        return _attach_missing_citations(answer, ordered), reduction_warnings


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
