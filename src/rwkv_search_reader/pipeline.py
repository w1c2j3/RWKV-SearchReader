from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from typing import Any
from urllib.parse import urlsplit

from .config import Settings, get_settings
from .documents import MarkdownPassage
from .models import (
    DocumentChunk,
    Evidence,
    EvidenceAudit,
    FieldEvidenceSelection,
    PipelineResult,
    ResearchPlan,
    ResearchRecord,
    SearchHit,
    SearchReaderError,
)
from .reader import (
    RWKVReader,
    _freshness_review_queries,
    _missing_field_queries,
    _numeric_route_label,
    citation_warnings,
)
from .retrieval import Retriever
from .search import create_search_provider
from .tracing import append_trace, session_turns

_HISTORY_REFERENCE_RE = re.compile(
    r"刚才|前面|上面|之前|此前|前文|对话|聊天|我们(?:聊|说)|"
    r"你(?:刚才|之前|前面)?(?:说|提到)|我(?:刚才|之前|前面)?(?:说|提到)|"
    r"你现在知道|你已经知道|已经聊过|讨论过"
)
_HISTORY_SUMMARY_RE = re.compile(
    r"总结|概括|归纳|整理|回顾|复述|要点|结论|"
    r"(?:说|聊|讨论)了什么|知道(?:了)?(?:哪些|什么)|记得.*(?:什么|吗)"
)
_DOCUMENT_REFERENCE_RE = re.compile(
    r"文档|文件|附件|markdown|\.md\b|这篇|这份材料|上面的材料",
    flags=re.IGNORECASE,
)
_EXPLICIT_WEB_RE = re.compile(
    r"请(?:再|同时|帮我)?(?:联网|搜索|检索)|帮我(?:联网|搜索|检索)|"
    r"(?:联网|网上|网络|网页)(?:查|搜|检索|搜索|补充)|"
    r"(?:搜索|检索|查找)(?:一下|网络|网页|外部)|结合(?:网络|网页|外部资料)|"
    r"最新|今天|本周|近期"
)
_CONFIRMATION_RE = re.compile(r"官方|权威|确认|核实|验证|公告|出处|来源")
_DATE_FACT_RE = re.compile(
    r"(?<!\d)(\d{4})(?:年|[-/.])(\d{1,2})(?:月|[-/.])(\d{1,2})(?:日)?(?!\d)"
)
_QUANTITY_FACT_RE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)(%|％|亿元|万元|万股|万|亿|元/股|元|美元|"
    r"人民币|人|家|项|个)(?![\w])"
)
_GAP_NOISE_PHRASES = (
    "正式发布日期",
    "发布日期",
    "发布时间",
    "官方确认",
    "权威确认",
    "再次验证",
    "更多来源",
    "什么时候",
    "什么时间",
    "是多少",
    "有多少",
    "为什么",
    "怎么样",
    "官方",
    "权威",
    "确认",
    "核实",
    "验证",
    "公告",
    "出处",
    "来源",
    "日期",
    "时间",
    "最新",
    "当前",
)
_QUERY_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_RELEASE_TITLE_RE = re.compile(
    r"(?<!\d)(\d+(?:\.\d+){1,3})\s*(?:全新|新)?\s*版本",
    re.IGNORECASE,
)
_OFFICIAL_TITLE_RE = re.compile(
    r"官网|官方网站|官方公告|official\s+(?:site|website)", re.IGNORECASE
)


def _history_only_request(question: str) -> bool:
    """Conservatively recognize requests that operate on this chat itself."""

    text = " ".join(str(question or "").split()).strip()
    return bool(
        text and _HISTORY_REFERENCE_RE.search(text) and _HISTORY_SUMMARY_RE.search(text)
    )


def _fact_atoms(text: str) -> set[str]:
    """Return stable date and quantity atoms used by the retrieval stop guard."""

    compact = re.sub(r"\s+", "", str(text or ""))
    atoms = {
        f"date:{int(year)}-{int(month)}-{int(day)}"
        for year, month, day in _DATE_FACT_RE.findall(compact)
    }
    atoms.update(
        f"quantity:{number}:{unit.replace('％', '%')}"
        for number, unit in _QUANTITY_FACT_RE.findall(compact)
    )
    return atoms


def _rechecks_supported_fact(
    query: str,
    question: str,
    evidence: list[object],
) -> bool:
    """Reject answer-seeded confirmation searches after independent agreement."""

    if _CONFIRMATION_RE.search(question):
        return False
    counts: Counter[str] = Counter()
    for item in evidence:
        counts.update(_fact_atoms(str(getattr(item, "claim", ""))))
    supported = {atom for atom, count in counts.items() if count >= 2}
    new_query_facts = _fact_atoms(query) - _fact_atoms(question)
    if not (supported & new_query_facts):
        return False
    if _CONFIRMATION_RE.search(query):
        return True

    reduced = str(query or "").casefold()
    reduced = _DATE_FACT_RE.sub(" ", reduced)
    reduced = _QUANTITY_FACT_RE.sub(" ", reduced)
    for phrase in _GAP_NOISE_PHRASES:
        reduced = reduced.replace(phrase, " ")
    terms = set(re.findall(r"[a-z][a-z0-9_.+-]{1,}", reduced))
    for segment in re.findall(r"[\u3400-\u9fff]{2,}", reduced):
        terms.add(segment)
        terms.update(segment[index : index + 2] for index in range(len(segment) - 1))
    evidence_text = "\n".join(
        str(getattr(item, "claim", "")).casefold() for item in evidence
    )
    return bool(terms) and all(term in evidence_text for term in terms)


def _sanitize_supplemental_query(query: str, question: str) -> str:
    """Remove answer-like numbers leaked from evidence and anchor recency safely."""

    chinese = bool(re.search(r"[\u3400-\u9fff]", question))
    replacements = {
        "date": "日期" if chinese else "date",
        "version": "版本" if chinese else "version",
        "event": "活动" if chinese else "event",
        "update_time": "更新时间" if chinese else "update time",
        "release_date": "发布日期" if chinese else "release date",
        "version_number": "版本号" if chinese else "version number",
        "version_name": "版本名称" if chinese else "version name",
        "theme": ("版本主题" if "版本" in question else "主题") if chinese else "theme",
        "station": "站点" if chinese else "station",
        "stations": "站点" if chinese else "stations",
    }
    query = re.sub(
        r"(?<![A-Za-z0-9_])(?:update_time|release_date|version_number|version_name|theme|date|version|event|stations?)(?![A-Za-z0-9_])",
        lambda match: replacements[match.group(0).casefold()],
        str(query or ""),
        flags=re.IGNORECASE,
    )
    question_numbers = {
        match.group(0).replace(",", "")
        for match in _QUERY_NUMBER_RE.finditer(
            _numeric_route_label(str(question or ""))
        )
    }

    def keep_question_number(match: re.Match[str]) -> str:
        value = match.group(0)
        return value if value.replace(",", "") in question_numbers else " "

    sanitized = _QUERY_NUMBER_RE.sub(keep_question_number, query)
    sanitized = re.sub(r"(?:^|\s)[年月日](?=\s|$)", " ", sanitized)
    sanitized = " ".join(sanitized.split()).strip(" -，,；;：:")
    sanitized = re.sub(
        r"(更新时间|发布日期|版本主题|主题)(?:\s+\1)+",
        r"\1",
        sanitized,
    )
    if sanitized and re.search(
        r"今日|今天|当前|现在|最新|本周|这周|近期|最近|本月|今年|下一个|下一版|后续版本",
        question,
    ):
        current = time.localtime()
        anchor = f"{current.tm_year}年{current.tm_mon}月"
        if anchor not in sanitized:
            sanitized = f"{sanitized} {anchor} 最新"
    return sanitized[:800]


def _fallback_audit_query(question: str, tried_queries: list[str]) -> str:
    """Keep a failed audit from silently ending review when no fields were planned."""

    suffixes = ["官方资料", "官方文档 示例"]
    if re.search(r"最近|本周|近期|当前|现在|最新|今天|下一个|下一版|后续版本", question):
        current = time.localtime()
        suffixes.insert(0, f"{current.tm_year}年{current.tm_mon}月 最新 官方公告")
    if re.search(r"怎样|怎么|如何|配置|示例|教程|代码", question):
        suffixes.insert(0, "官方文档 配置示例")
    tried = {" ".join(str(item).split()).casefold() for item in tried_queries}
    for suffix in dict.fromkeys(suffixes):
        candidate = " ".join(f"{question} {suffix}".split())[:800]
        if candidate and candidate.casefold() not in tried:
            return candidate
    return ""


def _mask_unanchored_next_release_fields(
    question: str,
    selections: list[FieldEvidenceSelection],
) -> tuple[list[FieldEvidenceSelection], bool]:
    """Do not persist a future-release detail without a verified future date."""

    if not re.search(r"下一个|下一版|后续版本", question):
        return selections, False
    date_field = re.compile(
        r"日期|时间|上线|发布|update_time|release_date|\bdate\b|\btime\b",
        re.IGNORECASE,
    )
    by_subject: dict[str, list[FieldEvidenceSelection]] = {}
    for item in selections:
        by_subject.setdefault(item.subject, []).append(item)
    unanchored: set[str] = set()
    for subject, items in by_subject.items():
        date_items = [item for item in items if date_field.search(item.field)]
        if not date_items:
            continue
        future = False
        for item in date_items:
            if item.status != "supported":
                continue
            for year, month, day in _DATE_FACT_RE.findall(item.fact):
                try:
                    future |= date(int(year), int(month), int(day)) > date.today()
                except ValueError:
                    continue
        if not future:
            unanchored.add(subject)
    if not unanchored:
        return selections, False
    masked = [
        FieldEvidenceSelection(
            subject=item.subject,
            field=item.field,
            source_id="",
            status="missing",
            fact="未确认下一个版本的未来生效日期，不能沿用旧版本字段。",
        )
        if item.subject in unanchored and item.status == "supported"
        else item
        for item in selections
    ]
    return masked, masked != selections


def _title_release_version(title: str) -> tuple[int, ...] | None:
    match = _RELEASE_TITLE_RE.search(title)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _current_release_scope(
    question: str, sources: list[SearchHit]
) -> tuple[tuple[int, ...] | None, set[str]]:
    """Scope a named current-version answer to the version already online."""

    if (
        len(re.findall(r"《[^》]{2,120}》", question)) != 1
        or not re.search(r"当前|现在|最新", question)
        or not re.search(r"版本", question)
        or re.search(r"下一个|下一版|后续版本", question)
    ):
        return None, set()
    online_sources = [
        source
        for source in sources
        if re.search(r"现已上线|已上线|当前版本|最新版本", source.title)
        and _title_release_version(source.title) is not None
    ]
    if not online_sources:
        return None, set()
    official_online_sources = [
        source for source in online_sources if _OFFICIAL_TITLE_RE.search(source.title)
    ]
    version = max(
        version
        for source in (official_online_sources or online_sources)
        if (version := _title_release_version(source.title)) is not None
    )
    official_release_sources = [
        source
        for source in sources
        if _title_release_version(source.title) == version
        and _OFFICIAL_TITLE_RE.search(source.title)
    ]
    selected_sources = official_release_sources or [
        source for source in sources if _title_release_version(source.title) == version
    ]
    selected = {
        source.source_id
        for source in selected_sources
    }
    return version, selected


def _record_current_release_scopes(
    question: str,
    sources: list[SearchHit],
    records: list[ResearchRecord],
) -> dict[str, tuple[tuple[int, ...], set[str]]]:
    """Find each named object's latest release that an official page says is online."""

    if (
        len(records) < 2
        or not re.search(r"当前|现在|最新", question)
        or not re.search(r"版本", question)
        or re.search(r"下一个|下一版|后续版本", question)
    ):
        return {}
    subjects = [record.subject.strip("《》 ") for record in records]

    def source_subjects(source: SearchHit) -> set[str]:
        title_matches = {subject for subject in subjects if subject in source.title}
        if title_matches:
            return title_matches
        branch = f"{source.retrieval_point} {source.retrieval_query}"
        return {subject for subject in subjects if subject in branch}

    scoped: dict[str, tuple[tuple[int, ...], set[str]]] = {}
    for subject in subjects:
        related = [source for source in sources if subject in source_subjects(source)]
        official_hosts = {
            host
            for source in related
            if _OFFICIAL_TITLE_RE.search(source.title)
            if (host := urlsplit(source.url).hostname)
        }

        def official(source: SearchHit) -> bool:
            return bool(
                _OFFICIAL_TITLE_RE.search(source.title)
                or (
                    bool(urlsplit(source.url).hostname)
                    and urlsplit(source.url).hostname in official_hosts
                )
            )

        online = [
            source
            for source in related
            if official(source)
            and _title_release_version(source.title) is not None
            and re.search(r"上线|当前版本|最新版本", source.title)
            and not re.search(r"即将|将于|预下载|前瞻|预览", source.title)
        ]
        if not online:
            continue
        version = max(
            version
            for source in online
            if (version := _title_release_version(source.title)) is not None
        )
        selected = {
            source.source_id
            for source in related
            if official(source) and _title_release_version(source.title) == version
        }
        scoped[subject] = (version, selected)
    return scoped


def _record_current_release_ids(
    sources: list[SearchHit],
    records: list[ResearchRecord],
    scopes: dict[str, tuple[tuple[int, ...], set[str]]],
) -> set[str]:
    subjects = [record.subject.strip("《》 ") for record in records]
    allowed: set[str] = set()
    for source in sources:
        title_matches = {subject for subject in subjects if subject in source.title}
        branch = f"{source.retrieval_point} {source.retrieval_query}"
        matches = title_matches or {subject for subject in subjects if subject in branch}
        if any(subject not in scopes for subject in matches) or any(
            source.source_id in scopes[subject][1] for subject in matches if subject in scopes
        ):
            allowed.add(source.source_id)
    return allowed


def _require_online_scope_for_current_records(
    question: str,
    records: list[ResearchRecord],
    scopes: dict[str, tuple[tuple[int, ...], set[str]]],
    selections: list[FieldEvidenceSelection],
    *,
    representative_scope: bool,
) -> list[FieldEvidenceSelection]:
    if (
        not representative_scope
        or len(records) < 2
        or not re.search(r"当前|现在|最新", question)
        or not re.search(r"版本", question)
        or re.search(r"下一个|下一版|后续版本", question)
    ):
        return selections
    return [
        replace(
            item,
            source_id="",
            source_ids=[],
            status="missing",
            fact="材料未确认该对象已有哪个版本上线。",
        )
        if item.subject.strip("《》 ") not in scopes
        and re.search(r"版本|主题", item.field)
        else item
        for item in selections
    ]


def _newer_release_ambiguities(
    sources: list[SearchHit],
    records: list[ResearchRecord],
    scopes: dict[str, tuple[tuple[int, ...], set[str]]],
) -> dict[str, list[str]]:
    """Find higher-version pages on the host of a confirmed online release."""

    subjects = [record.subject.strip("《》 ") for record in records]
    result: dict[str, list[str]] = {}
    for subject, (online_version, online_ids) in scopes.items():
        publisher_hosts = {
            host
            for source in sources
            if source.source_id in online_ids
            if (host := urlsplit(source.url).hostname)
        }
        newer: list[str] = []
        for source in sources:
            title_matches = {name for name in subjects if name in source.title}
            branch = f"{source.retrieval_point} {source.retrieval_query}"
            matches = title_matches or {name for name in subjects if name in branch}
            version = _title_release_version(source.title)
            if (
                subject in matches
                and version is not None
                and version > online_version
                and urlsplit(source.url).hostname in publisher_hosts
            ):
                newer.append(source.source_id)
        if newer:
            result[subject] = list(dict.fromkeys(newer))
    return result


def _flag_unresolved_newer_releases(
    selections: list[FieldEvidenceSelection],
    ambiguities: dict[str, list[str]],
) -> list[FieldEvidenceSelection]:
    return [
        replace(
            item,
            source_id="",
            source_ids=list(dict.fromkeys(
                [*item.source_ids, *([item.source_id] if item.source_id else []),
                 *ambiguities[subject]]
            )),
            status="conflict",
            fact="同一发布方站点还有更高版本资料，其上线状态尚未核实。",
        )
        if (subject := item.subject.strip("《》 ")) in ambiguities
        and item.status == "supported"
        and re.search(r"版本|主题", item.field)
        else item
        for item in selections
    ]


def _query_ngrams(value: str) -> set[str]:
    compact = re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", value.casefold())
    grams: set[str] = set()
    for size in (2, 3):
        grams.update(
            compact[index : index + size]
            for index in range(max(0, len(compact) - size + 1))
        )
    return grams


def _bind_queries_to_retrieval_points(
    queries: list[str], retrieval_points: list[str]
) -> dict[str, str]:
    """Attach audit-generated query variants to the original search branch."""

    points = list(dict.fromkeys(point for point in retrieval_points if point.strip()))
    if not points:
        return {query: query for query in queries}
    point_grams = [_query_ngrams(point) for point in points]
    shared = set.intersection(*point_grams) if len(point_grams) > 1 else set()
    bindings: dict[str, str] = {}
    for query in queries:
        query_grams = _query_ngrams(query)
        scores = [len((grams - shared) & query_grams) for grams in point_grams]
        best_score = max(scores, default=0)
        bindings[query] = points[scores.index(best_score)] if best_score else query
    return bindings


def _document_sources(
    passages: list[MarkdownPassage], question: str, start_index: int = 1
) -> tuple[list[SearchHit], list[DocumentChunk]]:
    sources: list[SearchHit] = []
    chunks: list[DocumentChunk] = []
    for offset, passage in enumerate(passages):
        source_id = f"S{start_index + offset}"
        score = max(0.1, 1.0 - offset * 0.03)
        sources.append(
            SearchHit(
                source_id=source_id,
                title=passage.title,
                url=passage.url,
                content=passage.content,
                provider="local-markdown",
                score=score,
                retrieval_point=passage.filename,
                retrieval_query=question,
            )
        )
        chunks.append(
            DocumentChunk(
                source_id=source_id,
                title=passage.title,
                url=passage.url,
                text=passage.content,
                score=score,
                chunk_index=passage.ordinal,
                retrieval_point=passage.filename,
                retrieval_query=question,
            )
        )
    return sources, chunks


def _conversation_reply(question: str) -> str | None:
    """Handle a tiny exact-match small-talk allow-list without search or a model."""

    canonical = re.sub(r"[\s,，。.!！?？~～]+", "", question).casefold()
    replies = {
        "你好": "你好！有什么想了解的，直接问我就好。",
        "您好": "你好！有什么想了解的，直接问我就好。",
        "嗨": "嗨！有什么想了解的吗？",
        "哈喽": "你好！有什么想了解的吗？",
        "hello": "你好！有什么想了解的吗？",
        "hi": "你好！有什么想了解的吗？",
        "在吗": "在的。你想了解什么？",
        "早上好": "早上好！今天想了解什么？",
        "下午好": "下午好！有什么可以帮你了解的？",
        "晚上好": "晚上好！有什么想了解的吗？",
        "谢谢": "不客气。还有什么想了解的吗？",
        "谢谢你": "不客气。还有什么想了解的吗？",
        "多谢": "不客气。还有什么想了解的吗？",
        "再见": "再见！需要时随时来问。",
        "拜拜": "再见！需要时随时来问。",
        "你是谁": "我是寻答，可以帮你查找资料、阅读来源并整理答案。",
        "你能做什么": "我可以帮你查找资料、阅读来源并整理答案。直接问一个具体问题就好。",
    }
    if re.fullmatch(r"(?:你好|您好|嗨)?(?:请)?(?:简单|简短)?(?:介绍一下|介绍)?(?:你自己|你是谁|你能做什么|自己)", canonical):
        return "我是 RWKV SearchReader，可以帮你解答问题、写作、阅读材料，并在需要时联网查找资料。"
    return replies.get(canonical)


class SearchReaderPipeline:
    """Deterministic greeting reply or Tavily -> parallel RWKV reader pipeline."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.provider = create_search_provider(self.settings)
        self.retriever = Retriever(self.settings, self.provider)
        self.reader = RWKVReader(self.settings)

    def answer(
        self,
        question: str,
        on_progress: Callable[[str, dict[str, object]], None] | None = None,
        session_id: str = "",
        conversation_history: list[dict[str, Any]] | None = None,
        document_passages: list[MarkdownPassage] | None = None,
        *,
        locale: str = "",
        on_delta: Callable[[str], None] | None = None,
        on_sources: Callable | None = None,
    ) -> PipelineResult:
        self.reader.output_locale = locale
        normalized = " ".join(str(question or "").split()).strip()
        if not normalized:
            raise SearchReaderError("input", "question is empty")
        request_id = uuid.uuid4().hex
        timings: dict[str, float] = {}
        started = time.perf_counter()
        trace: list[dict[str, object]] = []
        self.reader.reset_call_records()

        configuration = {
            "search_provider": self.settings.search_provider,
            "search_max_results": self.settings.search_max_results,
            "research_plan_enabled": self.settings.research_plan_enabled,
            "research_plan_max_queries": self.settings.research_plan_max_queries,
            "review_max_rounds": self.settings.review_max_rounds,
            "review_max_queries": self.settings.review_max_queries,
            "conversation_turns": self.settings.conversation_turns,
            "model": self.settings.rwkv_model,
            "final_model": self.settings.final_model,
            "protocol": self.settings.rwkv_protocol,
            "summary_workers": self.settings.rwkv_summary_workers,
            "summary_max_tokens": self.settings.rwkv_summary_max_tokens,
            "reduce_chunk_chars": self.settings.rwkv_reduce_chunk_chars,
            "reduce_fan_in": self.settings.rwkv_reduce_fan_in,
            "final_material_chars": self.settings.rwkv_final_material_chars,
            "document_full_read_max_chunks": (
                self.settings.document_full_read_max_chunks
            ),
            "max_tokens": self.settings.rwkv_max_tokens,
            "temperature": self.settings.rwkv_temperature,
            "max_model_calls_per_request": self.settings.rwkv_max_calls_per_request,
            "trace_retention_days": self.settings.trace_retention_days,
            "trace_max_total_bytes": self.settings.trace_max_total_bytes,
        }

        def emit(stage_name: str, detail: dict[str, object]) -> None:
            at_ms = round((time.perf_counter() - started) * 1000, 1)
            event = {"stage": stage_name, "at_ms": at_ms, "detail": detail}
            trace.append(event)
            if on_progress:
                on_progress(stage_name, {**detail, "at_ms": at_ms})

        try:
            conversation_reply = (
                _conversation_reply(normalized)
                if not locale or locale.lower().startswith("zh") else None
            )
            if conversation_reply is not None:
                emit("responding", {"message": "直接回应简单对话"})
                timings["total"] = round((time.perf_counter() - started) * 1000, 1)
                emit("completed", {"duration_ms": timings["total"]})
                result = PipelineResult(
                    request_id=request_id,
                    session_id=session_id,
                    question=normalized,
                    answer=conversation_reply,
                    queries=[],
                    sources=[],
                    evidence=[],
                    retrieval_mode="local-conversation",
                    search_provider="",
                    reader_mode="deterministic-conversation",
                    degraded=False,
                    warnings=[],
                    timings_ms=timings,
                    trace=trace,
                )
                if self.settings.trace_enabled:
                    append_trace(
                        self.settings.trace_directory,
                        {
                            "schema_version": "rwkv-search-reader.trace.v2",
                            "status": "ok",
                            **result.to_dict(),
                            "configuration": configuration,
                            "model_calls": [],
                        },
                    )
                return result

            pipeline_warnings: list[str] = []
            local_passages = list(document_passages or [])
            history = (
                conversation_history[-self.settings.conversation_turns :]
                if conversation_history is not None
                else session_turns(
                    self.settings.trace_directory,
                    session_id,
                    self.settings.conversation_turns,
                )
            )
            explicit_document_request = bool(_DOCUMENT_REFERENCE_RE.search(normalized))
            if (
                history
                and _history_only_request(normalized)
                and not (local_passages and explicit_document_request)
            ):
                emit(
                    "history_reading",
                    {
                        "turn_count": len(history),
                        "message": "正在阅读此前对话",
                    },
                )
                stage = time.perf_counter()
                answer = self.reader.answer_from_history(normalized, history)
                timings["conversation"] = round((time.perf_counter() - stage) * 1000, 1)
                timings["total"] = round((time.perf_counter() - started) * 1000, 1)
                emit("completed", {"duration_ms": timings["total"]})
                result = PipelineResult(
                    request_id=request_id,
                    session_id=session_id,
                    question=normalized,
                    answer=answer,
                    queries=[],
                    sources=[],
                    evidence=[],
                    retrieval_mode="conversation-history",
                    search_provider="",
                    reader_mode="rwkv-conversation-summary",
                    degraded=False,
                    warnings=[],
                    timings_ms=timings,
                    trace=trace,
                    conversation_state=(
                        history[-1].get("state", {})
                        if isinstance(history[-1].get("state", {}), dict)
                        else {}
                    ),
                )
                if self.settings.trace_enabled:
                    append_trace(
                        self.settings.trace_directory,
                        {
                            "schema_version": "rwkv-search-reader.trace.v2",
                            "status": "ok",
                            **result.to_dict(),
                            "configuration": configuration,
                            "model_calls": self.reader.call_records(),
                        },
                    )
                return result
            effective_question = normalized
            if history and self.reader.contextualization_required(normalized):
                emit(
                    "contextualizing",
                    {"turn_count": len(history), "message": "正在衔接此前对话"},
                )
                try:
                    effective_question = self.reader.contextualize(normalized, history)
                except SearchReaderError as exc:
                    pipeline_warnings.append(
                        f"conversation context degraded: {exc}"[:500]
                    )
                emit(
                    "context_resolved",
                    {
                        "turn_count": len(history),
                        "resolved_question": effective_question,
                    },
                )

            web_enabled = not local_passages or bool(
                _EXPLICIT_WEB_RE.search(normalized)
            )
            research_plan = None
            representative_scope = False
            if (
                web_enabled
                and not local_passages
                and self.settings.research_plan_enabled
            ):
                emit(
                    "planning",
                    {
                        "message": "正在判断问题范围并拆分需要回答的记录",
                    },
                )
                stage = time.perf_counter()
                try:
                    research_plan = self.reader.plan_research(effective_question)
                except SearchReaderError as exc:
                    pipeline_warnings.append(f"research plan degraded: {exc}"[:500])
                    emit(
                        "plan_completed",
                        {
                            "decision": "search",
                            "fallback": True,
                            "queries": [effective_question],
                            "records": [],
                            "message": "范围规划不可用，保留用户原问题检索",
                        },
                    )
                else:
                    if (
                        research_plan.decision == "clarify"
                        and (
                            not self.reader.clarification_required(effective_question)
                            or bool(self.reader.prior_facts(history))
                        )
                    ):
                        research_plan = ResearchPlan(
                            decision="search",
                            goal=effective_question,
                            queries=[effective_question],
                            scope_mode=(
                                "discover"
                                if re.search(
                                    r"各个|所有|哪些|有哪些|某类|一类|类别",
                                    effective_question,
                                )
                                else "fixed"
                            ),
                            fallback=True,
                        )
                        pipeline_warnings.append(
                            "unnecessary clarification was converted to retrieval"
                        )
                    emit(
                        "plan_completed",
                        {
                            "decision": research_plan.decision,
                            "scope_mode": research_plan.scope_mode,
                            "goal": research_plan.goal,
                            "queries": research_plan.queries,
                            "urls": research_plan.direct_urls,
                            "retrieval_points": research_plan.retrieval_points,
                            "clarification": research_plan.clarification,
                            "records": [
                                {
                                    "subject": record.subject,
                                    "question": record.question,
                                    "fields": record.fields,
                                }
                                for record in research_plan.records
                            ],
                            "message": (
                                "问题范围需要用户补充"
                                if research_plan.decision == "clarify"
                                else "已经形成可执行的检索计划"
                            ),
                        },
                    )
                timings["plan"] = round((time.perf_counter() - stage) * 1000, 1)
                if research_plan and research_plan.decision == "clarify":
                    timings["total"] = round((time.perf_counter() - started) * 1000, 1)
                    emit(
                        "clarifying",
                        {
                            "message": research_plan.clarification,
                        },
                    )
                    emit("completed", {"duration_ms": timings["total"]})
                    result = PipelineResult(
                        request_id=request_id,
                        session_id=session_id,
                        question=normalized,
                        answer=research_plan.clarification,
                        queries=[],
                        sources=[],
                        evidence=[],
                        retrieval_mode="model-clarification",
                        search_provider="",
                        reader_mode="rwkv-research-planner",
                        degraded=False,
                        warnings=pipeline_warnings,
                        timings_ms=timings,
                        trace=trace,
                    )
                    if self.settings.trace_enabled:
                        append_trace(
                            self.settings.trace_directory,
                            {
                                "schema_version": "rwkv-search-reader.trace.v2",
                                "status": "ok",
                                **result.to_dict(),
                                "configuration": configuration,
                                "model_calls": self.reader.call_records(),
                            },
                        )
                    return result

            stage = time.perf_counter()
            queries: list[str] = []
            sources: list[SearchHit] = []
            chunks: list[DocumentChunk] = []
            retrieval_warnings: list[str] = []
            stable_retrieval_points: list[str] = []
            direct_url_used = False
            search_used = False
            if web_enabled:
                planned_queries = (
                    research_plan.queries
                    if research_plan and research_plan.decision == "search"
                    else [effective_question]
                )
                direct_urls = (
                    research_plan.direct_urls
                    if research_plan and research_plan.decision == "search"
                    else []
                )
                planned_retrieval_points = list(planned_queries)
                if (
                    research_plan
                    and len(research_plan.retrieval_points) == len(planned_queries)
                ):
                    planned_retrieval_points = list(research_plan.retrieval_points)
                if research_plan and research_plan.decision == "search":
                    normalized_pairs: list[tuple[str, str]] = []
                    seen_queries: set[str] = set()
                    for query, point in zip(
                        planned_queries,
                        planned_retrieval_points,
                        strict=True,
                    ):
                        sanitized = _sanitize_supplemental_query(
                            query, effective_question
                        )
                        key = sanitized.casefold()
                        if sanitized and key not in seen_queries:
                            normalized_pairs.append((sanitized, point))
                            seen_queries.add(key)
                    planned_queries = [query for query, _point in normalized_pairs]
                    planned_retrieval_points = [
                        point for _query, point in normalized_pairs
                    ]
                direct_url_error: SearchReaderError | None = None
                if direct_urls:
                    emit(
                        "url_reading",
                        {
                            "urls": direct_urls,
                            "url_count": len(direct_urls),
                            "message": "正在读取指定网页",
                        },
                    )
                    try:
                        url_queries, url_sources, url_chunks, url_warnings = (
                            self.retriever.run_urls(
                                direct_urls,
                                question=effective_question,
                            )
                        )
                        queries.extend(url_queries)
                        sources.extend(url_sources)
                        chunks.extend(url_chunks)
                        retrieval_warnings.extend(url_warnings)
                        stable_retrieval_points.extend(url_queries)
                        direct_url_used = True
                    except SearchReaderError as exc:
                        direct_url_error = exc
                        pipeline_warnings.append(
                            f"direct URL reading degraded: {exc}"[:1200]
                        )
                if planned_queries:
                    emit(
                        "searching",
                        {
                            "query": planned_queries[0],
                            "queries": planned_queries,
                            "message": "正在查找相关资料",
                        },
                    )
                    try:
                        if research_plan and research_plan.decision == "search":
                            found_queries, found_sources, found_chunks, found_warnings = (
                                self.retriever.run_queries(
                                    planned_queries,
                                    question=effective_question,
                                    start_index=len(sources) + 1,
                                    retrieval_point_by_query=dict(
                                        zip(
                                            planned_queries,
                                            planned_retrieval_points,
                                            strict=True,
                                        )
                                    ),
                                )
                            )
                        else:
                            found_queries, found_sources, found_chunks, found_warnings = (
                                self.retriever.run(effective_question)
                            )
                    except SearchReaderError as exc:
                        if not chunks:
                            raise
                        pipeline_warnings.append(
                            f"supplemental planned search degraded: {exc}"[:1200]
                        )
                    else:
                        queries.extend(found_queries)
                        sources.extend(found_sources)
                        chunks.extend(found_chunks)
                        retrieval_warnings.extend(found_warnings)
                        stable_retrieval_points.extend(planned_retrieval_points)
                        search_used = True
                if not chunks and direct_url_error is not None:
                    raise direct_url_error
                emit(
                    "sources_found",
                    {
                        "queries": queries,
                        "source_count": len(chunks),
                        "sources": [
                            {
                                "source_id": source.source_id or f"S{index}",
                                "title": source.title,
                                "url": source.url,
                                "score": source.score,
                                "published_date": source.published_date,
                                "retrieval_point": source.retrieval_point,
                                "retrieval_query": source.retrieval_query,
                            }
                            for index, source in enumerate(
                                sources[: len(chunks)], start=1
                            )
                        ],
                    },
                )
                if (
                    research_plan
                    and research_plan.decision == "search"
                    and research_plan.scope_mode == "discover"
                ):
                    emit(
                        "scope_discovering",
                        {
                            "source_count": len(sources),
                            "message": "正在从首轮资料中识别具体对象",
                        },
                    )
                    try:
                        scoped_plan = self.reader.discover_research_scope(
                            effective_question,
                            sources,
                            required_fields=research_plan.discovery_fields,
                        )
                    except SearchReaderError as exc:
                        pipeline_warnings.append(
                            f"scope discovery degraded: {exc}"[:500]
                        )
                        emit(
                            "scope_discovery_failed",
                            {"message": "首轮资料未能确认具体对象"},
                        )
                        answer = (
                            "现有资料未能确认这类问题中的具体对象，因此无法可靠逐项核实。"
                            "请指定对象或更明确的范围。"
                            if re.search(r"[\u3400-\u9fff]", normalized)
                            else "The retrieved material did not establish concrete members of this group. "
                            "Please name the subjects or narrow the scope."
                        )
                        timings["retrieve"] = round(
                            (time.perf_counter() - stage) * 1000, 1
                        )
                        timings["total"] = round(
                            (time.perf_counter() - started) * 1000, 1
                        )
                        emit("completed", {"duration_ms": timings["total"]})
                        result = PipelineResult(
                            request_id=request_id,
                            session_id=session_id,
                            question=normalized,
                            answer=answer,
                            queries=queries,
                            sources=sources,
                            evidence=[],
                            retrieval_mode="forced",
                            search_provider=self.provider.name,
                            reader_mode="rwkv-scope-discovery",
                            degraded=True,
                            warnings=[*pipeline_warnings, *retrieval_warnings],
                            timings_ms=timings,
                            trace=trace,
                        )
                        if self.settings.trace_enabled:
                            append_trace(
                                self.settings.trace_directory,
                                {
                                    "schema_version": "rwkv-search-reader.trace.v2",
                                    "status": "degraded",
                                    **result.to_dict(),
                                    "configuration": configuration,
                                    "model_calls": self.reader.call_records(),
                                },
                            )
                        return result
                    else:
                        research_plan = scoped_plan
                        representative_scope = True
                        scoped_queries: list[str] = []
                        scoped_points: list[str] = []
                        tried = {query.casefold() for query in queries}
                        for query, point in zip(
                            scoped_plan.queries,
                            scoped_plan.retrieval_points,
                            strict=True,
                        ):
                            sanitized = _sanitize_supplemental_query(
                                query, effective_question
                            )
                            if sanitized and sanitized.casefold() not in tried:
                                scoped_queries.append(sanitized)
                                scoped_points.append(point)
                                tried.add(sanitized.casefold())
                        emit(
                            "scope_discovered",
                            {
                                "records": [
                                    {
                                        "subject": record.subject,
                                        "question": record.question,
                                        "fields": record.fields,
                                    }
                                    for record in scoped_plan.records
                                ],
                                "queries": scoped_queries,
                                "message": "已从资料中确认代表对象",
                            },
                        )
                        if scoped_queries:
                            emit(
                                "scope_searching",
                                {
                                    "queries": scoped_queries,
                                    "message": "正在逐对象查找当前资料",
                                },
                            )
                            try:
                                (
                                    new_queries,
                                    new_sources,
                                    new_chunks,
                                    new_warnings,
                                ) = self.retriever.run_queries(
                                    scoped_queries,
                                    question=effective_question,
                                    excluded_urls={source.url for source in sources},
                                    start_index=len(sources) + 1,
                                    retrieval_point_by_query=dict(
                                        zip(scoped_queries, scoped_points, strict=True)
                                    ),
                                    subject_by_query=dict(
                                        zip(scoped_queries, scoped_points, strict=True)
                                    ),
                                )
                            except SearchReaderError as exc:
                                pipeline_warnings.append(
                                    f"scope retrieval degraded: {exc}"[:500]
                                )
                            else:
                                queries.extend(new_queries)
                                sources.extend(new_sources)
                                retrieval_warnings.extend(new_warnings)
                                if new_chunks:
                                    # Category pages identify candidates; the
                                    # object pages provide answer evidence.
                                    chunks = new_chunks
                                    stable_retrieval_points = scoped_points
                                emit(
                                    "scope_sources_found",
                                    {
                                        "source_count": len(new_chunks),
                                        "total_source_count": len(sources),
                                        "queries": new_queries,
                                        "message": "已取得逐对象材料",
                                    },
                                )
            if local_passages:
                emit(
                    "document_searching",
                    {
                        "document_count": len(
                            {item.document_id for item in local_passages}
                        ),
                        "message": "正在定位文档中的相关段落",
                    },
                )
                document_sources, document_chunks = _document_sources(
                    local_passages, effective_question, len(chunks) + 1
                )
                sources.extend(document_sources)
                chunks.extend(document_chunks)
                emit(
                    "documents_found",
                    {
                        "document_count": len(
                            {item.document_id for item in local_passages}
                        ),
                        "source_count": len(document_chunks),
                        "sources": [
                            {
                                "source_id": source.source_id,
                                "title": source.title,
                                "url": source.url,
                                "score": source.score,
                                "published_date": "",
                            }
                            for source in document_sources
                        ],
                    },
                )
            if not chunks:
                raise SearchReaderError("search", "no usable web or document material")
            current_release_version: tuple[int, ...] | None = None
            current_record_release_scopes: dict[
                str, tuple[tuple[int, ...], set[str]]
            ] = {}
            if search_used:
                version, release_source_ids = _current_release_scope(
                    effective_question, sources
                )
                release_chunks = [
                    chunk for chunk in chunks if chunk.source_id in release_source_ids
                ]
                if version is not None and release_chunks:
                    current_release_version = version
                    chunks = release_chunks
                    emit(
                        "current_release_scoped",
                        {
                            "version": ".".join(map(str, version)),
                            "source_ids": [chunk.source_id for chunk in chunks],
                            "message": "已按已上线版本筛选当前版本材料",
                        },
                    )
                if research_plan and len(research_plan.records) > 1 and not local_passages:
                    record_scopes = _record_current_release_scopes(
                        effective_question, sources, research_plan.records
                    )
                    if record_scopes:
                        allowed_ids = _record_current_release_ids(
                            sources, research_plan.records, record_scopes
                        )
                        record_chunks = [
                            chunk for chunk in chunks if chunk.source_id in allowed_ids
                        ]
                        if record_chunks:
                            current_record_release_scopes = record_scopes
                            chunks = record_chunks
                            emit(
                                "current_release_scoped",
                                {
                                    "records": {
                                        subject: {
                                            "version": ".".join(map(str, value[0])),
                                            "source_ids": sorted(value[1]),
                                        }
                                        for subject, value in record_scopes.items()
                                    },
                                    "source_ids": [chunk.source_id for chunk in chunks],
                                    "message": "已逐对象筛选已上线版本材料",
                                },
                            )
            timings["retrieve"] = round((time.perf_counter() - stage) * 1000, 1)

            emit("reading", {"source_count": len(chunks)})
            stage = time.perf_counter()
            evidence, summary_warnings = self.reader.summarize(
                effective_question, chunks, on_progress=emit
            )
            timings["summarize"] = round((time.perf_counter() - stage) * 1000, 1)

            audit_warnings: list[str] = []
            audit_sufficient: bool | None = None
            evidence_selections = []
            selection_warnings: list[str] = []
            prior_facts = self.reader.prior_facts(history)

            def refresh_field_selections(round_number: int) -> None:
                nonlocal evidence_selections
                if not (research_plan and research_plan.records):
                    return
                field_count = sum(
                    max(1, len(record.fields)) for record in research_plan.records
                )
                emit(
                    "evidence_selecting",
                    {
                        "round": round_number,
                        "record_count": len(research_plan.records),
                        "field_count": field_count,
                        "message": "正在逐对象逐字段绑定直接来源",
                    },
                )
                selection_started = time.perf_counter()
                active_records = research_plan.records
                deferred_records: list[ResearchRecord] = []
                if (
                    representative_scope
                    and current_record_release_scopes
                    and re.search(r"当前|现在|最新", effective_question)
                    and re.search(r"版本", effective_question)
                ):
                    active_records = [
                        record
                        for record in research_plan.records
                        if record.subject.strip("《》 ") in current_record_release_scopes
                    ]
                    deferred_records = [
                        record
                        for record in research_plan.records
                        if record not in active_records
                    ]
                if active_records:
                    active_evidence = evidence
                    if deferred_records:
                        active_source_ids = {
                            source_id
                            for _version, source_ids in current_record_release_scopes.values()
                            for source_id in source_ids
                        }
                        active_evidence = [
                            item
                            for item in evidence
                            if item.source_id in active_source_ids
                        ]
                    selections, new_warnings = self.reader.select_evidence(
                        effective_question,
                        active_records,
                        active_evidence,
                        on_progress=emit,
                        prior_facts=prior_facts,
                    )
                else:
                    selections, new_warnings = [], []
                selections.extend(
                    FieldEvidenceSelection(
                        subject=record.subject,
                        field=field,
                        source_id="",
                        status="missing",
                        fact="材料未确认该对象已有哪个版本上线。",
                    )
                    for record in deferred_records
                    for field in record.fields
                )
                selection_by_field = {
                    (item.subject, item.field): item for item in selections
                }
                selections = [
                    selection_by_field[(record.subject, field)]
                    for record in research_plan.records
                    for field in record.fields
                    if (record.subject, field) in selection_by_field
                ]
                evidence_selections = _require_online_scope_for_current_records(
                    effective_question,
                    research_plan.records,
                    current_record_release_scopes,
                    selections,
                    representative_scope=representative_scope,
                )
                selection_warnings.extend(new_warnings)
                timings["evidence_select"] = round(
                    timings.get("evidence_select", 0.0)
                    + (time.perf_counter() - selection_started) * 1000,
                    1,
                )

            refresh_field_selections(1)
            audit_started = time.perf_counter()
            invalid_audit_fallback_used = False
            for review_index in range(self.settings.review_max_rounds + 1):
                station_list_complete = bool(evidence_selections) and all(
                    selection.status == "supported"
                    and re.search(
                        r"站点|站名|车站|stations?",
                        selection.field,
                        flags=re.IGNORECASE,
                    )
                    for selection in evidence_selections
                )
                if station_list_complete:
                    audit_sufficient = True
                    emit(
                        "audit_completed",
                        {
                            "round": review_index + 1,
                            "sufficient": True,
                            "message": "已核对完整站点列表及其来源",
                        },
                    )
                    break
                emit(
                    "auditing",
                    {
                        "round": review_index + 1,
                        "source_count": len(evidence),
                        "tried_queries": list(queries),
                        "field_states": [
                            {
                                "subject": item.subject,
                                "field": item.field,
                                "status": item.status,
                                "source_ids": list(item.source_ids),
                            }
                            for item in evidence_selections
                        ],
                        "message": "正在检查材料是否足够",
                    },
                )
                audit_origin = "reader"
                try:
                    if research_plan and research_plan.records:
                        audit = self.reader.audit(
                            effective_question,
                            evidence,
                            queries,
                            research_records=research_plan.records,
                            field_selections=evidence_selections,
                        )
                    else:
                        audit = self.reader.audit(effective_question, evidence, queries)
                except SearchReaderError as exc:
                    audit_origin = "fallback"
                    audit_warnings.append(f"evidence audit degraded: {exc}"[:500])
                    freshness_queries = _freshness_review_queries(
                        effective_question,
                        evidence,
                        queries,
                        evidence_selections,
                        1,
                    )
                    missing_queries = _missing_field_queries(
                        effective_question,
                        evidence_selections,
                        queries,
                        self.settings.review_max_queries,
                    )
                    missing_queries = list(
                        dict.fromkeys(freshness_queries + missing_queries)
                    )[: self.settings.review_max_queries]
                    if not missing_queries and not invalid_audit_fallback_used:
                        fallback_query = _fallback_audit_query(effective_question, queries)
                        if fallback_query:
                            missing_queries = [fallback_query]
                            invalid_audit_fallback_used = True
                    audit = EvidenceAudit(
                        sufficient=bool(evidence_selections)
                        and all(
                            item.status == "supported" for item in evidence_selections
                        ),
                        queries=missing_queries,
                    )

                if audit.sufficient:
                    audit_sufficient = True
                    emit(
                        "audit_completed",
                        {
                            "round": review_index + 1,
                            "sufficient": True,
                            "origin": audit_origin,
                            "message": "现有材料足以回答",
                        },
                    )
                    break

                tried = {query.casefold() for query in queries}
                raw_candidate_queries = [
                    query for query in audit.queries if query.casefold() not in tried
                ]
                candidate_queries = list(
                    dict.fromkeys(
                        sanitized
                        for query in raw_candidate_queries
                        if (
                            sanitized := _sanitize_supplemental_query(
                                query, effective_question
                            )
                        )
                        and sanitized.casefold() not in tried
                    )
                )
                supplemental_queries = [
                    query
                    for query in candidate_queries
                    if not _rechecks_supported_fact(query, effective_question, evidence)
                ]
                if candidate_queries and not supplemental_queries:
                    audit_sufficient = True
                    emit(
                        "audit_completed",
                        {
                            "round": review_index + 1,
                            "sufficient": True,
                            "origin": audit_origin,
                            "message": "关键答案已有多份材料支持，停止重复查找",
                        },
                    )
                    break
                audit_sufficient = False
                emit(
                    "audit_completed",
                    {
                        "round": review_index + 1,
                        "sufficient": False,
                        "origin": audit_origin,
                        "queries": supplemental_queries,
                        "message": "材料仍有缺口，需要补充查找",
                    },
                )
                if review_index >= self.settings.review_max_rounds:
                    audit_warnings.append(
                        "evidence remained incomplete after the maximum review rounds"
                    )
                    break
                if not web_enabled:
                    audit_warnings.append(
                        "attached document material may be incomplete; external search was not requested"
                    )
                    break
                if not supplemental_queries:
                    audit_warnings.append(
                        "evidence audit requested retrieval without a new query"
                    )
                    break

                supplement_round = review_index + 1
                supplemental_point_bindings = _bind_queries_to_retrieval_points(
                    supplemental_queries,
                    stable_retrieval_points,
                )
                emit(
                    "supplemental_searching",
                    {
                        "round": supplement_round,
                        "queries": supplemental_queries,
                        "retrieval_point_by_query": supplemental_point_bindings,
                        "message": "正在补充缺失资料",
                    },
                )
                stage = time.perf_counter()
                try:
                    (
                        new_queries,
                        new_sources,
                        new_chunks,
                        new_retrieval_warnings,
                    ) = self.retriever.run_queries(
                        supplemental_queries,
                        question=effective_question,
                        excluded_urls={source.url for source in sources},
                        start_index=len(sources) + 1,
                        retrieval_point_by_query=supplemental_point_bindings,
                        subject_by_query={
                            query: point
                            for query, point in supplemental_point_bindings.items()
                            if research_plan
                            and point in {
                                record.subject for record in research_plan.records
                            }
                        },
                    )
                except SearchReaderError as exc:
                    audit_warnings.append(
                        f"supplemental retrieval round {supplement_round} failed: {exc}"[
                            :500
                        ]
                    )
                    emit(
                        "supplemental_sources_found",
                        {
                            "round": supplement_round,
                            "source_count": 0,
                            "total_source_count": len(sources),
                            "message": "没有找到新的可用材料",
                        },
                    )
                    break
                timings["retrieve"] += round((time.perf_counter() - stage) * 1000, 1)
                queries.extend(new_queries)
                sources.extend(new_sources)
                retrieval_warnings.extend(new_retrieval_warnings)
                if current_release_version is not None:
                    scoped_sources = [
                        source
                        for source in sources
                        if _title_release_version(source.title)
                        == current_release_version
                    ]
                    official_release = any(
                        _OFFICIAL_TITLE_RE.search(source.title)
                        for source in scoped_sources
                    )
                    new_chunks = [
                        chunk
                        for chunk in new_chunks
                        if _title_release_version(chunk.title) == current_release_version
                        and (
                            not official_release
                            or _OFFICIAL_TITLE_RE.search(chunk.title)
                        )
                    ]
                if research_plan and len(research_plan.records) > 1 and not local_passages:
                    record_scopes = _record_current_release_scopes(
                        effective_question, sources, research_plan.records
                    )
                    if record_scopes:
                        current_record_release_scopes = record_scopes
                        allowed_ids = _record_current_release_ids(
                            sources, research_plan.records, record_scopes
                        )
                        evidence = [
                            item for item in evidence if item.source_id in allowed_ids
                        ]
                        new_chunks = [
                            chunk
                            for chunk in new_chunks
                            if chunk.source_id in allowed_ids
                        ]
                emit(
                    "supplemental_sources_found",
                    {
                        "round": supplement_round,
                        "queries": new_queries,
                        "source_count": len(new_chunks),
                        "total_source_count": len(sources),
                        "sources": [
                            {
                                "source_id": source.source_id,
                                "title": source.title,
                                "url": source.url,
                                "score": source.score,
                                "published_date": source.published_date,
                                "retrieval_point": source.retrieval_point,
                                "retrieval_query": source.retrieval_query,
                            }
                            for source in new_sources
                            if any(
                                chunk.source_id == source.source_id
                                for chunk in new_chunks
                            )
                        ],
                    },
                )
                emit(
                    "supplemental_reading",
                    {
                        "round": supplement_round,
                        "source_count": len(new_chunks),
                        "total_source_count": len(sources),
                    },
                )
                if not new_chunks:
                    if current_record_release_scopes:
                        refresh_field_selections(supplement_round + 1)
                    audit_warnings.append(
                        "supplemental pages did not match the current release"
                        if current_release_version is not None
                        else "supplemental retrieval found no usable material"
                    )
                    break
                stage = time.perf_counter()
                new_evidence, new_summary_warnings = self.reader.summarize(
                    effective_question, new_chunks, on_progress=emit
                )
                timings["summarize"] += round((time.perf_counter() - stage) * 1000, 1)
                evidence.extend(new_evidence)
                summary_warnings.extend(new_summary_warnings)
                refresh_field_selections(supplement_round + 1)

            timings["audit"] = round((time.perf_counter() - audit_started) * 1000, 1)
            if representative_scope and research_plan and current_record_release_scopes:
                ambiguities = _newer_release_ambiguities(
                    sources, research_plan.records, current_record_release_scopes
                )
                if ambiguities:
                    original_sources = {source.source_id: source for source in sources}
                    cited_ids = {item.source_id for item in evidence}
                    for source_ids in ambiguities.values():
                        for source_id in source_ids:
                            source = original_sources.get(source_id)
                            if source is None or source_id in cited_ids:
                                continue
                            evidence.append(
                                Evidence(
                                    source_id=source_id,
                                    title=source.title,
                                    url=source.url,
                                    claim=source.title,
                                    quote=(source.content or source.snippet)[:
                                        self.settings.max_evidence_chars
                                    ],
                                    score=source.score,
                                    published_date=source.published_date,
                                    retrieval_point=source.retrieval_point,
                                    retrieval_query=source.retrieval_query,
                                )
                            )
                            cited_ids.add(source_id)
                    evidence_selections = _flag_unresolved_newer_releases(
                        evidence_selections, ambiguities
                    )
                    audit_sufficient = False
                    emit(
                        "release_freshness_unresolved",
                        {
                            "subjects": ambiguities,
                            "message": "检到同域更高版本，但尚未确认其是否已经上线",
                        },
                    )

            evidence_selections, next_release_masked = _mask_unanchored_next_release_fields(
                effective_question, evidence_selections
            )
            if next_release_masked:
                audit_sufficient = False
                emit(
                    "next_release_unanchored",
                    {
                        "message": "下一个版本没有经核验的未来日期，相关字段不写入已确认状态",
                    },
                )

            stage = time.perf_counter()
            if on_sources is not None:
                on_sources(sources)
            answer, final_warnings = self.reader.finalize(
                effective_question,
                evidence,
                on_progress=emit,
                conversation_history=history,
                current_message=normalized,
                research_records=(research_plan.records if research_plan else None),
                evidence_sufficient=audit_sufficient,
                evidence_selections=evidence_selections,
                representative_scope=representative_scope,
                **({"on_delta": on_delta} if on_delta is not None else {}),
            )
            timings["finalize"] = round((time.perf_counter() - stage) * 1000, 1)
            timings["total"] = round((time.perf_counter() - started) * 1000, 1)
            emit("completed", {"duration_ms": timings["total"]})

            conversation_state = {
                "resolved_question": effective_question,
                "records": [
                    {
                        "subject": record.subject,
                        "question": record.question,
                        "fields": list(record.fields),
                    }
                    for record in (research_plan.records if research_plan else [])
                ],
                "facts": [
                    {
                        "subject": item.subject,
                        "field": item.field,
                        "status": item.status,
                        "fact": item.fact,
                        "source_ids": list(dict.fromkeys([*item.source_ids, item.source_id]))
                        if item.source_id
                        else list(item.source_ids),
                    }
                    for item in evidence_selections
                ],
            }
            warnings = [
                *pipeline_warnings,
                *retrieval_warnings,
                *summary_warnings,
                *audit_warnings,
                *selection_warnings,
                *final_warnings,
                *citation_warnings(answer, evidence),
            ]
            retrieval_modes: list[str] = []
            provider_names: list[str] = []
            if local_passages:
                retrieval_modes.append("document")
                provider_names.append("local-markdown")
            if direct_url_used:
                retrieval_modes.append("direct-url")
                provider_names.append("direct-url")
            if search_used:
                retrieval_modes.append("forced")
                provider_names.append(self.provider.name)
            result = PipelineResult(
                request_id=request_id,
                session_id=session_id,
                question=normalized,
                answer=answer,
                queries=queries,
                sources=sources,
                evidence=evidence,
                retrieval_mode="+".join(retrieval_modes) or "forced",
                search_provider="+".join(provider_names) or self.provider.name,
                reader_mode="rwkv-reviewed-parallel-funnel-final-summary",
                degraded=bool(warnings),
                warnings=warnings,
                timings_ms=timings,
                trace=trace,
                conversation_state=conversation_state,
            )
            if self.settings.trace_enabled:
                append_trace(
                    self.settings.trace_directory,
                    {
                        "schema_version": "rwkv-search-reader.trace.v2",
                        "status": "ok",
                        **result.to_dict(),
                        "configuration": configuration,
                        "model_calls": self.reader.call_records(),
                    },
                )
            return result
        except Exception as exc:
            timings["total"] = round((time.perf_counter() - started) * 1000, 1)
            error_stage = (
                exc.stage if isinstance(exc, SearchReaderError) else "unexpected"
            )
            emit(
                "failed",
                {
                    "error_stage": error_stage,
                    "message": str(exc)[:500],
                    "duration_ms": timings["total"],
                },
            )
            if self.settings.trace_enabled:
                try:
                    append_trace(
                        self.settings.trace_directory,
                        {
                            "schema_version": "rwkv-search-reader.trace.v2",
                            "status": "error",
                            "request_id": request_id,
                            "session_id": session_id,
                            "question": normalized,
                            "error": {
                                "stage": error_stage,
                                "type": type(exc).__name__,
                                "message": str(exc)[:1000],
                            },
                            "timings_ms": timings,
                            "trace": trace,
                            "configuration": configuration,
                            "model_calls": self.reader.call_records(),
                        },
                    )
                except OSError:
                    pass
            raise
