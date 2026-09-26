from __future__ import annotations

from rwkv_search_reader.models import Evidence, ResearchRecord
from rwkv_search_reader.pipeline import _sanitize_supplemental_query
from rwkv_search_reader.reader import (
    RWKVReader,
    _deterministic_station_answer,
    _bind_query_subjects,
    _field_query_phrase,
    _route_number,
    _verified_source_relation,
)
from rwkv_search_reader.config import Settings


QUESTION = "深圳地铁1号线有哪些站点？"
STATIONS = [
    "罗湖",
    "国贸",
    "老街",
    "大剧院",
    "科学馆",
    "华强路",
    "岗厦",
    "会展中心",
    "购物公园",
    "香蜜湖",
    "车公庙",
    "竹子林",
    "侨城东",
    "华侨城",
    "世界之窗",
    "白石洲",
    "高新园",
    "深大",
    "桃园",
    "大新",
    "鲤鱼门",
    "前海湾",
    "新安",
    "宝安中心",
    "宝体",
    "坪洲",
    "西乡",
    "固戍",
    "后瑞",
    "机场东",
]
SOURCE_TEXT = """**1号线**
沿途站点：罗湖—国贸—老街—大剧院—科学馆—华强路—岗厦—会展中心—购物公园—香蜜湖—车公庙—竹子林—侨城东—华侨城—世界之窗—白石洲—高新园—深大—桃园—大新—鲤鱼门—前海湾—新安—宝安中心—宝体—坪洲—西乡—固戍—后瑞—机场东。

**2号线&8号线**
沿途站点：世界之窗—市民中心—岗厦北—福民—莲塘口岸。
"""


def route_source(source_id: str, url: str, text: str = SOURCE_TEXT) -> Evidence:
    return Evidence(
        source_id=source_id,
        title="深圳地铁线路信息",
        url=url,
        claim="模型生成的摘要可能混入其他线路站名。",
        quote=text,
        score=0.9,
        source_relation=_verified_source_relation(url, "unclear"),
    )


def test_government_hosts_are_identified_as_official() -> None:
    assert _verified_source_relation("https://jtys.sz.gov.cn/line", "unclear") == "official"
    assert _verified_source_relation("https://www.sz.gov.cn/line", "unclear") == "official"


def test_station_schema_field_is_rendered_as_local_search_terms() -> None:
    assert _route_number("甲城地铁三号线") == 3
    assert _field_query_phrase("甲城地铁3号线", "station", "有哪些站点？") == "甲城地铁3号线 站点"


def test_chinese_numeral_route_search_keeps_whole_line_query() -> None:
    reader = RWKVReader(Settings(rwkv_model="test"))
    reader._complete = lambda *args, **kwargs: (
        '{"points":[["甲站","甲城地铁三号线 甲站"],'
        '["戊站","甲城地铁三号线 戊站"]]}'
    )

    queries, _ = reader.design_record_queries(
        "甲城地铁三号线从甲站到戊站依次经过哪些车站？",
        [ResearchRecord("甲城地铁三号线", "从甲站到戊站依次经过哪些车站？", ["车站名称"])],
        [],
    )

    assert queries[0] == "甲城地铁3号线 站点"
    assert "甲城地铁3号线 甲站" in queries
    assert _sanitize_supplemental_query(
        "甲城地铁3号线 站点", "甲城地铁三号线有哪些站点？"
    ) == "甲城地铁3号线 站点"
    assert _bind_query_subjects(
        ["甲城地铁3号线 站点"], ["车站名称"],
        [ResearchRecord("甲城地铁三号线", "有哪些车站？", ["车站名称"])],
    ) == ["甲城地铁3号线 站点"]


def test_two_independent_route_tables_can_confirm_complete_station_order() -> None:
    reader = RWKVReader(Settings(rwkv_model="test"))
    question = "甲城地铁3号线有哪些站点？"
    record = ResearchRecord("甲城地铁3号线", question, ["station"])
    stations = ["甲站", "乙站", "丙站", "丁站", "戊站"]
    table = Evidence(
        "S1", "甲城地铁3号线概览", "https://route-one.example/line/3", "",
        "# 甲城地铁3号线概览\n## 3号线\n5 座车站\n| 车站 | 换乘 |\n| --- | --- |\n"
        + "\n".join(f"| [{name}](/cities/example/stations/{index}) | |" for index, name in enumerate(stations)),
    )
    links = Evidence(
        "S2", "三号线 | 甲城线路", "https://route-two.example/line/3", "",
        "# 三号线\n## 甲站 – 戊站\n"
        + "\n".join(f"[{name}](../../station/{index}/)" for index, name in enumerate(stations)),
    )

    selections, warnings = reader.select_evidence(question, [record], [table, links])
    answer, final_warnings = _deterministic_station_answer(question, [record], selections, [table, links])

    assert selections[0].status == "supported"
    assert selections[0].fact.splitlines() == stations
    assert selections[0].source_ids == ["S1", "S2"]
    assert "nonofficial" in warnings[0]
    assert final_warnings == []
    assert "两份独立线路站表" in answer
    assert answer.endswith("[S1][S2]")


def test_one_nonofficial_route_table_is_insufficient() -> None:
    reader = RWKVReader(Settings(rwkv_model="test"))
    question = "甲城地铁3号线有哪些站点？"
    record = ResearchRecord("甲城地铁3号线", question, ["station"])
    table = Evidence(
        "S1", "甲城地铁3号线概览", "https://route-one.example/line/3", "",
        "# 甲城地铁3号线概览\n## 3号线\n5 座车站\n"
        + "\n".join(f"| [{name}](/cities/example/stations/{index}) | |" for index, name in enumerate(["甲站", "乙站", "丙站", "丁站", "戊站"])),
    )

    selections, _ = reader.select_evidence(question, [record], [table])

    assert selections[0].status == "missing"


def test_timetable_station_column_can_confirm_linked_route_list() -> None:
    reader = RWKVReader(Settings(rwkv_model="test"))
    question = "甲城地铁3号线有哪些站点？"
    record = ResearchRecord("甲城地铁3号线", question, ["station"])
    stations = ["甲站", "乙站", "丙站", "丁站", "戊站"]
    route = Evidence(
        "S1", "甲城地铁3号线概览", "https://route-one.example/line/3", "",
        "# 甲城地铁3号线概览\n## 3号线\n5 座车站\n"
        + "\n".join(f"| [{name}](/cities/example/stations/{index}) | |" for index, name in enumerate(stations)),
    )
    timetable = Evidence(
        "S2", "甲城地铁3号线时刻表", "https://route-two.example/line/3", "",
        "# 甲城地铁3号线时刻表\n| 车站 | 往戊站 |\n| --- | --- |\n"
        + "\n".join(f"| [{name}](/rail/stop/{index}) | 06:00 |" for index, name in enumerate(stations))
        + "\n\n末班车时刻表\n",
    )

    selections, warnings = reader.select_evidence(question, [record], [route, timetable])

    assert selections[0].status == "supported"
    assert selections[0].fact.splitlines() == stations
    assert selections[0].source_ids == ["S1", "S2"]
    assert "nonofficial" in warnings[0]


def test_truncated_timetable_is_not_accepted_as_complete() -> None:
    reader = RWKVReader(Settings(rwkv_model="test"))
    question = "甲城地铁3号线有哪些站点？"
    record = ResearchRecord("甲城地铁3号线", question, ["station"])
    truncated = Evidence(
        "S1", "甲城地铁3号线时刻表", "https://route-two.example/line/3", "",
        "# 甲城地铁3号线时刻表\n| 车站 | 往戊站 |\n| --- | --- |\n"
        + "\n".join(f"| [{name}](/rail/stop/{index}) | 06:00 |" for index, name in enumerate(["甲站", "乙站", "丙站", "丁站", "戊站"])),
    )

    selections, _ = reader.select_evidence(question, [record], [truncated])

    assert selections[0].status == "missing"


def test_station_field_uses_agreeing_official_route_text_only() -> None:
    reader = RWKVReader(Settings(rwkv_model="test"))
    record = ResearchRecord(
        subject="深圳地铁1号线",
        question="有哪些站点？",
        fields=["stations"],
    )
    evidence = [
        route_source("S6", "https://www.sz.gov.cn/content/post_1.html"),
        route_source("S8", "https://jtys.sz.gov.cn/content/post_2.html"),
        route_source(
            "S9",
            "https://example.com/line",
            "深圳地铁1号线站点：岗厦北、福民、莲塘口岸。",
        ),
    ]

    selections, warnings = reader.select_evidence(QUESTION, [record], evidence)

    assert warnings == []
    assert len(selections) == 1
    assert selections[0].status == "supported"
    assert selections[0].fact.splitlines() == STATIONS
    assert selections[0].source_ids == ["S6", "S8"]


def test_conflicting_official_station_lists_are_not_combined() -> None:
    reader = RWKVReader(Settings(rwkv_model="test"))
    record = ResearchRecord(
        subject="深圳地铁1号线",
        question="有哪些站点？",
        fields=["stations"],
    )
    shortened = SOURCE_TEXT.replace("—新安—宝安中心", "—宝安中心")
    evidence = [
        route_source("S6", "https://www.sz.gov.cn/content/post_1.html"),
        route_source("S8", "https://jtys.sz.gov.cn/content/post_2.html", shortened),
    ]

    selections, warnings = reader.select_evidence(QUESTION, [record], evidence)

    assert selections[0].status == "missing"
    assert "disagree" in warnings[0]
    answer, final_warnings = _deterministic_station_answer(
        QUESTION, [record], selections, evidence
    )
    assert "不能可靠列出" in answer
    assert final_warnings


def test_final_station_answer_preserves_order_and_cites_direct_sources() -> None:
    record = ResearchRecord(
        subject="深圳地铁1号线",
        question="有哪些站点？",
        fields=["stations"],
    )
    evidence = [
        route_source("S6", "https://www.sz.gov.cn/content/post_1.html"),
        route_source("S8", "https://jtys.sz.gov.cn/content/post_2.html"),
    ]
    reader = RWKVReader(Settings(rwkv_model="test"))
    selections, _ = reader.select_evidence(QUESTION, [record], evidence)

    answer, warnings = _deterministic_station_answer(
        QUESTION, [record], selections, evidence
    )

    assert warnings == []
    assert "共30站（罗湖至机场东）" in answer
    assert "、".join(STATIONS) in answer
    assert answer.endswith("[S6][S8]")
