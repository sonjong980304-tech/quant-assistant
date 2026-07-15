"""router_node가 KR뿐 아니라 US 종목도 실시간 가격 캐치 대상에 포함하는지 (TDD, C-5 AC5).

.omc/specs/brainstorming-us-nl-sql-integration.md 참고. 기존 router_node는 company
테이블의 KR 종목코드만 ensure_live_prices에 넘겼다 — us_company 종목도 함께
ensure_live_prices_us로 넘겨야 "오늘 애플 주가" 같은 질의가 실시간 반영된다.

실사용 재현 버그(2026-07-14) 이후: router_node는 이제 테이블 전체가 아니라 질문에
실제로 name이 언급된 종목만 넘긴다(_mentioned_stock_codes, 리터럴 부분일치 —
LLM 미사용). "애플"처럼 us_company.name(영문 정식명, 예: "Apple Inc. Common
Stock")과 문자 그대로 일치하지 않는 한글 콜로키얼 명칭은 이 라우터 단계에서
실시간 갱신 대상으로 못 잡는다(known trade-off) — 그래도 최종 답변 자체는 이미
DB에 있는 캐시 데이터로 정상 응답되며, 단지 "오늘자 강제 갱신"만 스킵될 뿐이다.
전체 종목(국내 3,924+미국 7,123)을 매번 실시간 호출하던 이전 동작은 질문 하나에
외부 API 호출 수천 건을 유발해 요청이 사실상 멈춘 것처럼 보일 만큼 느렸다.
"""
from __future__ import annotations

import sqlite3
from datetime import date

from src.db import init_db
from src.legacy.graph.nodes import Deps, make_nodes
from tests.conftest import FakeLLM, seed_kr_companies, seed_us_companies


def _deps_with_schema(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    seed_kr_companies(conn, ["000001"])
    seed_us_companies(conn, ["AAPL"])
    return Deps(conn=conn, store=None, schema="(schema)", today=date(2026, 7, 12), llm=FakeLLM("{}"))


def test_router_calls_ensure_live_prices_us_for_live_price_question(tmp_path, monkeypatch):
    import src.legacy.graph.nodes as nodes_module

    calls = {"kr": None, "us": None}
    monkeypatch.setattr(
        nodes_module, "ensure_live_prices",
        lambda conn, codes, on: calls.__setitem__("kr", list(codes)) or {"fetched": 0, "cached": len(codes), "today": "2026-07-12"},
    )
    monkeypatch.setattr(
        nodes_module, "ensure_live_prices_us",
        lambda conn, codes, on: calls.__setitem__("us", list(codes)) or {"fetched": 0, "cached": len(codes), "today": "2026-07-12"},
    )

    deps = _deps_with_schema(tmp_path)
    nodes = make_nodes(deps)
    nodes["router_node"]({"question": "오늘 애플 주가 얼마야", "raw_question": "오늘 애플 주가 얼마야"})

    # 질문에 언급된 종목만 대상이어야 한다 — "000001"(seed_kr_companies가 이름=코드로
    # 채운 시드 종목)은 질문에 등장하지 않으므로 KR 쪽은 빈 리스트가 정답이다. US 쪽도
    # seed_us_companies가 name="AAPL"(티커 그대로)로 채우는데 질문은 한글 "애플"이라
    # 리터럴 일치가 안 돼 빈 리스트다(모듈 docstring의 known trade-off) — 최종 답변은
    # 이미 캐시된 데이터로 정상 응답되고, "오늘자 강제 갱신"만 스킵된다.
    assert calls["kr"] == []
    assert calls["us"] == []


def test_router_scopes_live_price_calls_to_mentioned_companies_only(tmp_path, monkeypatch):
    """실사용 재현 버그: 언급 안 된 종목까지 통째로 실시간 조회하면 회사가 많을수록
    요청이 사실상 멈춘 것처럼 느려진다(실측: 국내 3,924 + 미국 7,123 종목 전체를
    ensure_live_prices(_us)에 넘겨 질문 하나에 외부 API 호출 수천 건 발생).
    질문에 실제로 이름이 나온 종목만 넘겨야 한다 — seed 종목이 여러 개일 때만
    "우연히 1개라 전체=언급종목"이었던 기존 테스트의 사각지대를 드러낸다."""
    import src.legacy.graph.nodes as nodes_module

    calls = {"kr": None, "us": None}
    monkeypatch.setattr(
        nodes_module, "ensure_live_prices",
        lambda conn, codes, on: calls.__setitem__("kr", sorted(codes)) or {"fetched": 0, "cached": len(codes), "today": "2026-07-12"},
    )
    monkeypatch.setattr(
        nodes_module, "ensure_live_prices_us",
        lambda conn, codes, on: calls.__setitem__("us", sorted(codes)) or {"fetched": 0, "cached": len(codes), "today": "2026-07-12"},
    )

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        ("005930", "삼성전자", "KOSPI", "전기·전자"),
    )
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        ("000660", "SK하이닉스", "KOSPI", "전기·전자"),
    )
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("AAPL", "Apple", "NASDAQ", "Technology", 1.0e9, "2026-07-12T00:00:00"),
    )
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("MSFT", "Microsoft", "NASDAQ", "Technology", 1.0e9, "2026-07-12T00:00:00"),
    )
    conn.commit()
    deps = Deps(conn=conn, store=None, schema="(schema)", today=date(2026, 7, 12), llm=FakeLLM("{}"))
    nodes = make_nodes(deps)

    nodes["router_node"]({"question": "오늘 삼성전자 종가 얼마야", "raw_question": "오늘 삼성전자 종가 얼마야"})

    assert calls["kr"] == ["005930"]  # SK하이닉스(000660)는 언급 안 됐으므로 제외
    assert calls["us"] == []          # 미국 종목은 질문에 아예 언급 없음


def test_router_skips_live_price_calls_when_question_has_no_live_hint(tmp_path, monkeypatch):
    import src.legacy.graph.nodes as nodes_module

    calls = {"kr": False, "us": False}
    monkeypatch.setattr(nodes_module, "ensure_live_prices", lambda *a, **kw: calls.__setitem__("kr", True))
    monkeypatch.setattr(nodes_module, "ensure_live_prices_us", lambda *a, **kw: calls.__setitem__("us", True))

    deps = _deps_with_schema(tmp_path)
    nodes = make_nodes(deps)
    nodes["router_node"]({"question": "애플 PER 알려줘", "raw_question": "애플 PER 알려줘"})

    assert calls["kr"] is False
    assert calls["us"] is False
