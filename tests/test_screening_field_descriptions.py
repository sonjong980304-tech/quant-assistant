"""스크리닝 지표 노출을 "단일 정의처(canonical source)"로 만든 리팩터 검증.

배경: "이 지표가 스크리닝에서 안 됨" 이 세션에서 반복됐다(가장 최근: operating_profit).
원인은 metrics_at() 반환 dict(데이터 계산)와 domain_kr.py의 _KR_SCREEN_FIELDS(LLM
프롬프트 노출용 목록)가 서로 손으로 베껴 적은 별도 목록이라, 하나를 고치면서 다른 하나를
깜빡하기 쉬운 구조였기 때문이다. 이제 src/backtest/data_access.py(KR)/data_access_us.py(US)의
METRIC_FIELD_DESCRIPTIONS(_US) 딕셔너리가 유일한 정의처다. _KR_SCREEN_FIELDS/
_US_SCREEN_FIELDS 는 이 딕셔너리의 key만 파생하며, LLM 프롬프트(_screening_prompt)도
이 딕셔너리를 순회해 "키: 설명"을 동적으로 나열한다. 새 지표 추가 시 이 딕셔너리 한 곳만
고치면 프롬프트 노출과 유효 필드 목록이 자동으로 따라온다.

한글 별칭 사전(_SCREEN_METRIC_ALIASES)은 LLM이 없을 때(또는 실패했을 때) 쓰는 결정론적
폴백 경로에서만 쓰인다 — 주경로(LLM 있음)는 이제 별칭사전을 거치지 않고 프롬프트의
필드 설명을 LLM이 직접 읽고 판단한다.
"""
from __future__ import annotations

import json
import sqlite3

import src.agents.domain_kr as kr
from src.agents.domain_kr import _KR_SCREEN_FIELDS, _screening_prompt
from src.agents.domain_us import _US_SCREEN_FIELDS
from src.backtest.data_access import METRIC_FIELD_DESCRIPTIONS, metrics_at
from src.backtest.data_access_us import METRIC_FIELD_DESCRIPTIONS_US
from src.db import init_db

_META_FIELDS = {"stock_code", "name", "sector", "market", "quarter", "close", "market_cap"}


def _seed_kr_one_stock(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "sync.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        ("005930", "삼성전자", "KOSPI", "반도체"),
    )
    q, disclosed = "2026Q1", "2026-05-15"
    for key, amount in (
        ("operating_profit", 1_000.0), ("revenue", 10_000.0), ("net_income", 800.0),
        ("total_equity", 50_000.0),
    ):
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
            "VALUES (?,?,?,?,?)",
            ("005930", q, disclosed, key, amount),
        )
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
        ("005930", "2026-06-30", 72000.0, 4.1e14),
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# (a) 단일 정의처 ↔ 실제 출력 key 집합 일치 (어긋나면 실패 — 자동 동기화 강제)
# ---------------------------------------------------------------------------
def test_metric_field_descriptions_keys_match_metrics_at_output(tmp_path):
    conn = _seed_kr_one_stock(tmp_path)
    rows = metrics_at(conn, "2026-06-30")
    actual_financial_fields = set(rows[0].keys()) - _META_FIELDS
    assert actual_financial_fields == set(METRIC_FIELD_DESCRIPTIONS.keys())
    conn.close()


def test_kr_screen_fields_derived_from_single_source():
    assert set(_KR_SCREEN_FIELDS) == set(METRIC_FIELD_DESCRIPTIONS.keys())


def test_us_screen_fields_derived_from_single_source():
    assert set(_US_SCREEN_FIELDS) == set(METRIC_FIELD_DESCRIPTIONS_US.keys())


# ---------------------------------------------------------------------------
# (b) 프롬프트가 필드 설명을 동적으로 나열 + LLM이 그 설명만 보고(별칭사전 안 거치고) 매핑
# ---------------------------------------------------------------------------
def test_screening_prompt_lists_operating_profit_description_kr():
    prompt = _screening_prompt("영업이익 가장 높은 기업", _KR_SCREEN_FIELDS, domain="KR")
    assert "operating_profit" in prompt
    assert METRIC_FIELD_DESCRIPTIONS["operating_profit"] in prompt


def test_screening_prompt_lists_operating_profit_description_us():
    prompt = _screening_prompt("영업이익 가장 높은 기업", _US_SCREEN_FIELDS, domain="US")
    assert "operating_profit" in prompt
    assert METRIC_FIELD_DESCRIPTIONS_US["operating_profit"] in prompt


def test_screening_prompt_us_domain_does_not_leak_kr_only_fields():
    """US 프롬프트에는 KR 전용 필드(revenue_growth 등)가 섞이지 않는다."""
    prompt = _screening_prompt("매출 성장률 가장 높은 기업", _US_SCREEN_FIELDS, domain="US")
    assert "revenue_growth" not in prompt


def test_llm_maps_operating_profit_from_prompt_description_not_alias_table():
    """별칭사전을 거치지 않고, LLM이 프롬프트의 필드설명만 보고 스스로 판단한 결과를 채택한다."""
    seen_prompts = []

    def fake_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        return json.dumps(
            {"criteria": [{"key": "operating_profit", "direction": "high"}], "top_n": 1}
        )

    spec, err = kr._extract_screening_spec(
        "26년 1분기 영업이익 가장 높은 기업", fake_llm, _KR_SCREEN_FIELDS, domain="KR",
    )
    assert err is None
    assert spec["criteria"][0]["key"] == "operating_profit"
    # 프롬프트에 실제로 설명이 포함돼 LLM이 판단할 근거가 있었는지 확인.
    assert METRIC_FIELD_DESCRIPTIONS["operating_profit"] in seen_prompts[0]


# ---------------------------------------------------------------------------
# (c) LLM 없을 때 폴백(별칭사전 경로)은 회귀 없이 그대로 동작
# ---------------------------------------------------------------------------
def test_heuristic_fallback_still_maps_operating_profit_without_llm():
    spec = kr._heuristic_screening_spec("영업이익 가장 높은 기업", domain="KR")
    assert spec["criteria"][0]["key"] == "operating_profit"


def test_heuristic_fallback_still_distinguishes_margin_and_growth_from_absolute_regression():
    assert kr._heuristic_screening_spec(
        "영업이익률 가장 높은 기업", domain="KR"
    )["criteria"][0]["key"] == "operating_margin"
    assert kr._heuristic_screening_spec(
        "영업이익성장 가장 높은 기업", domain="KR"
    )["criteria"][0]["key"] == "op_growth"
    assert kr._heuristic_screening_spec(
        "매출성장 가장 높은 기업", domain="KR"
    )["criteria"][0]["key"] == "revenue_growth"
    assert kr._heuristic_screening_spec(
        "순이익률 가장 높은 기업", domain="KR"
    )["criteria"][0]["key"] == "net_margin"


# ---------------------------------------------------------------------------
# (d) 딕셔너리에 필드 하나만 추가하면 프롬프트에 자동으로 나타난다(설계 의도 자체 검증)
# ---------------------------------------------------------------------------
def test_adding_one_field_to_dict_automatically_appears_in_prompt(monkeypatch):
    fake_descriptions = dict(METRIC_FIELD_DESCRIPTIONS)
    fake_descriptions["fake_metric_xyz"] = "가상 테스트 지표 설명"
    monkeypatch.setattr(kr, "METRIC_FIELD_DESCRIPTIONS", fake_descriptions)

    prompt = _screening_prompt(
        "가상지표 높은 기업", tuple(fake_descriptions.keys()), domain="KR",
    )
    assert "fake_metric_xyz" in prompt
    assert "가상 테스트 지표 설명" in prompt
