"""POST /api/backtest 회귀 테스트 — 지표 별칭 치환 + 사용자 입력 오류 400 반환.

실사용 재현 버그: 백테스트 UI에서 '가격모멘텀'(metric_def key='momentum')을 선택해 실행하면
백테스트 단면에 'momentum' 필드가 없어(있는 것은 return_12m) selection._validate_criteria_keys가
ValueError를 던지고, 그게 잡히지 않아 500(평문 'Internal Server Error')으로 나갔다. 프런트의
r.json() 이 그 평문을 파싱하다 실패해 사파리에서 "The string did not match the expected pattern."
라는 엉뚱한 메시지가 화면에 떴다.

이 테스트는 서버 계약만 고정한다(실제 DB/백테스트 엔진은 monkeypatch로 대체):
- momentum → return_12m 로 별칭 치환돼 엔진에 넘어간다.
- 존재하지 않는 지표(ValueError) 는 500 이 아니라 400(JSON detail) 으로 반환된다.

fastapi 미설치 환경(셸 기본 python3)에서는 통째로 skip 한다(다른 web 테스트 관례와 동일).
공유 venv(/Users/gyuyeong/projects/.venv)에서 실행하면 통과한다.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import src.backtest.data_access as bt_da  # noqa: E402
import src.backtest.engine as bt_engine  # noqa: E402
import web.app as webapp  # noqa: E402


class _Cursor:
    def __init__(self, one=None, rows=()):
        self._one = one
        self._rows = list(rows)

    def fetchone(self):
        return self._one

    def __iter__(self):
        return iter(self._rows)


class _FakeConn:
    """api_backtest 가 쓰는 SQL 세 개(MAX(date)/company/close)만 최소 대응."""

    def __init__(self):
        self.closed = False

    def execute(self, sql, *args):
        if "MAX(date)" in sql:
            return _Cursor(one=["2026-12-31"])
        if "FROM company" in sql:
            return _Cursor(rows=[])       # 성공 경로: 종목명 매핑용(holdings 비어있으면 안 쓰임)
        return _Cursor(one=[None], rows=[])

    def close(self):
        self.closed = True


def _patch_common(monkeypatch):
    monkeypatch.setattr(webapp, "connect", lambda *a, **k: _FakeConn())
    monkeypatch.setattr(bt_da, "rebalance_dates",
                        lambda sy, ey, rb: ["2024-01-31", "2024-04-30", "2024-07-31"])
    monkeypatch.setattr(bt_da, "build_callbacks",
                        lambda conn: ((lambda d: [{}]), (lambda code, d: None)))
    monkeypatch.setattr(bt_da, "build_benchmark_fn",
                        lambda dates, mfn, pfn: None)
    monkeypatch.setattr(bt_engine, "save_backtest_run", lambda *a, **k: None)


def test_momentum_is_aliased_to_return_12m(monkeypatch):
    _patch_common(monkeypatch)
    captured = {}

    def fake_run_backtest(dates, mfn, pfn, params, benchmark_fn=None):
        captured["criteria"] = params["criteria"]
        return {"dates": [], "navs": [], "benchmark": None,
                "performance": {}, "holdings": []}

    monkeypatch.setattr(bt_engine, "run_backtest", fake_run_backtest)
    client = TestClient(webapp.app)

    r = client.post("/api/backtest", json={
        "criteria": [{"key": "momentum", "direction": "high", "weight": 1},
                     {"key": "gp_a", "direction": "high", "weight": 1}]})

    assert r.status_code == 200, r.text
    # momentum 만 return_12m 으로 치환되고, direction/weight 등 다른 필드와 gp_a 는 그대로.
    assert captured["criteria"] == [
        {"key": "return_12m", "direction": "high", "weight": 1},
        {"key": "gp_a", "direction": "high", "weight": 1}]


class _MaxdConn(_FakeConn):
    """MAX(date)를 원하는 값으로 돌려주는 _FakeConn 변형(진행중 구간 이어붙이기 검증용)."""

    def __init__(self, maxd):
        super().__init__()
        self._maxd = maxd

    def execute(self, sql, *args):
        if "MAX(date)" in sql:
            return _Cursor(one=[self._maxd])
        if "FROM company" in sql:
            return _Cursor(rows=[])
        return _Cursor(one=[None], rows=[])


def _patch_trailing(monkeypatch, maxd, full):
    monkeypatch.setattr(webapp, "connect", lambda *a, **k: _MaxdConn(maxd))
    monkeypatch.setattr(bt_da, "rebalance_dates", lambda sy, ey, rb: list(full))
    monkeypatch.setattr(bt_da, "build_callbacks",
                        lambda conn: ((lambda d: [{}]), (lambda code, d: None)))
    monkeypatch.setattr(bt_da, "build_benchmark_fn", lambda dates, mfn, pfn: None)
    monkeypatch.setattr(bt_engine, "save_backtest_run", lambda *a, **k: None)


def _capture_dates(monkeypatch, captured):
    def fake_run_backtest(dates, mfn, pfn, params, benchmark_fn=None):
        captured["dates"] = list(dates)
        return {"dates": list(dates), "navs": [1.0] * len(dates), "benchmark": None,
                "performance": {}, "holdings": []}

    monkeypatch.setattr(bt_engine, "run_backtest", fake_run_backtest)


def test_future_rebalance_extends_to_latest_data(monkeypatch):
    """다음 리밸런싱이 아직 미래라 잘리면, 데이터 최신일(maxd)을 말단 구간으로 이어붙인다.

    실사용 재현: 반기 전략(1월말/7월말)에서 오늘이 2026-07-17이고 데이터 최신일이 2026-07-16이면
    다음 리밸런싱 2026-07-31이 미도래라 잘려 차트가 2026-01-31에서 멈추는 것처럼 보였다. 이제
    현재 보유 포트폴리오의 미완결 구간을 2026-07-16까지 이어 붙여 최근 데이터까지 반영한다.
    """
    full = ["2024-01-31", "2024-07-31", "2025-01-31", "2025-07-31",
            "2026-01-31", "2026-07-31"]
    _patch_trailing(monkeypatch, "2026-07-16", full)
    captured = {}
    _capture_dates(monkeypatch, captured)
    client = TestClient(webapp.app)

    r = client.post("/api/backtest", json={
        "start_year": 2024, "end_year": 2026, "rebalance": "semiannual",
        "criteria": [{"key": "gp_a", "direction": "high", "weight": 1}]})

    assert r.status_code == 200, r.text
    # 미래 리밸런싱(2026-07-31)은 빠지고, 그 대신 데이터 최신일(2026-07-16)이 말단에 붙는다.
    assert captured["dates"] == ["2024-01-31", "2024-07-31", "2025-01-31",
                                 "2025-07-31", "2026-01-31", "2026-07-16"]


def test_past_end_year_does_not_extend(monkeypatch):
    """종료연도가 과거인 백테스트는 잘린 미래 리밸런싱이 없으므로 말단 연장을 하지 않는다."""
    full = ["2022-01-31", "2022-07-31", "2023-01-31", "2023-07-31"]
    _patch_trailing(monkeypatch, "2026-07-16", full)
    captured = {}
    _capture_dates(monkeypatch, captured)
    client = TestClient(webapp.app)

    r = client.post("/api/backtest", json={
        "start_year": 2022, "end_year": 2023, "rebalance": "semiannual",
        "criteria": [{"key": "gp_a", "direction": "high", "weight": 1}]})

    assert r.status_code == 200, r.text
    # 요청 종료연도(2023) 리밸런싱이 전부 데이터 범위 안이라(잘림 없음) 그대로 실행된다.
    assert captured["dates"] == full


def test_unknown_metric_returns_400_not_500(monkeypatch):
    _patch_common(monkeypatch)

    def fake_run_backtest(dates, mfn, pfn, params, benchmark_fn=None):
        # selection._validate_criteria_keys 가 실제로 던지는 예외를 그대로 흉내낸다.
        raise ValueError("존재하지 않는 필드: ['pcr']. 사용 가능한 필드: ['gp_a', 'return_12m']")

    monkeypatch.setattr(bt_engine, "run_backtest", fake_run_backtest)
    client = TestClient(webapp.app)

    r = client.post("/api/backtest", json={
        "criteria": [{"key": "pcr", "direction": "low", "weight": 1}]})

    assert r.status_code == 400            # 500 이 아니라 400
    assert "존재하지 않는 필드" in r.json()["detail"]


def test_domain_default_preserves_kr_pipeline(monkeypatch):
    """domain 생략 시 기본 'kr' — 기존 KR 경로가 그대로 돈다(회귀 방지)."""
    _patch_common(monkeypatch)
    captured = {}

    def fake_run_backtest(dates, mfn, pfn, params, benchmark_fn=None):
        captured["ran"] = True
        return {"dates": [], "navs": [], "benchmark": None, "performance": {}, "holdings": []}

    monkeypatch.setattr(bt_engine, "run_backtest", fake_run_backtest)
    client = TestClient(webapp.app)

    r = client.post("/api/backtest", json={
        "criteria": [{"key": "per", "direction": "low", "weight": 1}]})

    assert r.status_code == 200, r.text
    assert captured.get("ran") is True


def test_invalid_domain_returns_400(monkeypatch):
    """kr 이외의 domain은 사용자 입력 오류이므로 400."""
    _patch_common(monkeypatch)
    client = TestClient(webapp.app)

    r = client.post("/api/backtest", json={
        "domain": "jp",
        "criteria": [{"key": "per", "direction": "low", "weight": 1}]})

    assert r.status_code == 400, r.text
    assert "domain" in r.json()["detail"]


def test_domain_kr_rejects_us_market(monkeypatch):
    """대칭 검증: domain='kr'(기본)에서 US 거래소값(NASDAQ)을 보내면 400."""
    _patch_common(monkeypatch)
    client = TestClient(webapp.app)

    r = client.post("/api/backtest", json={
        "markets": ["NASDAQ"],
        "criteria": [{"key": "per", "direction": "low", "weight": 1}]})

    assert r.status_code == 400, r.text
    assert "시장" in r.json()["detail"]


class _SectorConn:
    """GET /api/sectors 의 도메인별 DISTINCT sector 조회만 대응."""

    def __init__(self):
        self.closed = False

    def execute(self, sql, *args):
        if "FROM company" in sql:
            return _Cursor(rows=[{"sector": "전기전자"}])
        return _Cursor(rows=[])

    def close(self):
        self.closed = True


def test_sectors_domain_default_queries_kr_company(monkeypatch):
    """domain 생략 시 기존 company(KR) 테이블 조회 그대로(회귀 방지)."""
    monkeypatch.setattr(webapp, "connect", lambda *a, **k: _SectorConn())
    client = TestClient(webapp.app)

    r = client.get("/api/sectors")

    assert r.status_code == 200, r.text
    assert r.json() == ["전기전자"]


def test_winsorize_z_flows_into_engine_params(monkeypatch):
    """BacktestReq.winsorize_z 가 /api/backtest params dict를 거쳐 엔진까지 전달되는지 배선 검증."""
    _patch_common(monkeypatch)
    captured = {}

    def fake_run_backtest(dates, mfn, pfn, params, benchmark_fn=None):
        captured["winsorize_z"] = params.get("winsorize_z")
        return {"dates": [], "navs": [], "benchmark": None, "performance": {}, "holdings": []}

    monkeypatch.setattr(bt_engine, "run_backtest", fake_run_backtest)
    client = TestClient(webapp.app)

    # 명시하면 그 값이 그대로 params에 실린다.
    r = client.post("/api/backtest", json={
        "criteria": [{"key": "gp_a", "direction": "high", "weight": 1}], "winsorize_z": 3.0})
    assert r.status_code == 200, r.text
    assert captured["winsorize_z"] == 3.0

    # 미지정(기본값)이면 None — 기존 동작(회귀 없음).
    r = client.post("/api/backtest", json={
        "criteria": [{"key": "gp_a", "direction": "high", "weight": 1}]})
    assert r.status_code == 200, r.text
    assert captured["winsorize_z"] is None


def test_winsorize_pct_flows_into_engine_params(monkeypatch):
    """BacktestReq.winsorize_pct 가 /api/backtest params dict를 거쳐 엔진까지 전달되는지 배선 검증."""
    _patch_common(monkeypatch)
    captured = {}

    def fake_run_backtest(dates, mfn, pfn, params, benchmark_fn=None):
        captured["winsorize_pct"] = params.get("winsorize_pct")
        return {"dates": [], "navs": [], "benchmark": None, "performance": {}, "holdings": []}

    monkeypatch.setattr(bt_engine, "run_backtest", fake_run_backtest)
    client = TestClient(webapp.app)

    r = client.post("/api/backtest", json={
        "criteria": [{"key": "gp_a", "direction": "high", "weight": 1}], "winsorize_pct": 0.01})
    assert r.status_code == 200, r.text
    assert captured["winsorize_pct"] == 0.01

    r = client.post("/api/backtest", json={
        "criteria": [{"key": "gp_a", "direction": "high", "weight": 1}]})
    assert r.status_code == 200, r.text
    assert captured["winsorize_pct"] is None
