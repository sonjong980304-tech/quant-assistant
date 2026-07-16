"""매크로 웹 노출 테스트 (MAC-5).

.omc/specs/brainstorming-macro-indicator-agent.md AC15/AC16/AC17 + 회귀(기존 /api/macro).
FastAPI TestClient로 신규 엔드포인트를 검증하고, 기존 /api/macro(환율+지수)가
이름/동작 변경 없이 그대로 동작함을 회귀로 고정한다(라우팅 순서 충돌 방지).

fastapi 미설치 환경(셸 기본 python3)에서는 통째로 skip한다 — 웹 계층은 공유 venv에서만
구동되며, 순수 로직(파이프라인/판정/수집)은 별도 테스트가 fastapi 없이 커버한다.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import web.app as webapp  # noqa: E402
from src.config import CONFIG  # noqa: E402
from src.db import connect, init_db  # noqa: E402
from src.ingest.macro_signal import run_signal  # noqa: E402


@pytest.fixture
def client_with_signals(tmp_path, monkeypatch):
    """macro_signal 이력을 시드한 임시 DB로 CONFIG.db_path를 바꾼 TestClient."""
    db = str(tmp_path / "web_macro.db")
    init_db(db)
    conn = connect(db)
    try:
        run_signal(conn, spread=0.6, cnn=50, vix=14.0, as_of="2026-07-12")   # GREEN
        run_signal(conn, spread=0.2, cnn=30, vix=22.0, as_of="2026-07-13")   # YELLOW
        run_signal(conn, spread=-0.1, cnn=10, vix=35.0, as_of="2026-07-14")  # RED
    finally:
        conn.close()
    monkeypatch.setattr(CONFIG, "db_path", db)
    return TestClient(webapp.app)


# --------------------------------------------------------------------------
# AC15 — GET /api/macro/signal : 최신 신호 + 각 지표 현재값/밴드
# --------------------------------------------------------------------------
def test_api_macro_signal_returns_latest_overall_and_bands(client_with_signals):
    r = client_with_signals.get("/api/macro/signal")
    assert r.status_code == 200
    data = r.json()
    assert data["overall"] == "RED"           # 최신(2026-07-14) 판정
    assert data["as_of"] == "2026-07-14"
    assert data["spread"]["value"] == -0.1
    assert data["spread"]["regime"] == "역전"
    assert data["cnn"]["value"] == 10
    assert data["cnn"]["band"] == "극단공포"
    assert data["vix"]["value"] == 35.0
    assert data["vix"]["band"] == "공포"


def test_api_macro_signal_empty_db_is_graceful(tmp_path, monkeypatch):
    # 이력이 없어도 500이 아니라 available=False로 200 응답.
    db = str(tmp_path / "empty.db")
    init_db(db)
    monkeypatch.setattr(CONFIG, "db_path", db)
    r = TestClient(webapp.app).get("/api/macro/signal")
    assert r.status_code == 200
    data = r.json()
    assert data["available"] is False
    assert data["overall"] is None


# --------------------------------------------------------------------------
# AC16 — GET /api/macro/history?days=N : 최근 N일 시계열(스파크라인용)
# --------------------------------------------------------------------------
def test_api_macro_history_returns_series(client_with_signals):
    r = client_with_signals.get("/api/macro/history?days=7")
    assert r.status_code == 200
    data = r.json()
    assert data["days"] == 7
    series = data["series"]
    assert len(series) == 3                              # 시드한 3일치
    assert [s["as_of"] for s in series] == ["2026-07-12", "2026-07-13", "2026-07-14"]  # 과거→최신
    assert series[0]["overall"] == "GREEN"
    assert series[-1]["overall"] == "RED"
    assert {"as_of", "overall", "spread", "cnn", "vix"} <= set(series[0].keys())


def test_api_macro_history_respects_days_limit(client_with_signals):
    r = client_with_signals.get("/api/macro/history?days=2")
    assert r.status_code == 200
    series = r.json()["series"]
    assert len(series) == 2                              # 최근 2일만
    assert [s["as_of"] for s in series] == ["2026-07-13", "2026-07-14"]


# --------------------------------------------------------------------------
# AC17 — GET /macro : macro.html 서빙
# --------------------------------------------------------------------------
def test_macro_route_serves_html(client_with_signals):
    r = client_with_signals.get("/macro")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "매크로" in body or "macro" in body.lower()


# --------------------------------------------------------------------------
# 회귀 — 기존 GET /api/macro(환율+지수)가 이름/동작 그대로 유지된다.
# 신규 /api/macro/signal, /api/macro/history가 /api/macro를 가로채지 않는다.
# --------------------------------------------------------------------------
def test_existing_api_macro_still_works(monkeypatch):
    monkeypatch.setattr(webapp, "fetch_usdkrw_rate_live", lambda: 1350.5)
    monkeypatch.setattr(webapp, "_fetch_index_quote",
                        lambda ticker: {"close": 100.0, "change_pct": 1.23})
    monkeypatch.setattr(webapp, "_fetch_krx_index_quote",
                        lambda naver_code: {"close": 200.0, "change_pct": 2.34})
    r = TestClient(webapp.app).get("/api/macro")
    assert r.status_code == 200
    data = r.json()
    # 기존 응답 스키마 그대로: usdkrw / indices / note / fetched_at
    assert data["usdkrw"] == {"rate": 1350.5}
    assert isinstance(data["indices"], list) and len(data["indices"]) == 6
    assert "note" in data and "fetched_at" in data


# --------------------------------------------------------------------------
# 코스피/코스닥은 yfinance(^KS11/^KQ11)가 며칠씩 지연·결측되는 문제가 실서버에서
# 재현됨(2026-07-13 이후 데이터가 안 붙어 등락률이 -8.95%로 왜곡). 네이버 실시간
# 지수 API로 교체해 당일 종가를 정확히 반영한다(개별종목 가격의 기존 "네이버가
# source of truth" 원칙과 동일).
# --------------------------------------------------------------------------
def test_kospi_kosdaq_use_naver_realtime_index_not_yfinance(monkeypatch):
    calls = []

    def fake_krx(naver_code):
        calls.append(naver_code)
        return {"close": 6856.83, "change_pct": 0.73} if naver_code == "KOSPI" else {"close": 783.98, "change_pct": -1.92}

    def fake_yf(ticker):
        # US 지수는 여전히 yfinance 경로를 타야 한다(호출되면 그 자체가 증거).
        return {"close": 100.0, "change_pct": 0.0}

    monkeypatch.setattr(webapp, "fetch_usdkrw_rate_live", lambda: 1490.9)
    monkeypatch.setattr(webapp, "_fetch_krx_index_quote", fake_krx)
    monkeypatch.setattr(webapp, "_fetch_index_quote", fake_yf)

    r = TestClient(webapp.app).get("/api/macro")
    assert r.status_code == 200
    data = r.json()
    by_label = {item["label"]: item for item in data["indices"]}

    assert by_label["코스피"]["close"] == 6856.83
    assert by_label["코스피"]["change_pct"] == 0.73
    assert by_label["코스닥"]["close"] == 783.98
    # 코스피/코스닥은 네이버 코드("KOSPI"/"KOSDAQ")로 호출됨 — yfinance 티커(^KS11 등)로 안 감.
    assert set(calls) == {"KOSPI", "KOSDAQ"}
    # 나머지 4개(나스닥/다우/S&P500/필라델피아반도체)는 그대로 yfinance 경로.
    us_labels = {"나스닥종합", "다우존스", "S&P500", "필라델피아반도체"}
    for label in us_labels:
        assert by_label[label]["close"] == 100.0


def test_fetch_krx_index_quote_parses_naver_comma_formatted_fields(monkeypatch):
    """네이버 API는 숫자를 "6,856.83" 같은 쉼표 포함 문자열로 준다 — 그대로 float 변환하면 깨진다."""
    class _FakeResp:
        def json(self):
            return {"datas": [{
                "closePrice": "6,856.83",
                "fluctuationsRatio": "0.73",
            }]}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(webapp.requests, "get", lambda *a, **k: _FakeResp())
    result = webapp._fetch_krx_index_quote("KOSPI")
    assert result == {"close": 6856.83, "change_pct": 0.73}


# --------------------------------------------------------------------------
# 회귀(실사용 버그) — 프론트가 60초마다 /api/macro를 폴링하는데, 환율만 당일 캐시된
# get_usdkrw_rate()를 쓰고 있어 하루 중 첫 요청 값이 그날 내내 고정돼 "실시간으로
# 안 바뀐다"는 문제가 재현됐다. fetch_usdkrw_rate_live()로 바꿔 매 요청마다 새로
# 가져오는지 확인한다(코스피/코스닥/해외지수와 동일한 무캐시 정책).
# --------------------------------------------------------------------------
def test_api_macro_usdkrw_refetches_every_request_no_daily_cache(monkeypatch):
    rates = iter([1500.0, 1501.5])
    monkeypatch.setattr(webapp, "fetch_usdkrw_rate_live", lambda: next(rates))
    monkeypatch.setattr(webapp, "_fetch_index_quote",
                        lambda ticker: {"close": 100.0, "change_pct": 1.23})
    monkeypatch.setattr(webapp, "_fetch_krx_index_quote",
                        lambda naver_code: {"close": 200.0, "change_pct": 2.34})
    client = TestClient(webapp.app)

    first = client.get("/api/macro").json()["usdkrw"]["rate"]
    second = client.get("/api/macro").json()["usdkrw"]["rate"]

    assert first == 1500.0
    assert second == 1501.5  # 캐시됐다면 첫 값(1500.0)과 같았을 것


def test_macro_routes_are_distinct_and_not_shadowed():
    paths = {getattr(rt, "path", None): getattr(rt, "endpoint", None) for rt in webapp.app.routes}
    assert "/api/macro" in paths
    assert "/api/macro/signal" in paths
    assert "/api/macro/history" in paths
    assert paths["/api/macro"].__name__ == "api_macro"   # 기존 함수가 그대로 바인딩됨
