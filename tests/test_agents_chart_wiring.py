"""총괄 차트 배선(src/agents/supervisor.py) 단위 테스트 (TDD — 로직 파트).

검증 대상:
- wants_chart(question): 명시적 차트 키워드가 있을 때만 True(오탐 방지). 재시도 피드백이
  덧붙기 전 '원본 question'으로 판단해야 한다.
- _extract_chart_data(domain_results, conn): 그릴 시계열 데이터를 우선순위
  (backtest > kr/us 가격 > macro)로 하나만 고른다. 그릴 게 없으면 None.
- answer_with_verification: wants_chart가 True이고 그릴 데이터가 있을 때만 성공 응답에
  chart_base64/chart_title를 덧붙인다. 키워드 없으면/불확실 응답이면 차트 필드 없음.
"""
from __future__ import annotations

import base64

from src.agents.supervisor import answer_with_verification, wants_chart

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ── wants_chart — 결정론적 키워드 감지(오탐 방지) ────────────────────────────

def test_wants_chart_true_for_graph_keyword():
    assert wants_chart("삼성전자 최근 1년 주가 그래프 그려줘") is True


def test_wants_chart_true_for_chart_keyword():
    assert wants_chart("골든크로스 전략 백테스트 차트로 보여줘") is True


def test_wants_chart_true_for_english_plot():
    assert wants_chart("plot the nav curve") is True
    assert wants_chart("show me a chart") is True


def test_wants_chart_true_for_visualize_and_trend():
    assert wants_chart("스프레드 시각화 해줘") is True
    assert wants_chart("금리차 추이 보여줘") is True


def test_wants_chart_false_for_plain_question():
    """일반 질문(키워드 없음)은 False — 불필요한 차트 생성을 막는다(오탐 방지)."""
    assert wants_chart("삼성전자 PER 알려줘") is False
    assert wants_chart("지금 매크로 신호 어때?") is False
    assert wants_chart("PER 낮은 5개 회사") is False


def test_wants_chart_empty_is_false():
    assert wants_chart("") is False
    assert wants_chart(None) is False


# ── _extract_chart_data — 우선순위/데이터 선택 ────────────────────────────────

def test_extract_chart_data_backtest_uses_dates_navs_and_benchmark():
    import src.agents.supervisor as sup

    domain_results = {
        "backtest": {
            "blocked": False,
            "result": {
                "dates": ["2024-01-01", "2024-02-01", "2024-03-01"],
                "navs": [1.0, 1.1, 1.2],
                "benchmark": [1.0, 1.05, 1.08],
            },
        }
    }
    out = sup._extract_chart_data(domain_results, conn=None)
    assert out is not None
    dates, series, title = out
    assert dates == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert series["전략"] == [1.0, 1.1, 1.2]
    assert series["벤치마크"] == [1.0, 1.05, 1.08]
    assert "백테스트" in title


def test_extract_chart_data_backtest_without_benchmark():
    import src.agents.supervisor as sup

    domain_results = {"backtest": {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}}
    dates, series, title = sup._extract_chart_data(domain_results, conn=None)
    assert list(series) == ["전략"]
    assert series["전략"] == [1.0, 1.1]


def test_extract_chart_data_prefers_backtest_over_kr(monkeypatch):
    """여러 조건이 동시 해당하면 backtest가 우선(질문당 차트 1개)."""
    import src.agents.supervisor as sup

    # kr 히스토리 함수가 호출되면 실패로 표시 — backtest 우선이면 호출되면 안 된다.
    monkeypatch.setattr(sup, "get_price_history_kr", lambda *a, **k: (_ for _ in ()).throw(AssertionError("kr 우선순위 위반")))
    domain_results = {
        "backtest": {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}},
        "kr": {"stock_code": "005930"},
    }
    _, _, title = sup._extract_chart_data(domain_results, conn=None)
    assert "백테스트" in title


def test_extract_chart_data_kr_price_history(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(
        sup, "get_price_history_kr",
        lambda conn, code, **k: [
            {"stock_code": "005930", "date": "2026-01-01", "close": 70000.0},
            {"stock_code": "005930", "date": "2026-01-02", "close": 71000.0},
        ],
    )
    domain_results = {"kr": {"stock_code": "005930", "price": [{"close": 71000.0}]}}
    dates, series, title = sup._extract_chart_data(domain_results, conn=None)
    assert dates == ["2026-01-01", "2026-01-02"]
    assert list(series.values())[0] == [70000.0, 71000.0]
    assert "005930" in title


def test_extract_chart_data_us_price_history(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(
        sup, "get_price_history_us",
        lambda conn, code, **k: [
            {"stock_code": "AAPL", "date": "2026-01-01", "close": 190.0},
            {"stock_code": "AAPL", "date": "2026-01-02", "close": 195.0},
        ],
    )
    domain_results = {"us": {"ok": True, "stock_code": "AAPL", "price": [{"close": 195.0}]}}
    dates, series, title = sup._extract_chart_data(domain_results, conn=None)
    assert dates == ["2026-01-01", "2026-01-02"]
    assert list(series.values())[0] == [190.0, 195.0]
    assert "AAPL" in title


def test_extract_chart_data_macro_spread_only(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(
        sup, "get_macro_history",
        lambda conn, **k: [
            {"as_of": "2026-01-01", "spread": 0.6},
            {"as_of": "2026-01-02", "spread": 0.4},
        ],
    )
    domain_results = {"macro": {"available": True, "spread": {"value": 0.4}}}
    dates, series, title = sup._extract_chart_data(domain_results, conn=None)
    assert dates == ["2026-01-01", "2026-01-02"]
    assert list(series.values())[0] == [0.6, 0.4]
    assert "금리차" in title


def test_extract_chart_data_returns_none_for_screening_list():
    """스크리닝 결과(여러 종목 리스트, 단일 시계열 아님)는 조용히 None — 에러 아님."""
    import src.agents.supervisor as sup

    domain_results = {"kr": {"intent": "screening", "result": [{"name": "A"}, {"name": "B"}]}}
    assert sup._extract_chart_data(domain_results, conn=None) is None


def test_extract_chart_data_returns_none_when_kr_history_empty(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(sup, "get_price_history_kr", lambda *a, **k: [])
    domain_results = {"kr": {"stock_code": "005930"}}
    assert sup._extract_chart_data(domain_results, conn=None) is None


def test_extract_chart_data_returns_none_for_empty_results():
    import src.agents.supervisor as sup

    assert sup._extract_chart_data({}, conn=None) is None


# ── answer_with_verification — 성공 경로에 chart_base64/chart_title 배선 ──────

def _valid_verify(question, domain_results, llm_fn):
    return {"valid": True, "reason": "일치"}


def test_answer_with_verification_adds_chart_when_requested_backtest():
    bt = {"result": {"dates": ["2024-01-01", "2024-02-01", "2024-03-01"],
                     "navs": [1.0, 1.1, 1.2], "benchmark": [1.0, 1.05, 1.08]}}

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": bt}

    res = answer_with_verification(
        "골든크로스 전략 백테스트 그래프로 보여줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
    )
    assert res["uncertain"] is False
    assert res.get("chart_base64")
    raw = base64.b64decode(res["chart_base64"])
    assert raw[:8] == _PNG_MAGIC          # 실제 PNG인지까지 확인
    assert res.get("chart_title") and "백테스트" in res["chart_title"]


def test_answer_with_verification_no_chart_without_keyword():
    """차트 키워드가 없으면 데이터가 있어도 차트를 만들지 않는다(오탐 방지)."""
    bt = {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": bt}

    res = answer_with_verification(
        "이 전략 수익률 알려줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
    )
    assert res.get("chart_base64") is None


def test_answer_with_verification_uses_original_question_for_wants_chart():
    """재시도 피드백이 붙은 dispatch_question이 아니라 원본 question으로 차트 여부 판단."""
    bt = {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}
    verdicts = iter([{"valid": False, "reason": "1차 실패"}, {"valid": True, "reason": "통과"}])

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": bt}

    def flaky_verify(question, domain_results, llm_fn):
        return next(verdicts)

    res = answer_with_verification(
        "전략 그래프 그려줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=flaky_verify,
    )
    assert res["attempts"] == 2
    assert res.get("chart_base64")  # 원본 질문의 '그려줘'로 차트 생성


def test_answer_with_verification_uncertain_has_no_chart():
    """3회 실패(불확실) 응답에는 차트 필드가 없다(또는 None)."""
    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}}

    def always_invalid(question, domain_results, llm_fn):
        return {"valid": False, "reason": "실패"}

    res = answer_with_verification(
        "전략 그래프 그려줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=always_invalid,
    )
    assert res["uncertain"] is True
    assert res.get("chart_base64") is None


def test_answer_with_verification_no_chart_when_no_series_data():
    """차트 키워드는 있으나 그릴 시계열이 없으면(스크리닝 등) 차트 없이 텍스트 응답만."""
    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"intent": "screening", "result": [{"name": "A"}]}}

    res = answer_with_verification(
        "저PER 종목 그래프로 보여줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
    )
    assert res["uncertain"] is False
    assert res.get("chart_base64") is None
