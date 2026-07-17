"""신호 기반 기술지표 백테스트 프리미티브 run_signal_backtest 단위 테스트 (TDD).

배경: 기존 run_backtest는 "매 리밸런싱마다 팩터 점수로 상위 N개 담기"(크로스섹셔널)만
지원해, "삼성전자 20/60일 이동평균 골든크로스면 매수·데드크로스면 매도" 같은 개별종목
시그널 매매타이밍 전략을 표현할 수 없었다. 이 테스트는 새 프리미티브가:
  - entry_rule/exit_rule 신호를 올바른 타이밍(t일 확정 → t+1일 체결, 미래참조 금지)에 반영하고,
  - 반환 형식({dates,navs,benchmark,performance,holdings})이 auditor.post_audit을 그대로
    통과하며,
  - 종목 수 상한/기존 파이프라인 상한과 충돌하지 않는지
를 결정론적 fixture(지표 시계열을 DI로 주입 — TA-Lib 수치 자체는 검증 대상 아님)로 검증한다.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.backtest.signal_engine import run_signal_backtest
from src.db import init_db


# --------------------------------------------------------------------------
# 결정론 fixture 헬퍼: price_history_fn / indicator_series_fn 주입
# --------------------------------------------------------------------------
def _history_fn_factory(series_by_code: dict):
    """{code: [(date, close), ...]} → price_history_fn(conn, codes, asof, lookback_days)."""
    def fn(conn, codes, asof, lookback_days):
        return {
            c: [{"date": d, "close": px} for d, px in series_by_code.get(c, [])]
            for c in codes
        }
    return fn


def _indicator_fn_factory(arrays_by_name_period: dict):
    """{(name, period): [values...]} → indicator_series_fn(name, closes, period)."""
    def fn(name, closes, period=None):
        return list(arrays_by_name_period[(name, period)])
    return fn


_DATES6 = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]


# --------------------------------------------------------------------------
# 골든크로스/데드크로스 매매 타이밍(t일 확정 → t+1일 체결) + NAV + holdings
# --------------------------------------------------------------------------
def test_golden_cross_entry_and_dead_cross_exit_timing_and_nav():
    # sma(2)=단기, sma(3)=장기. 단기가 j=2에서 장기 위로 교차(골든크로스) → t+1(j=3) 체결,
    # j=2..: 단기가 j=4에서 장기 아래로 교차(데드크로스) → t+1(j=5) 청산.
    # 체결가는 체결일 종가 → 보유수익은 체결 다음날부터 반영(j=4, j=5 수익만 포착).
    short = [1, 1, 3, 3, 3, 1]   # sma period=2
    long = [2, 2, 2, 2, 4, 2]    # sma period=3
    closes = [100.0, 100.0, 100.0, 100.0, 110.0, 121.0]
    history_fn = _history_fn_factory({"AAA": list(zip(_DATES6, closes))})
    indicator_fn = _indicator_fn_factory({("sma", 2): short, ("sma", 3): long})

    out = run_signal_backtest(
        conn=None, stock_codes=["AAA"],
        start_date=_DATES6[0], end_date=_DATES6[-1],
        entry_rule={"left": {"kind": "indicator", "name": "sma", "period": 2},
                    "op": "cross_above",
                    "right": {"kind": "indicator", "name": "sma", "period": 3}},
        exit_rule={"left": {"kind": "indicator", "name": "sma", "period": 2},
                   "op": "cross_below",
                   "right": {"kind": "indicator", "name": "sma", "period": 3}},
        params={"fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
        price_history_fn=history_fn, indicator_series_fn=indicator_fn,
    )
    assert out["dates"] == _DATES6
    assert out["benchmark"] is None
    # 보유수익은 j=4(110/100), j=5(121/110)에서만 → nav 1→1.1→1.21, 그 전엔 평탄.
    assert out["navs"] == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.1, 1.21])
    # 보유 종목 집합이 바뀐 날(j=4, 현금→AAA)만 holdings에 기록. 구간수익률(j=4,5 보유:
    # 1.1×1.1-1)도 함께 저장된다.
    assert out["holdings"] == [
        {"date": "2026-01-08", "codes": ["AAA"], "period_return": pytest.approx(0.21)}
    ]
    assert "cagr" in out["performance"] and "mdd" in out["performance"]


def test_signal_on_last_day_is_not_executed_no_lookahead():
    """마지막 날 진입신호가 확정돼도 t+1 체결일이 창(window) 밖이라 실행되지 않는다(미래참조 금지)."""
    short = [1, 1, 1, 1, 3]   # 마지막 j=4에서만 골든크로스
    long = [2, 2, 2, 2, 2]
    closes = [100.0, 100.0, 100.0, 100.0, 100.0]
    dates5 = _DATES6[:5]
    history_fn = _history_fn_factory({"AAA": list(zip(dates5, closes))})
    indicator_fn = _indicator_fn_factory({("sma", 2): short, ("sma", 3): long})
    out = run_signal_backtest(
        conn=None, stock_codes=["AAA"], start_date=dates5[0], end_date=dates5[-1],
        entry_rule={"left": {"kind": "indicator", "name": "sma", "period": 2},
                    "op": "cross_above",
                    "right": {"kind": "indicator", "name": "sma", "period": 3}},
        exit_rule={"left": {"kind": "indicator", "name": "sma", "period": 2},
                   "op": "cross_below",
                   "right": {"kind": "indicator", "name": "sma", "period": 3}},
        params={"fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
        price_history_fn=history_fn, indicator_series_fn=indicator_fn,
    )
    assert out["navs"] == pytest.approx([1.0] * 5)  # 진입 미실행 → 완전 평탄
    assert out["holdings"] == []


def test_multi_code_equal_weight_when_both_held():
    """여러 종목을 동시에 보유하면 균등가중으로 배분한다(보유 종목 평균 수익률)."""
    dates4 = _DATES6[:4]
    hist = {
        "AAA": list(zip(dates4, [100.0, 100.0, 110.0, 121.0])),
        "BBB": list(zip(dates4, [50.0, 50.0, 60.0, 72.0])),
    }
    history_fn = _history_fn_factory(hist)
    # price>0 상시참, 청산조건 price<0 은 상시거짓 → 두 종목 모두 즉시 진입(t+1) 후 계속 보유.
    entry = {"left": {"kind": "price"}, "op": ">", "right": {"kind": "const", "value": 0.0}}
    exit_ = {"left": {"kind": "price"}, "op": "<", "right": {"kind": "const", "value": 0.0}}
    out = run_signal_backtest(
        conn=None, stock_codes=["AAA", "BBB"], start_date=dates4[0], end_date=dates4[-1],
        entry_rule=entry, exit_rule=exit_,
        params={"fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
        price_history_fn=history_fn,
        indicator_series_fn=lambda *a, **k: None,  # price/const만 쓰므로 지표 불필요
    )
    # j=2,3에서 두 종목 보유: 각 날 평균수익 = (0.1+0.2)/2 = 0.15 → 1.15, 1.3225.
    assert out["navs"] == pytest.approx([1.0, 1.0, 1.15, 1.3225])
    entry = next(h for h in out["holdings"] if h["date"] == "2026-01-06")
    assert entry["codes"] == ["AAA", "BBB"]
    # 단일 보유구간(전 구간 보유) → 구간수익률 = 최종 nav-1 = 0.3225.
    assert entry["period_return"] == pytest.approx(0.3225)


def test_holdings_log_records_period_return_per_segment():
    """각 보유구간(리밸런싱 구간)의 실제 포트폴리오 수익률이 holdings 항목에 저장된다.
    비용 0이면 구간수익률들을 곱한 값이 최종 nav와 일치해야 한다(불변식)."""
    import math

    # AAA는 전 구간 보유(항상 >100), BBB는 뒤늦게 편입(j>=3부터 >100) → 보유집합이 바뀌며 2개 구간.
    dates7 = _DATES6 + ["2026-01-12"]
    aaa = [200.0, 200.0, 200.0, 220.0, 242.0, 242.0, 266.2]
    bbb = [50.0, 50.0, 50.0, 150.0, 165.0, 181.5, 199.65]
    hist = {"AAA": list(zip(dates7, aaa)), "BBB": list(zip(dates7, bbb))}
    entry = {"left": {"kind": "price"}, "op": ">", "right": {"kind": "const", "value": 100.0}}
    exit_ = {"left": {"kind": "price"}, "op": "<", "right": {"kind": "const", "value": 0.0}}
    out = run_signal_backtest(
        conn=None, stock_codes=["AAA", "BBB"], start_date=dates7[0], end_date=dates7[-1],
        entry_rule=entry, exit_rule=exit_,
        params={"fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
        price_history_fn=_history_fn_factory(hist),
        indicator_series_fn=lambda *a, **k: None,  # price/const만 쓰므로 지표 불필요
    )
    holdings = out["holdings"]
    assert len(holdings) >= 2                            # 보유집합이 바뀌어 여러 구간 발생
    assert all("period_return" in h for h in holdings)   # 모든 구간에 구간수익률 저장
    prod = math.prod(1 + h["period_return"] for h in holdings)
    assert prod == pytest.approx(out["navs"][-1])        # 비용0 → 구간수익률 누적곱 = 최종 nav


def test_rejects_empty_stock_codes():
    with pytest.raises(ValueError):
        run_signal_backtest(
            conn=None, stock_codes=[], start_date="2026-01-01", end_date="2026-12-31",
            entry_rule={"left": {"kind": "price"}, "op": ">", "right": {"kind": "const", "value": 0.0}},
            exit_rule={"left": {"kind": "price"}, "op": "<", "right": {"kind": "const", "value": 0.0}},
            price_history_fn=_history_fn_factory({}),
        )


def test_rejects_more_than_ten_stock_codes():
    codes = [f"C{i:02d}" for i in range(11)]
    with pytest.raises(ValueError):
        run_signal_backtest(
            conn=None, stock_codes=codes, start_date="2026-01-01", end_date="2026-12-31",
            entry_rule={"left": {"kind": "price"}, "op": ">", "right": {"kind": "const", "value": 0.0}},
            exit_rule={"left": {"kind": "price"}, "op": "<", "right": {"kind": "const", "value": 0.0}},
            price_history_fn=_history_fn_factory({c: [] for c in codes}),
        )


# --------------------------------------------------------------------------
# auditor 배선: 반환 형식이 post_audit을 그대로 통과 + pre_audit 오탐 없음
# --------------------------------------------------------------------------
def _seed_conn(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "sig.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
                 ("005930", "삼성전자", "KOSPI", "전기·전자"))
    conn.commit()
    return conn


def test_result_passes_post_audit_end_to_end(tmp_path):
    from src.agents.backtest_verification import run_backtest_with_audit

    conn = _seed_conn(tmp_path)
    hist = {"005930": list(zip(_DATES6, [100.0, 100.0, 100.0, 100.0, 110.0, 121.0]))}
    result = run_signal_backtest(
        conn=None, stock_codes=["005930"], start_date=_DATES6[0], end_date=_DATES6[-1],
        entry_rule={"left": {"kind": "indicator", "name": "sma", "period": 2},
                    "op": "cross_above",
                    "right": {"kind": "indicator", "name": "sma", "period": 3}},
        exit_rule={"left": {"kind": "indicator", "name": "sma", "period": 2},
                   "op": "cross_below",
                   "right": {"kind": "indicator", "name": "sma", "period": 3}},
        params={"fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
        price_history_fn=_history_fn_factory(hist),
        indicator_series_fn=_indicator_fn_factory(
            {("sma", 2): [1, 1, 3, 3, 3, 1], ("sma", 3): [2, 2, 2, 2, 4, 2]}
        ),
    )
    steps = [{"op": "run_signal_backtest", "params": {}, "out": "bt"}]
    audit = run_backtest_with_audit(
        steps, conn, "삼성전자 골든크로스 백테스트",
        run_pipeline_fn=lambda s, conn=None: result, llm_fn=None, market="KR",
    )
    assert audit["blocked"] is False                  # 살아있는 종목 → 하드차단 없음
    assert audit["result"]["navs"][-1] == pytest.approx(1.21)
    sins = {v["sin"] for v in audit["hard"]}
    assert "survivorship" in sins and "lookahead" in sins  # 하드검사 실제 실행됨


def test_pre_audit_skips_signal_op_without_false_positive(tmp_path):
    from src.backtest import auditor

    conn = _seed_conn(tmp_path)
    steps = [{"op": "run_signal_backtest",
              "params": {"stock_codes": ["005930"]}, "out": "bt"}]
    # 공매도(음수 비중) 사전검사는 run_backtest(weights) 전용 — 신호 백테스트엔 비중이 없다.
    # 크래시 없이 None(개입 안 함)을 돌려줘야 한다(오탐으로 정상 실행을 막지 않음).
    verdict = auditor.pre_audit(steps, conn, run_pipeline_fn=lambda s, conn=None: {})
    assert verdict is None


# --------------------------------------------------------------------------
# 파이프라인 실행기 등록(고정 dict 디스패치 + conn 자동주입 대상)
# --------------------------------------------------------------------------
def test_registered_in_pipeline_ops_and_needs_conn():
    from src.backtest.pipeline_exec import _NEEDS_CONN, PRIMITIVE_OPS

    assert "run_signal_backtest" in PRIMITIVE_OPS
    assert "run_signal_backtest" in _NEEDS_CONN


def test_pipeline_prompt_documents_signal_backtest():
    from src.agents.domain_backtest import _PIPELINE_PROMPT

    assert "run_signal_backtest" in _PIPELINE_PROMPT
    assert "골든크로스" in _PIPELINE_PROMPT  # 예시 포함


# --------------------------------------------------------------------------
# 기본 경로 스모크: 실제 TA-Lib으로 지표 계산(수치정확성 아닌 배선만 확인)
# --------------------------------------------------------------------------
def test_default_indicator_path_uses_real_talib(tmp_path):
    pytest.importorskip("talib")
    # 70거래일 상승 추세 → 단기/장기 이동평균이 계산되고 골든크로스가 잡혀 매수가 발생.
    import datetime

    base = datetime.date(2026, 1, 1)
    series = []
    for i in range(70):
        d = (base + datetime.timedelta(days=i)).isoformat()
        series.append((d, 100.0 + i))  # 단조 상승
    history_fn = _history_fn_factory({"AAA": series})
    out = run_signal_backtest(
        conn=None, stock_codes=["AAA"],
        start_date=series[65][0], end_date=series[-1][0],
        entry_rule={"left": {"kind": "indicator", "name": "sma", "period": 5},
                    "op": "cross_above",
                    "right": {"kind": "indicator", "name": "sma", "period": 20}},
        exit_rule={"left": {"kind": "indicator", "name": "sma", "period": 5},
                   "op": "cross_below",
                   "right": {"kind": "indicator", "name": "sma", "period": 20}},
        price_history_fn=history_fn,  # indicator_series_fn 미주입 → 실제 talib 경로
    )
    assert out["benchmark"] is None
    assert out["navs"][0] == 1.0
    assert len(out["navs"]) == len(out["dates"])
    assert "cagr" in out["performance"]


# --------------------------------------------------------------------------
# search_signal_strategy — 개별종목 시그널 전략 탐색(여러 규칙 후보 자동 시도)
# --------------------------------------------------------------------------
# 후보 규칙 헬퍼: 항상 진입(price>0)·절대 청산 안 함(price<0) = 계속 보유,
#              절대 진입 안 함(price<0)·항상 청산(price>0) = 계속 현금(평탄).
_ENTRY_ALWAYS = {"left": {"kind": "price"}, "op": ">", "right": {"kind": "const", "value": 0.0}}
_EXIT_NEVER = {"left": {"kind": "price"}, "op": "<", "right": {"kind": "const", "value": 0.0}}
_ENTRY_NEVER = {"left": {"kind": "price"}, "op": "<", "right": {"kind": "const", "value": 0.0}}
_EXIT_ALWAYS = {"left": {"kind": "price"}, "op": ">", "right": {"kind": "const", "value": 0.0}}
# 진입 후 계속 보유하면 매 봉 +10% → 우상향(수익률·샤프 양수).
_RISING_CLOSES = [100.0, 100.0, 110.0, 121.0, 133.1, 146.41]


def _search_kwargs(**overrides):
    """search_signal_strategy 호출 기본 인자(상승 시계열 1종목, 지표 불필요)."""
    from src.backtest.signal_engine import search_signal_strategy  # noqa: F401

    hist = {"AAA": list(zip(_DATES6, _RISING_CLOSES))}
    base = dict(
        conn=None, stock_codes=["AAA"], start_date=_DATES6[0], end_date=_DATES6[-1],
        market="KR",
        price_history_fn=_history_fn_factory(hist),
        indicator_series_fn=lambda *a, **k: None,  # price/const만 쓰므로 지표 불필요
    )
    base.update(overrides)
    return base


def test_search_signal_strategy_rejects_empty_candidates():
    from src.backtest.signal_engine import search_signal_strategy

    with pytest.raises(ValueError):
        search_signal_strategy(**_search_kwargs(candidates=[]))


def test_search_signal_strategy_rejects_too_many_candidates():
    from src.backtest.signal_engine import (
        _MAX_SIGNAL_SEARCH_CANDIDATES,
        search_signal_strategy,
    )

    one = {"entry_rule": _ENTRY_ALWAYS, "exit_rule": _EXIT_NEVER}
    too_many = [one] * (_MAX_SIGNAL_SEARCH_CANDIDATES + 1)
    with pytest.raises(ValueError):
        search_signal_strategy(**_search_kwargs(candidates=too_many))


def test_search_signal_strategy_rejects_bad_constraint_op_before_running():
    from src.backtest.signal_engine import search_signal_strategy

    cands = [{"entry_rule": _ENTRY_ALWAYS, "exit_rule": _EXIT_NEVER}]
    with pytest.raises(ValueError):
        search_signal_strategy(**_search_kwargs(
            candidates=cands,
            constraints=[{"metric": "total_return", "op": "≥", "value": 5.0}],
        ))


def test_search_signal_strategy_returns_satisfying_candidates_ranked():
    from src.backtest.signal_engine import search_signal_strategy

    cands = [
        {"entry_rule": _ENTRY_NEVER, "exit_rule": _EXIT_ALWAYS},   # 평탄(수익 0)
        {"entry_rule": _ENTRY_ALWAYS, "exit_rule": _EXIT_NEVER},   # 우상향(수익 +46%)
    ]
    out = search_signal_strategy(**_search_kwargs(
        candidates=cands,
        constraints=[{"metric": "total_return", "op": ">=", "value": 5.0}],
        rank_by="sharpe",
    ))
    assert out["constraints_met"] is True
    # 제약 만족(우상향)이 1위, best는 그것과 동일 참조.
    assert out["best"] is out["results"][0]
    assert out["results"][0]["constraints_met"] is True
    # 우상향 +46%에서 기본 거래비용 차감 → 45.81(결정론적). 제약(>=5)은 충분히 만족.
    assert out["results"][0]["performance"]["total_return"] == pytest.approx(45.81)
    assert out["results"][0]["entry_rule"] == _ENTRY_ALWAYS
    # 미충족(평탄) 후보도 결과에 남되 뒤로 밀리고 constraints_met=False.
    assert out["results"][-1]["constraints_met"] is False
    # 반환 형식: 후보별로 navs/dates/holdings 포함(auditor·차트 배선 호환).
    assert out["results"][0]["navs"][-1] == pytest.approx(1.4581, abs=1e-3)
    assert out["results"][0]["dates"] == _DATES6


def test_search_signal_strategy_best_effort_when_none_satisfies():
    """제약을 만족하는 후보가 하나도 없어도 에러/빈손이 아니라 '가장 근접한 시도'를 정직히 반환."""
    from src.backtest.signal_engine import search_signal_strategy

    cands = [
        {"entry_rule": _ENTRY_NEVER, "exit_rule": _EXIT_ALWAYS},   # 0%
        {"entry_rule": _ENTRY_ALWAYS, "exit_rule": _EXIT_NEVER},   # +46% (그래도 100% 미달)
    ]
    out = search_signal_strategy(**_search_kwargs(
        candidates=cands,
        constraints=[{"metric": "total_return", "op": ">=", "value": 100.0}],
        rank_by="sharpe",
    ))
    assert out["constraints_met"] is False           # 아무도 못 넘김
    assert out["best"] is not None                    # 그래도 최선의 시도를 준다
    assert out["best"] is out["results"][0]
    assert len(out["results"]) == 2                   # 후보 전부 결과에 남음
    assert all(r["constraints_met"] is False for r in out["results"])
    # rank_by(sharpe) 내림차순 → 우상향(거래비용 차감 후 +45.81%)이 최선의 시도로 1위.
    assert out["best"]["performance"]["total_return"] == pytest.approx(45.81)


def test_search_signal_strategy_no_constraints_marks_all_met():
    from src.backtest.signal_engine import search_signal_strategy

    cands = [{"entry_rule": _ENTRY_ALWAYS, "exit_rule": _EXIT_NEVER}]
    out = search_signal_strategy(**_search_kwargs(candidates=cands))
    assert out["constraints_met"] is True
    assert out["results"][0]["constraints_met"] is True


def test_search_signal_strategy_skips_failing_candidate_and_continues():
    """한 후보가 예외(잘못된 op)를 던지면 그 후보만 건너뛰고 나머지는 계속 진행."""
    from src.backtest.signal_engine import search_signal_strategy

    bad = {"entry_rule": {"left": {"kind": "price"}, "op": "bogus_op",
                          "right": {"kind": "const", "value": 0.0}},
           "exit_rule": _EXIT_NEVER}
    good = {"entry_rule": _ENTRY_ALWAYS, "exit_rule": _EXIT_NEVER}
    out = search_signal_strategy(**_search_kwargs(candidates=[bad, good]))
    assert len(out["results"]) == 1                   # 나쁜 후보는 조용히 skip
    assert out["results"][0]["entry_rule"] == _ENTRY_ALWAYS


def test_search_signal_strategy_all_candidates_fail_raises_clear_error():
    from src.backtest.signal_engine import search_signal_strategy

    bad = {"entry_rule": {"left": {"kind": "price"}, "op": "bogus_op",
                          "right": {"kind": "const", "value": 0.0}},
           "exit_rule": _EXIT_NEVER}
    with pytest.raises(ValueError) as exc:
        search_signal_strategy(**_search_kwargs(candidates=[bad, bad]))
    # 디버깅 가능하도록 마지막 예외 사유를 포함해야 한다.
    assert "bogus_op" in str(exc.value) or "op" in str(exc.value)


def test_search_signal_strategy_registered_in_pipeline_ops_and_needs_conn():
    from src.backtest.pipeline_exec import _NEEDS_CONN, PRIMITIVE_OPS

    assert "search_signal_strategy" in PRIMITIVE_OPS
    assert "search_signal_strategy" in _NEEDS_CONN


def test_pipeline_prompt_documents_search_signal_strategy():
    from src.agents.domain_backtest import _PIPELINE_PROMPT

    assert "search_signal_strategy" in _PIPELINE_PROMPT
