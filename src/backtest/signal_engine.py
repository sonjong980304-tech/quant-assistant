"""신호 기반(개별종목 매매타이밍) 백테스트 엔진.

기존 engine.run_backtest는 "매 리밸런싱마다 팩터 점수로 상위 N개를 담는" 크로스섹셔널
팩터 전략만 표현한다. 이 모듈은 "삼성전자 20일/60일 이동평균 골든크로스면 매수,
데드크로스면 매도" 같은 **개별종목 시그널 매매타이밍 전략**을 표현하는 run_signal_backtest를
추가한다.

불변 규칙(engine.run_backtest와 동일 정신):
- 미래참조 금지: t일 종가로 확정된 신호는 t+1일 종가에 체결한다(체결일 종가부터 보유수익 반영).
- 거래비용 반드시 반영: 보유 종목 집합이 바뀌는 날(리밸런싱)에 회전율×거래비용을 차감한다
  (engine.run_backtest의 cost_per_turn 모델을 그대로 재사용 — 새 비용모델을 만들지 않는다).
- 롱온리: 각 종목은 "현금 보유" 또는 "매수 후 보유" 두 상태뿐(공매도 없음).
- 균등가중: 동시에 보유하는 종목들에 동일 비중 배분(보유 집합이 바뀌는 날에만 리밸런싱).

반환 형식은 auditor.post_audit이 무수정으로 소비할 수 있도록 engine.run_backtest와 동일하게
{dates, navs, benchmark, performance, holdings:[{date, codes}]} 를 지킨다.

지표 계산(SERIES kind=indicator)은 primitives.compute_indicator_series(전 구간 TA-Lib 배열)를
재사용하고, 가격 시계열은 data_access.price_history_batch(KR)/data_access_us.us_price_history_batch(US)를
재사용한다 — 모두 DI로 주입 가능(기본은 실제 구현)해 TA-Lib/DB 없이 단위테스트가 된다.
"""
from __future__ import annotations

import datetime
import math
from typing import Callable

from .performance import performance

# 신호 백테스트는 일단위 시뮬레이션이라 연환산 계수는 거래일 기준 252를 쓴다.
_PERIODS_PER_YEAR_DAILY = 252
_MAX_SIGNAL_CODES = 10          # 개별종목 시그널 전략 상한(스펙 확정 — 1~10개)
_WARMUP_BUFFER_DAYS = 40        # 거래일→캘린더일 환산(주말·공휴일) 여유 버퍼
_VALID_OPS = {"cross_above", "cross_below", ">", "<"}
# 시그널 전략 탐색 후보 상한. 크로스섹셔널 search_strategy(_MAX_SEARCH_CANDIDATES)와 값을
# 통일한다(사용자 지정) — 다만 이쪽은 후보마다 run_signal_backtest가 종목별(최대 10개) 전
# 구간 일단위 시뮬레이션을 새로 돌리고 가격/지표를 후보 간 공유하지 않아(search_strategy는
# 콜백을 후보 간 재사용해 캐시를 공유) 후보당 비용이 더 무겁다. 최악의 경우(20×10=200회
# 일단위 시뮬레이션)는 파이프라인 상한(MAX_TIMEOUT=120초)을 넘을 수 있다 — 그 경우 무한정
# 도는 게 아니라 run_pipeline이 TimeoutError로 안전하게 실패한다(하드 상한 자체는 유지됨).
_MAX_SIGNAL_SEARCH_CANDIDATES = 20
_DEFAULT_INDICATOR_PERIODS = {  # SERIES에 period가 없을 때 lookback 산정용 대략치
    "sma": 20, "ema": 20, "rsi": 14, "macd": 35, "macd_signal": 35,
    "bollinger_upper": 20, "bollinger_lower": 20,
}


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or v != v


def _series_for(spec: dict, closes: list[float], indicator_series_fn: Callable) -> list:
    """SERIES 스펙을 종목 종가 시계열(closes)에 맞춘 값 배열로 해석한다."""
    kind = spec.get("kind")
    if kind == "price":
        return list(closes)
    if kind == "const":
        return [float(spec["value"])] * len(closes)
    if kind == "indicator":
        return list(indicator_series_fn(spec["name"], closes, spec.get("period")))
    raise ValueError(f"지원하지 않는 SERIES kind: {kind} (price|const|indicator만 가능)")


def _eval_op(op: str, left: list, right: list, j: int) -> bool:
    """지표/가격 배열 left,right의 인덱스 j에서 연산 op의 참/거짓. NaN(워밍업)은 항상 거짓."""
    lv, rv = left[j], right[j]
    if _isnan(lv) or _isnan(rv):
        return False
    if op == ">":
        return lv > rv
    if op == "<":
        return lv < rv
    # 교차(cross)는 직전 봉과의 관계가 필요 — 첫 봉(j=0)이나 직전 NaN이면 판정 불가(거짓).
    if j == 0:
        return False
    lp, rp = left[j - 1], right[j - 1]
    if _isnan(lp) or _isnan(rp):
        return False
    if op == "cross_above":
        return lp <= rp and lv > rv
    if op == "cross_below":
        return lp >= rp and lv < rv
    raise ValueError(f"지원하지 않는 op: {op} ({sorted(_VALID_OPS)}만 가능)")


def _own_by_date_for_code(
    dates: list[str], closes: list[float],
    entry_rule: dict, exit_rule: dict, indicator_series_fn: Callable,
) -> dict[str, bool]:
    """한 종목의 '해당 거래일에 보유 중인가'(own) 여부를 날짜별로 계산한다.

    상태기계: 현금이면 entry 신호에 매수, 보유면 exit 신호에 매도. 신호는 그날 종가로 확정되나
    체결은 t+1일 종가 → 실제로 그날 수익을 얻는 보유 여부는 신호 상태를 2봉 지연시킨 값이다
    (own[j] = state[j-2]: state[j-2]는 j-2일 종가에 확정된 상태 → j-1일 종가에 체결 →
    j일 수익(close[j]/close[j-1])을 그 포지션이 얻음). 이 지연이 미래참조를 원천 차단한다.
    """
    n = len(closes)
    e_left = _series_for(entry_rule["left"], closes, indicator_series_fn)
    e_right = _series_for(entry_rule["right"], closes, indicator_series_fn)
    x_left = _series_for(exit_rule["left"], closes, indicator_series_fn)
    x_right = _series_for(exit_rule["right"], closes, indicator_series_fn)

    state = [False] * n
    holding = False
    for j in range(n):
        if not holding and _eval_op(entry_rule["op"], e_left, e_right, j):
            holding = True
        elif holding and _eval_op(exit_rule["op"], x_left, x_right, j):
            holding = False
        state[j] = holding

    own: dict[str, bool] = {}
    for j, d in enumerate(dates):
        own[d] = state[j - 2] if j >= 2 else False
    return own


def _turnover(prev: set, new: set) -> float:
    """보유 집합 변화의 회전율(engine.run_backtest와 동일 정신: 새로 편입된 비중).

    전량 청산(new 비어있음, prev 있음)은 1.0(전액 매도)으로 본다.
    """
    if not new:
        return 1.0 if prev else 0.0
    return len(new - prev) / len(new)


def _max_lookback_days(rules: list[dict]) -> int:
    """entry/exit 규칙의 지표 기간 중 최대치를 캘린더일 워밍업으로 넉넉히 환산한다."""
    periods = [_DEFAULT_INDICATOR_PERIODS.get("sma", 20)]
    for rule in rules:
        for side in ("left", "right"):
            spec = rule.get(side) or {}
            if spec.get("kind") == "indicator":
                p = spec.get("period") or _DEFAULT_INDICATOR_PERIODS.get(spec.get("name"), 20)
                periods.append(int(p))
    return max(periods) * 2 + _WARMUP_BUFFER_DAYS


def run_signal_backtest(
    conn,
    stock_codes,
    start_date: str,
    end_date: str,
    entry_rule: dict,
    exit_rule: dict,
    market: str = "KR",
    params: dict | None = None,
    price_history_fn: Callable | None = None,
    indicator_series_fn: Callable | None = None,
) -> dict:
    """개별종목 신호(entry/exit) 매매타이밍 백테스트를 실행한다.

    stock_codes(1~10개) 각각에 대해 entry_rule이 참이 되는 날 매수(현금→보유), exit_rule이
    참이 되는 날 매도(보유→현금)를 독립적으로 평가하고, 동시에 보유 중인 종목들에 균등가중
    배분한다(보유 집합이 바뀌는 날에만 리밸런싱·거래비용 차감). 신호는 t일 종가로 확정되나
    체결은 t+1일 종가 → 미래참조가 원천 차단된다.

    반환: {dates, navs, benchmark:None, performance, holdings:[{date, codes}]}
    (engine.run_backtest와 동일 형식 → auditor.post_audit이 무수정으로 소비 가능).

    price_history_fn/indicator_series_fn은 테스트 주입용(기본=KR/US 배치 가격조회 + TA-Lib
    전구간 지표). market='US'면 us_prices, 그 외 prices를 기본으로 쓴다.
    """
    codes = [str(c) for c in (stock_codes or [])]
    if not (1 <= len(codes) <= _MAX_SIGNAL_CODES):
        raise ValueError(
            f"stock_codes는 1~{_MAX_SIGNAL_CODES}개여야 합니다(받음: {len(codes)}개)"
        )
    if start_date >= end_date:
        raise ValueError("start_date는 end_date보다 앞서야 합니다")
    for rule, label in ((entry_rule, "entry_rule"), (exit_rule, "exit_rule")):
        if rule.get("op") not in _VALID_OPS:
            raise ValueError(f"{label}.op이 유효하지 않습니다: {rule.get('op')} ({sorted(_VALID_OPS)}만 가능)")

    if price_history_fn is None:
        if market == "US":
            from .data_access_us import us_price_history_batch as price_history_fn
        else:
            from .data_access import price_history_batch as price_history_fn
    if indicator_series_fn is None:
        from .primitives import compute_indicator_series as indicator_series_fn

    lookback = (datetime.date.fromisoformat(end_date) - datetime.date.fromisoformat(start_date)).days
    lookback += _max_lookback_days([entry_rule, exit_rule])
    history = price_history_fn(conn, codes, asof=end_date, lookback_days=lookback)

    # 종목별: 종가맵 + 보유여부맵(own) 계산.
    close_by_date: dict[str, dict[str, float]] = {}
    own_by_date: dict[str, dict[str, bool]] = {}
    for code in codes:
        series = history.get(code, []) or []
        dates_c = [p["date"] for p in series]
        closes_c = [p["close"] for p in series]
        close_by_date[code] = dict(zip(dates_c, closes_c))
        own_by_date[code] = _own_by_date_for_code(
            dates_c, closes_c, entry_rule, exit_rule, indicator_series_fn
        )

    # 창(window) 내 거래일 마스터 축(종목 간 합집합, 오름차순).
    master = sorted({
        d for code in codes for d in close_by_date[code]
        if start_date <= d <= end_date
    })
    if len(master) < 2:
        raise ValueError("선택 기간에 시뮬레이션할 거래일이 부족합니다(가격 시계열 범위 확인).")

    fee = (params or {}).get("fee_rate", 0.00015)
    tax = (params or {}).get("tax_rate", 0.0018)
    slip = (params or {}).get("slippage_rate", 0.0010)
    # engine.run_backtest와 동일한 1회 교체(매도+매수) 비용 모델 재사용(새 모델 금지).
    cost_per_turn = (fee + slip) + (fee + tax + slip)

    def _owned_set(i: int) -> set:
        d = master[i]
        return {c for c in codes if own_by_date[c].get(d) and d in close_by_date[c]}

    nav = 1.0
    navs = [1.0]
    holdings_log: list[dict] = []
    turnovers: list[float] = []
    prev_set = _owned_set(0)
    if prev_set:
        holdings_log.append({"date": master[0], "codes": sorted(prev_set)})

    for i in range(1, len(master)):
        cur_set = _owned_set(i)
        if cur_set != prev_set:  # 보유 집합이 바뀐 날 = 체결(리밸런싱) → 거래비용 차감
            turn = _turnover(prev_set, cur_set)
            turnovers.append(turn)
            nav *= (1 - turn * cost_per_turn)
            if cur_set:
                holdings_log.append({"date": master[i], "codes": sorted(cur_set)})
        # 그날 보유 종목의 균등가중 수익(close[i]/close[i-1]-1).
        rets = []
        for c in cur_set:
            p0 = close_by_date[c].get(master[i - 1])
            p1 = close_by_date[c].get(master[i])
            if p0 and p1 and p0 > 0:
                rets.append(p1 / p0 - 1)
        period_ret = sum(rets) / len(rets) if (cur_set and rets) else 0.0
        nav *= (1 + period_ret)
        navs.append(nav)
        prev_set = cur_set

    perf = performance(navs, _PERIODS_PER_YEAR_DAILY, benchmark=None)
    perf["avg_turnover"] = round(sum(turnovers) / len(turnovers) * 100, 1) if turnovers else 0.0

    return {
        "dates": master,
        "navs": navs,
        "benchmark": None,
        "performance": perf,
        "holdings": holdings_log,
    }


def _rank_metric(performance: dict | None, rank_by: str) -> float:
    """정렬 키. rank_by 지표가 없거나 숫자가 아니면 최하위(-inf)로 취급한다(정렬 안전)."""
    val = (performance or {}).get(rank_by)
    return float(val) if isinstance(val, (int, float)) else float("-inf")


def search_signal_strategy(
    conn,
    stock_codes,
    start_date: str,
    end_date: str,
    candidates: list[dict],
    market: str = "KR",
    constraints: list[dict] | None = None,
    rank_by: str = "sharpe",
    price_history_fn: Callable | None = None,
    indicator_series_fn: Callable | None = None,
) -> dict:
    """개별종목 시그널 전략을 여러 규칙 후보 중에서 탐색한다(시그널 버전의 역백테스트).

    "삼성전자로 MDD 30% 이내·누적수익률 35% 이상인 (기술지표) 전략을 찾아줘"처럼, 규칙이
    명시되지 않고 성과 목표(constraints)만 주어진 탐색형 질문을 위한 프리미티브다. 각 후보는
    {"entry_rule": SERIES-OP-SERIES, "exit_rule": ...} 조합이며, 후보마다 run_signal_backtest를
    그대로 호출해(롱온리·미래참조 차단·거래비용 반영을 자동 상속) 성과를 모은다.

    primitives.search_strategy(크로스섹셔널)와의 결정적 차이: search_strategy는 제약을 만족하는
    후보가 없으면 빈 리스트를 돌려주지만, 이 함수는 그때도 빈손으로 돌아오지 않고 rank_by 기준
    '가장 근접한 시도'를 constraints_met=False로 정직하게 반환한다(사용자가 성과 숫자를 볼 수 있게).

    Args:
        candidates: [{"entry_rule": {...}, "exit_rule": {...}}, ...]. 비어 있으면 ValueError,
            _MAX_SIGNAL_SEARCH_CANDIDATES(20) 초과 시 ValueError(자원 남용 방지).
        constraints: [{"metric": "mdd", "op": ">=", "value": -30.0}, ...] 전부 AND. metric은
            performance 키(total_return/cagr/mdd/sharpe 등, mdd는 음수), op는 >=/<=/>/</==/!=.
            연산자는 무거운 시뮬레이션 '전에' 사전검증한다(오타 하나로 전 후보를 돌린 뒤 실패 방지).
        rank_by: 결과 정렬 기준 performance 키(기본 "sharpe"). 내림차순.

    Returns:
        {"constraints_met": bool,   # 후보 중 하나라도 제약을 전부 만족했으면 True
         "results": [{entry_rule, exit_rule, performance, holdings, dates, navs,
                      constraints_met}, ...],  # 제약충족 우선 + rank_by 내림차순 정렬
         "best": results[0] | None}  # 가장 근접/최선의 시도(편의 필드)

    한 후보가 예외를 던지면 그 후보만 조용히 건너뛰고 나머지를 계속 진행한다. 단 후보 전부가
    실패하면 마지막 예외 사유를 포함한 ValueError를 던진다(디버깅 가능하도록).
    """
    if not candidates:
        raise ValueError("candidates가 비어 있습니다(탐색할 후보 전략이 없습니다)")
    if len(candidates) > _MAX_SIGNAL_SEARCH_CANDIDATES:
        raise ValueError(
            f"후보 전략 {len(candidates)}개 > 상한 {_MAX_SIGNAL_SEARCH_CANDIDATES}개(자원 남용 방지)"
        )

    # 제약조건 연산자를 무거운 시뮬레이션 '전'에 미리 검증한다(search_strategy와 동일 원칙).
    # 순환 import 회피: primitives의 제약 어휘(_CONSTRAINT_OPS/_satisfies_constraints)를
    # 함수 내부에서 지연 import해 재사용한다(run_signal_backtest가 compute_indicator_series를
    # 지연 import하는 것과 동일한 관례).
    from .primitives import _CONSTRAINT_OPS, _satisfies_constraints

    for c in constraints or []:
        if c.get("op") not in _CONSTRAINT_OPS:
            raise ValueError(
                f"지원하지 않는 연산자: {c.get('op')} ({sorted(_CONSTRAINT_OPS)}만 가능)"
            )

    results: list[dict] = []
    last_error: Exception | None = None
    for cand in candidates:
        try:
            res = run_signal_backtest(
                conn, stock_codes, start_date, end_date,
                entry_rule=cand["entry_rule"], exit_rule=cand["exit_rule"],
                market=market,
                price_history_fn=price_history_fn,
                indicator_series_fn=indicator_series_fn,
            )
        except Exception as exc:  # noqa: BLE001 — 후보 하나 실패는 건너뛰고 계속 진행
            last_error = exc
            continue
        perf = res.get("performance")
        met = _satisfies_constraints(perf, constraints) if constraints else True
        results.append({
            "entry_rule": cand["entry_rule"],
            "exit_rule": cand["exit_rule"],
            "performance": perf,
            "holdings": res.get("holdings"),
            "dates": res.get("dates"),
            "navs": res.get("navs"),
            "constraints_met": met,
        })

    if not results:
        raise ValueError(
            f"모든 후보 백테스트가 실패했습니다(마지막 오류: {last_error})"
        )

    # 제약충족 후보를 앞으로, 그 안에서 rank_by 내림차순. (True > False, 큰 값 우선 → reverse=True)
    results.sort(
        key=lambda r: (r["constraints_met"], _rank_metric(r["performance"], rank_by)),
        reverse=True,
    )
    any_met = any(r["constraints_met"] for r in results)
    return {"constraints_met": any_met, "results": results, "best": results[0]}
