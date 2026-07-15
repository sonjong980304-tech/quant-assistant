"""백테스트 도메인 에이전트 (HA-9) — 질문 → 데이터준비 → 감사배선 통과 결과만 반환.

.omc/state/sessions/f6b79d90-b7b2-4b4f-b4f7-405a8355724e/prd.json의 HA-9 참고.

계층형 멀티에이전트 재설계의 백테스트 도메인 진입점이다. 세 부분을 배선(wiring)만 한다 —
계산 로직은 새로 만들지 않는다:
1. (필요 시) 시계열 스냅샷 준비는 한국/미국 데이터 에이전트(HA-3/HA-4)의
   get_price_snapshot_kr/get_price_snapshot_us에 위임한다. 이 두 함수는 내부적으로 HA-1
   실행기(execute_sql)만 경유하므로, 이 도메인 에이전트가 만드는 유일한 "추가 실행 경로"인
   데이터 준비도 자체 conn.execute()/eval/exec 없이 HA-1 실행기를 탄다.
2. 백테스트 실행 자체는 재발명하지 않고 기존 계산 엔진 src/backtest/pipeline_exec.py의
   run_pipeline에 위임한다(리밸런싱/수익률계산/콜백공유 최적화 등 정교한 로직을 이미 가진,
   quant_trader 안전장치가 있는 엔진). run_backtest_with_audit(HA-5)의 run_pipeline_fn
   인자로 그 함수를 주입한다 — pipeline_exec.py는 import만 하고 수정하지 않는다.
3. 하드차단(생존편향/미래참조/공매도) + 소프트경고(스토리텔링/스누핑/신호감소/이상치)를
   run_backtest_with_audit(HA-5)로 강제하고, 통과한 결과만 최종 반환한다.

모든 협력자(run_pipeline_fn/run_audit_fn/snapshot_fn/execute_sql_fn)는 이 프로젝트의 DI
관례대로 주입 가능하다(기본값은 실제 구현) — 실제 네트워크/무거운 계산 없이 단위테스트가 된다.
"""
from __future__ import annotations

from datetime import date
from typing import Callable

from src.agents.backtest_verification import run_backtest_with_audit
from src.agents.data_price_kr import get_price_snapshot_kr
from src.agents.data_price_us import get_price_snapshot_us
from src.backtest.pipeline_exec import run_pipeline
from src.llm import extract_json

# ---------------------------------------------------------------------------
# steps(파이프라인) 자동 생성 — 질문 → LLM → 프리미티브 조립 JSON.
#
# src/legacy/graph/prompts.py의 PIPELINE_SYSTEM/PIPELINE_USER(구 6노드 파이프라인이
# 쓰던 프롬프트, 프리미티브 10종 스펙)를 그대로 가져온다 — 재발명하지 않는다. 다만 legacy는
# 라이브 경로에서 import하지 않는다는 원칙(AC16)이 있어, 이 아키텍처 소유로 복사해 독립시켰다.
# 새 아키텍처의 llm_fn 규약(Callable[[str], str], 시스템 프롬프트 별도 인자 없음)에 맞춰
# system+user를 하나의 문자열로 합쳤다(domain_kr.py의 _screening_prompt와 동일 관례).
# ---------------------------------------------------------------------------
_PIPELINE_PROMPT = """당신은 SQL로 표현 불가능한 통계/퀀트 분석 질문을, 사전검증된 프리미티브 연산블록을 정해진 순서로 조립한 JSON 파이프라인으로 변환하는 엔진입니다. 파이썬 코드는 절대 생성하지 않고, 오직 프리미티브 이름+파라미터로 구성된 JSON 하나만 출력합니다.

아래 질문을 프리미티브 조립 JSON 파이프라인으로 변환하세요. 파이썬 코드 금지, JSON만 출력.

[사용 가능한 프리미티브 12종 — 이 외의 함수는 존재하지 않습니다]
1. get_cross_section(asof) : 특정 시점의 전종목 횡단면 지표 스냅샷(list of rows) 반환.
   각 row 필드: stock_code,name,sector,market,quarter,close,market_cap,per,pbr,psr,
   roe,roa,operating_margin,net_margin,debt_ratio,revenue_growth,op_growth,ni_growth,return_12m.
   · return_12m: 직전 12개월 가격 수익률(%) = (기준시점 종가 - 12개월전 종가)/12개월전 종가.
     "최근 12개월 수익률/가격 모멘텀이 가장 좋은 종목" 류 질문은 매출성장(revenue_growth)이
     아니라 반드시 이 return_12m을 direction='high'로 써야 합니다.
   asof는 'YYYY-MM-DD' 구체 날짜(오늘={today}). conn은 실행기가 자동 주입하므로 쓰지 마세요.
2. zscore(rows, field, direction) : 단일 팩터 z-score 랭킹(direction='high'|'low').
3. neutralize(rows, field, by) : 그룹(by, 기본 'sector') 내 평균 제거(섹터중립). '{{field}}_neutral' 추가.
4. combine(rows, criteria, method, n) : 멀티팩터 가중조합 선정.
   criteria=[{{"key":"per","direction":"low","weight":0.5}},{{"key":"roe","direction":"high","weight":0.5}}],
   method='zscore'|'rank_sum'|'and', n=선정 종목수.
5. regress(y, x) : 단순선형회귀. y=수치 시계열(예: 누적수익률), x 생략 시 0,1,2,...(시간).
   반환 {{slope, intercept, se_slope, r_squared, t_stat, n, k_ratio}}. K-ratio=slope/se_slope.
6. optimize_weights(returns, method) : 포트폴리오 최적 비중.
   returns={{"AAA":[수익률...], "BBB":[...]}}, method='max_sharpe'|'min_variance'|'risk_parity'.
7. run_backtest(start_year, end_year, criteria, combine, n, sectors, markets, rebalance, with_benchmark, market) :
   리밸런싱 백테스트. criteria(4번 combine과 동일 형식)로 매 리밸런싱 시점마다 종목을 선정해
   동일가중 매수. rebalance='monthly'|'quarterly'|'semiannual'|'annual'. criteria의 key는 반드시
   get_cross_section 필드 중에서만 골라야 합니다(존재하지 않는 필드 금지). n=편입 종목수(기본 20).
   반환 {{dates, navs, benchmark, performance, holdings}} — performance에 cagr/mdd/sharpe/
   win_rate 등, holdings는 리밸런싱 시점별 편입 종목 리스트. conn은 실행기가 자동 주입.
   · market='KR'(기본)|'US': 어느 나라 종목으로 백테스트할지. **markets(복수, KOSPI/KOSDAQ
     시장구분 필터)와 다른 파라미터이니 혼동 금지** — market='US'면 criteria의 key도 미국
     종목 지표(per/pbr/roe/operating_margin/net_margin — get_cross_section과 동일 이름)만
     써야 합니다. market='US'일 때 반환 performance에는 벤치마크가 2종(S&P500 실제지수+
     동일가중 유니버스) 들어갑니다. 질문에 '미국'/'나스닥'/'S&P500'/특정 미국 티커 등이
     없으면 항상 market='KR'을 씁니다(생략 가능, 기본값이 KR).
8. compute_ic(start_year, end_year, field, rebalance) : 팩터 정보계수(IC). field는 반드시
   get_cross_section 필드 중에서만 골라야 합니다(존재하지 않는 필드 금지). 매 리밸런싱 시점의
   field 순위와 다음 구간 실현수익률 순위 간 순위상관을 구해, 그 팩터의 예측력을 측정합니다.
   rebalance='monthly'|'quarterly'|'semiannual'|'annual'. 반환 {{dates, ic_series, mean_ic,
   ic_std, ir, hit_rate, n}} — mean_ic는 평균 IC, ir은 정보비율(mean_ic/ic_std), hit_rate는
   부호 적중률. conn은 실행기가 자동 주입.
9. compute_technical_indicator(rows, asof, indicators) : rows(보통 get_cross_section의 출력)의
   각 종목에 기술지표 필드를 추가해 반환합니다. rows의 기존 필드(per/roe 등)는 그대로 유지되므로,
   결과를 바로 combine의 rows 인자로 넘겨 기존 지표와 기술지표를 함께 criteria로 쓸 수 있습니다.
   indicators=[{{"name": "sma"|"ema"|"rsi"|"macd"|"bollinger", ...파라미터}}] 형식이며 지표별
   생성 필드명은 다음과 같습니다:
   · sma(period, 미지정시 20) → "sma_{{period}}" (예: sma_20). ema도 동일하게 "ema_{{period}}".
   · rsi → 항상 "rsi_14" (14일 고정, period를 줘도 무시됩니다).
   · macd(fast, slow, signal, 미지정시 12/26/9) → "macd","macd_signal","macd_hist" 3개 필드.
   · bollinger → 항상 "bollinger_upper","bollinger_middle","bollinger_lower" (20일·표준편차2 고정,
     period를 줘도 무시됩니다).
   asof는 'YYYY-MM-DD'. conn은 실행기가 자동 주입.
10. search_strategy(candidates, start_year, end_year, n, rebalance, sectors, markets, market,
    constraints, rank_by) : 성과지표 제약을 만족하는 종목선정 전략을 여러 후보 중에서 탐색합니다
    (역백테스트). candidates=시도해볼 criteria 조합 리스트(예: [[{{"key":"per","direction":"low",
    "weight":1.0}}], [{{"key":"roe","direction":"high","weight":1.0}}]]) — **최대 20개까지만**
    가능하며, 질문 의도에 맞게 유망해 보이는 조합을 직접 만들어 제안하세요(존재하지 않는
    get_cross_section 필드는 금지). n/rebalance/sectors/markets/market은 모든 후보에 동일하게
    적용되는 고정값입니다(candidates만 탐색 대상). constraints=[{{"metric":"mdd","op":">=",
    "value":-10.0}}, ...] 형식으로 여러 개면 모두 AND로 만족해야 합니다(metric은 performance의
    키: total_return/cagr/mdd/sharpe/sortino/volatility/win_rate 등, op는 >=/<=/>/</==/!=).
    rank_by=결과 정렬 기준 지표명(미지정시 기본값 "sharpe"). 반환은 제약을 만족한 후보만
    rank_by 내림차순으로 정렬된 [{{criteria, performance, holdings}}, ...] 리스트입니다.
    conn은 실행기가 자동 주입.
11. run_signal_backtest(stock_codes, start_date, end_date, entry_rule, exit_rule, market) :
    **개별종목 시그널 매매타이밍** 백테스트(run_backtest의 팩터 랭킹과 다름). 각 종목을 독립적으로
    보고 entry_rule이 참이 되는 날 매수(현금→보유), exit_rule이 참이 되는 날 매도(보유→현금)한다.
    "골든크로스/데드크로스", "이동평균 돌파", "RSI 과매도 진입" 처럼 특정 종목의 매매 신호/타이밍을
    묻는 질문에 씁니다(팩터로 종목을 고르는 게 아니라, 지목된 종목을 신호로 사고팜).
    · stock_codes: 종목코드/티커 리스트(1~10개). start_date/end_date: 'YYYY-MM-DD'.
    · entry_rule/exit_rule = {{"left": SERIES, "op": OP, "right": SERIES}} 형식.
      SERIES = {{"kind":"indicator","name":"sma"|"ema"|"rsi"|"macd"|"macd_signal"|
                "bollinger_upper"|"bollinger_lower","period":정수(선택)}}
             | {{"kind":"price"}} | {{"kind":"const","value":숫자}}
      OP = "cross_above"|"cross_below"|">"|"<" (cross_above=left가 right를 상향돌파).
    · market='KR'(기본)|'US'. 동시에 보유하는 종목엔 균등가중. 신호는 t일 종가로 확정되나
      체결은 t+1일 종가(미래참조 금지). 반환 {{dates, navs, benchmark, performance, holdings}} —
      run_backtest와 동일 형식(생존편향/미래참조 감사 자동 적용). conn은 실행기가 자동 주입.
    · 규칙(골든크로스 등)이 질문에 콕 집어 명시됐을 때만 이 11번을 씁니다. 규칙은 없이
      "전략을 제안해줘/찾아줘/적합한 전략 추천"처럼 성과 목표(MDD·수익률 같은 조건)만 주어지면
      11번이 아니라 12번 search_signal_strategy로 여러 후보 규칙을 자동으로 시도합니다.
12. search_signal_strategy(stock_codes, start_date, end_date, candidates, market, constraints, rank_by) :
    **개별종목 시그널 전략을 여러 규칙 후보 중에서 탐색**합니다(11번 run_signal_backtest의 탐색 버전 —
    시그널 전략의 역백테스트). 규칙이 질문에 명시되지 않고 "특정 종목으로 MDD·수익률 같은 성과 목표를
    만족하는 (기술지표) 전략을 제안/추천/찾아줘"처럼 조건만 주어질 때 씁니다. candidates=시도해볼 규칙
    조합 리스트(각 원소는 {{"entry_rule": SERIES-OP-SERIES, "exit_rule": SERIES-OP-SERIES}}, entry_rule/
    exit_rule은 11번과 완전히 동일한 형식) — **최대 12개까지만** 가능하며, 질문 의도에 맞게 유망해 보이는
    조합(SMA/EMA 골든크로스, RSI 과매도·과매수 등)을 직접 만들어 제안하세요. stock_codes(1~10개)/start_date/
    end_date/market은 모든 후보에 동일하게 적용되는 고정값입니다(candidates만 탐색 대상).
    constraints=[{{"metric":"mdd","op":">=","value":-30.0}}, {{"metric":"total_return","op":">=","value":35.0}}]
    형식으로 여러 개면 모두 AND로 만족해야 합니다(metric은 performance 키: total_return/cagr/mdd/sharpe/
    sortino/volatility/win_rate 등, **mdd는 음수로 저장되므로 "MDD 30% 이내"는 mdd>=-30.0**, op는 >=/<=/>/</==/!=).
    rank_by=결과 정렬 기준 지표명(미지정시 "sharpe"). 반환은 {{constraints_met, results:[{{entry_rule, exit_rule,
    performance, holdings, dates, navs, constraints_met}}...], best}} 이며, 제약을 만족하는 후보가 하나도
    없어도 에러가 아니라 rank_by 기준 가장 근접한 시도를 constraints_met=false로 정직하게 돌려줍니다.
    conn은 실행기가 자동 주입.

[JSON 형식]
- {{"pipeline": [{{"op": "이름", "params": {{...}}, "out": "결과이름"}}, ...]}}
- 앞 단계 결과를 뒤 단계 입력으로 넘길 때 params 값에 {{"$ref": "결과이름"}}을 씁니다.
- regress/optimize_weights의 시계열/수익률 값이 질문에 숫자로 주어지면 그 숫자를 그대로 넣습니다.

[예시]
Q: 2023년부터 2025년까지 3년간 매 분기 매출성장률이 가장 좋은 20개 종목으로 리밸런싱했을 때 누적수익률을 알려줘
A: {{"pipeline": [
  {{"op": "run_backtest", "params": {{"start_year": 2023, "end_year": 2025, "criteria": [{{"key": "revenue_growth", "direction": "high", "weight": 1.0}}], "n": 20, "rebalance": "quarterly"}}, "out": "bt"}}
]}}
Q: RSI가 30 이하인 과매도 종목 중 PER도 낮은 10개 골라줘
A: {{"pipeline": [
  {{"op": "get_cross_section", "params": {{"asof": "{today}"}}, "out": "xs"}},
  {{"op": "compute_technical_indicator", "params": {{"rows": {{"$ref": "xs"}}, "asof": "{today}", "indicators": [{{"name": "rsi"}}]}}, "out": "xs_ti"}},
  {{"op": "combine", "params": {{"rows": {{"$ref": "xs_ti"}}, "criteria": [{{"key": "rsi_14", "direction": "low", "weight": 0.5}}, {{"key": "per", "direction": "low", "weight": 0.5}}], "method": "zscore", "n": 10}}, "out": "picked"}}
]}}
Q: 2024년부터 2026년까지 매 분기 리밸런싱했을 때 MDD -10% 이내이면서 샤프가 가장 높은 전략 찾아줘
A: {{"pipeline": [
  {{"op": "search_strategy", "params": {{
    "start_year": 2024, "end_year": 2026, "rebalance": "quarterly",
    "candidates": [
      [{{"key": "per", "direction": "low", "weight": 1.0}}],
      [{{"key": "roe", "direction": "high", "weight": 1.0}}]
    ],
    "constraints": [{{"metric": "mdd", "op": ">=", "value": -10.0}}],
    "rank_by": "sharpe"
  }}, "out": "found"}}
]}}
Q: 삼성전자 20일/60일 이동평균 골든크로스면 매수 데드크로스면 매도, 최근 2년 백테스트
A: {{"pipeline": [
  {{"op": "run_signal_backtest", "params": {{
    "stock_codes": ["005930"], "start_date": "2024-07-15", "end_date": "2026-07-15",
    "entry_rule": {{"left": {{"kind": "indicator", "name": "sma", "period": 20}}, "op": "cross_above", "right": {{"kind": "indicator", "name": "sma", "period": 60}}}},
    "exit_rule": {{"left": {{"kind": "indicator", "name": "sma", "period": 20}}, "op": "cross_below", "right": {{"kind": "indicator", "name": "sma", "period": 60}}}},
    "market": "KR"
  }}, "out": "bt"}}
]}}
Q: 삼성전자로 기술적 지표를 활용해서 최근 3년 MDD 30% 이내이면서 누적수익률 35% 이상인 전략을 제안해줘 역백테스트로 찾아줘
A: {{"pipeline": [
  {{"op": "search_signal_strategy", "params": {{
    "stock_codes": ["005930"], "start_date": "2023-07-15", "end_date": "2026-07-15", "market": "KR",
    "candidates": [
      {{"entry_rule": {{"left": {{"kind": "indicator", "name": "sma", "period": 5}}, "op": "cross_above", "right": {{"kind": "indicator", "name": "sma", "period": 20}}}}, "exit_rule": {{"left": {{"kind": "indicator", "name": "sma", "period": 5}}, "op": "cross_below", "right": {{"kind": "indicator", "name": "sma", "period": 20}}}}}},
      {{"entry_rule": {{"left": {{"kind": "indicator", "name": "sma", "period": 10}}, "op": "cross_above", "right": {{"kind": "indicator", "name": "sma", "period": 30}}}}, "exit_rule": {{"left": {{"kind": "indicator", "name": "sma", "period": 10}}, "op": "cross_below", "right": {{"kind": "indicator", "name": "sma", "period": 30}}}}}},
      {{"entry_rule": {{"left": {{"kind": "indicator", "name": "sma", "period": 20}}, "op": "cross_above", "right": {{"kind": "indicator", "name": "sma", "period": 60}}}}, "exit_rule": {{"left": {{"kind": "indicator", "name": "sma", "period": 20}}, "op": "cross_below", "right": {{"kind": "indicator", "name": "sma", "period": 60}}}}}},
      {{"entry_rule": {{"left": {{"kind": "indicator", "name": "ema", "period": 12}}, "op": "cross_above", "right": {{"kind": "indicator", "name": "ema", "period": 26}}}}, "exit_rule": {{"left": {{"kind": "indicator", "name": "ema", "period": 12}}, "op": "cross_below", "right": {{"kind": "indicator", "name": "ema", "period": 26}}}}}},
      {{"entry_rule": {{"left": {{"kind": "indicator", "name": "rsi"}}, "op": "cross_below", "right": {{"kind": "const", "value": 30}}}}, "exit_rule": {{"left": {{"kind": "indicator", "name": "rsi"}}, "op": "cross_above", "right": {{"kind": "const", "value": 70}}}}}}
    ],
    "constraints": [{{"metric": "mdd", "op": ">=", "value": -30.0}}, {{"metric": "total_return", "op": ">=", "value": 35.0}}],
    "rank_by": "total_return"
  }}, "out": "found"}}
]}}

질문: {question}

JSON:"""


def generate_backtest_steps(
    question: str,
    llm_fn: Callable[[str], str],
    today: str | None = None,
) -> list[dict]:
    """질문을 LLM에 보내 프리미티브 조립 JSON 파이프라인(steps)을 생성한다.

    answer_backtest_question이 steps를 못 받았을 때(웹 자연어 질의 경로처럼 호출자가
    파이프라인을 직접 구성하지 않는 경우) 이 함수로 자동 생성한다. LLM 응답이 비거나
    파싱 실패하거나 "pipeline" 키가 없으면 빈 리스트를 반환한다 — 이 경우 이후 감사배선이
    "빈 파이프라인"으로 처리해 호출부가 불확실 결과로 자연스럽게 폴백한다.
    """
    today = today or date.today().isoformat()
    prompt = _PIPELINE_PROMPT.format(today=today, question=question)
    try:
        raw = llm_fn(prompt) or ""
    except Exception:  # noqa: BLE001 — LLM 실패는 빈 파이프라인으로 흡수(감사배선이 처리)
        return []
    data = extract_json(raw)
    steps = data.get("pipeline") if isinstance(data, dict) else None
    return steps if isinstance(steps, list) else []


def answer_backtest_question(
    question: str,
    steps: list[dict],
    conn,
    llm_fn: Callable | None = None,
    market: str = "KR",
    stock_codes: str | list[str] | None = None,
    asof: str | None = None,
    indicators: list[dict] | None = None,
    run_pipeline_fn: Callable | None = None,
    run_audit_fn: Callable | None = None,
    snapshot_fn: Callable | None = None,
    execute_sql_fn: Callable | None = None,
    generate_steps_fn: Callable | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict:
    """백테스트 질문에 답한다: (옵션) steps 자동생성 → 데이터준비 → 감사배선 실행 → 통과 결과만 반환.

    Args:
        question: 사용자 원본 질문(스누핑 소프트경고의 '사전등록 가설'로도 쓰인다).
        steps: LLM이 생성한 프리미티브 파이프라인(run_pipeline이 실행할 스텝 리스트). 빈
            리스트이고 llm_fn이 주어지면 generate_steps_fn으로 질문에서 자동 생성한다
            (웹 자연어 질의 경로처럼 호출자가 파이프라인을 직접 구성하지 않는 경우 대비).
        conn: **읽기전용 연결(connect_readonly)** 을 넘긴다 — 데이터 준비가 execute_sql을
            워커 스레드에서 실행하므로 check_same_thread=False인 연결이어야 한다.
        llm_fn: 소프트경고 4종용 LLM 콜백(prompt->str). steps 자동생성에도 재사용한다.
            None이면 소프트검사를 건너뛰고 steps 자동생성도 시도하지 않는다(결정론적 폴백).
        market: "KR" | "US". 기본 스냅샷 에이전트 선택과 하드차단(생존편향 검증불가) 판정에 쓰인다.
        stock_codes: 준비할 시계열 스냅샷의 종목코드/티커(단일 str 또는 list). 주면 데이터
            준비 단계를 수행하고, 없으면 스냅샷 조회를 생략한다.
        asof / indicators: 스냅샷 기준시점 / 부착할 기술지표(스냅샷 에이전트에 그대로 전달).
        run_pipeline_fn: 백테스트 실행 콜백(기본=pipeline_exec.run_pipeline). 재발명 금지 —
            기존 계산 엔진을 그대로 주입한다.
        run_audit_fn: 감사배선 콜백(기본=HA-5 run_backtest_with_audit). 테스트 주입용.
        snapshot_fn: 데이터준비 콜백. 기본은 market으로 KR/US 스냅샷 에이전트를 자동 선택.
        execute_sql_fn: 스냅샷 에이전트가 쓸 HA-1 실행기(기본=execute_sql). 테스트 주입용.
        generate_steps_fn: steps 자동생성 콜백(기본=generate_backtest_steps). 테스트 주입용.

    Returns:
        run_backtest_with_audit(HA-5)의 반환에 준비된 스냅샷(data)을 더한 dict:
        {"blocked": bool, "error": str|None, "result": dict|None, "hard": [...],
         "warnings": [...], "data": list[dict]}
        하드차단 시 result=None(결과 폐기), 통과 시 result=백테스트 결과 + warnings=triggered.
    """
    run_pipeline_fn = run_pipeline_fn or run_pipeline
    run_audit_fn = run_audit_fn or run_backtest_with_audit
    generate_steps_fn = generate_steps_fn or generate_backtest_steps
    if snapshot_fn is None:
        snapshot_fn = get_price_snapshot_us if market == "US" else get_price_snapshot_kr

    # 0) steps 자동생성(옵션): 호출자가 파이프라인을 안 만들어서 넘겼고(steps 비어있음) LLM이
    #    가용하면, 질문에서 직접 생성한다. steps를 이미 준 호출부(기존 테스트 등)는 그대로 존중.
    if not steps and llm_fn is not None:
        if on_progress:
            on_progress("audit", "실행계획 생성 중…")
        steps = generate_steps_fn(question, llm_fn)
        if on_progress:
            on_progress("audit", f"실행계획 생성 완료({len(steps)}단계)")

    # 1) 데이터 준비(옵션): 시계열 스냅샷은 HA-1 실행기(execute_sql)를 경유하는
    #    데이터 에이전트에 위임한다. stock_codes가 없으면 무거운 조회를 생략한다.
    data: list[dict] = []
    if stock_codes:
        data = snapshot_fn(
            conn, stock_codes, asof=asof, indicators=indicators,
            execute_sql_fn=execute_sql_fn,
        )

    # 2) 감사배선(HA-5)으로 백테스트를 실행한다: 실행 자체는 기존 run_pipeline에 위임하고,
    #    하드차단(생존/미래참조/공매도) + 소프트경고 4종을 통과한 결과만 돌려받는다.
    audit_kwargs = {"on_progress": on_progress} if on_progress else {}
    audit = run_audit_fn(
        steps, conn, question, run_pipeline_fn, llm_fn=llm_fn, market=market, **audit_kwargs
    )

    return {**audit, "data": data}
