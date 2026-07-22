"""백테스트 도메인 에이전트 (HA-9) — 질문 → 데이터준비 → 감사배선 통과 결과만 반환.

.omc/state/sessions/f6b79d90-b7b2-4b4f-b4f7-405a8355724e/prd.json의 HA-9 참고.

계층형 멀티에이전트 재설계의 백테스트 도메인 진입점이다. 세 부분을 배선(wiring)만 한다 —
계산 로직은 새로 만들지 않는다:
1. (필요 시) 시계열 스냅샷 준비는 한국 데이터 에이전트(HA-3)의
   get_price_snapshot_kr에 위임한다. 이 함수는 내부적으로 HA-1
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
from src.backtest.data_access import mode_financial_quarter_at
from src.backtest.pipeline_exec import PRIMITIVE_OPS, run_pipeline
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

[사용 가능한 프리미티브 26종 — 이 외의 함수는 존재하지 않습니다]
1. get_cross_section(asof, markets) : 특정 시점의 전종목 횡단면 지표 스냅샷(list of rows) 반환.
   각 row 필드: stock_code,name,sector,market,quarter,close,market_cap,per,pbr,psr,
   roe,roa,operating_margin,net_margin,debt_ratio,revenue_growth,op_growth,ni_growth,return_12m,
   gross_profit,total_assets,gp_a,cfo_ratio,earnings_yield,roc,roc_estimated.
   · cfo_ratio: 영업활동현금흐름(TTM) ÷ 총자산(%) — CFO비율(질/수익성 팩터, 높을수록 우수).
   · return_12m: 직전 12개월 가격 수익률(%) = (기준시점 종가 - 12개월전 종가)/12개월전 종가.
     "최근 12개월 수익률/가격 모멘텀이 가장 좋은 종목" 류 질문은 매출성장(revenue_growth)이
     아니라 반드시 이 return_12m을 direction='high'로 써야 합니다.
   · gp_a: 매출총이익(TTM) ÷ 총자산(%) — GPA 수익성 팩터. "GPA"/"매출총이익/자산" 질문에 씁니다.
   · earnings_yield: 이익수익률(EY, %) = 마법공식 EBIT ÷ 기업가치EV. "이익수익률"/"마법공식"
     밸류 팩터(높을수록 저평가). roc: 투하자본수익률(ROC, %) = EBIT ÷ 투하자본IC.
     "투하자본수익률"/"마법공식" 수익성 팩터(높을수록 우수). 마법공식 산점도는 이 둘을 씁니다.
   · roc_estimated: bool|null. 감가상각비 데이터가 없는 종목(삼성전자 등 다수 대형주 —
     DART 표준 API에 계정 자체가 없음)은 감가상각비를 0으로 근사해 roc를 계산하고
     이때 true가 됩니다(실측이면 false, roc 자체가 null이면 null). "삼성전자 ROC 알려줘"
     류 질문에 답할 때 이 값이 true면 반드시 "감가상각비 데이터가 없어 근사치"라고
     답변에 명시하세요 — 실측치처럼 단정하면 안 됩니다.
   asof는 'YYYY-MM-DD' 구체 날짜(오늘={today}). 질문에 "{{YYYY-MM-DD}}"나 "(예: ...)"처럼
   채워지지 않은 자리표시자(placeholder)가 그대로 적혀 있어도 그 문자열을 절대 그대로
   복사해 넣지 마세요 — 반드시 위에 안내한 오늘 날짜({today})를 실제 값으로 쓰세요.
   conn은 실행기가 자동 주입하므로 쓰지 마세요.
   · markets(선택, 예: ["KOSPI"]) : 시장 필터. "코스피 전종목"/"코스닥 전종목"처럼 특정
     시장으로 한정하는 질문은 이 단계에서 markets로 걸러야 합니다. **run_backtest의 markets
     파라미터와 이름·의미가 동일하지만 별개 함수의 파라미터입니다 — get_cross_section에는
     이 markets 외에 다른 파라미터(예: market 단수형)가 없으니 절대 존재하지 않는 파라미터를
     만들어 넣지 마세요.**

[공통 규칙 — start_year/end_year 기본값] 7번 run_backtest, 8번 compute_ic, 10번
search_strategy, 21번 run_qvm_backtest처럼 start_year/end_year를 쓰는 모든 연산에 공통
적용됩니다: 질문에 종료 시점(연도)이 명시돼 있지 않으면 end_year는 항상 오늘 날짜
({today})가 속한 연도를 쓰세요. 아래 [예시] 섹션 JSON들에 나오는 연도(2023, 2024, 2025,
2026 등)는 각 예시 질문에 실제로 적힌 기간을 그대로 옮긴 것일 뿐 고정 규칙이 아니므로,
질문에 종료연도가 없는데 예시의 숫자를 그대로 베끼면 안 됩니다 — asof와 마찬가지로 오늘
날짜를 기준으로 실제 값을 계산해 쓰세요.

2. zscore(rows, field, direction) : 단일 팩터 z-score 랭킹(direction='high'|'low').
3. neutralize(rows, field, by, method) : 그룹(by, 기본 'sector') 내 중립화. '{{field}}_neutral' 추가.
   · method='demean'(기본) : 그룹 평균만 뺀다(그룹 간 수준차만 제거, 변동성 차이는 남음).
   · method='zscore' : 그룹 평균을 빼고 그룹 표준편차로 나눠 정규화한다(그룹 간 변동성
     차이까지 보정). "섹터 중립 포트폴리오"/"섹터별 z-score"처럼 섹터 내 표준화된 순위로
     전 종목을 한 번에 랭킹해야 하는 질문은 method='zscore'로 중립화한 뒤 그 결과를
     zscore(rows, field='{{field}}_neutral', direction='high') 단계로 이어서 전체 랭킹하세요
     (neutralize 자체는 랭킹하지 않고 정규화된 값만 부여합니다).
   · by=null(섹터 등 그룹 구분 없이 "전체 시장 기준") : "PER 전체 시장 기준 z-score"처럼
     섹터 구분 없이 전 종목을 하나의 모집단으로 표준화하고 싶을 때는 by를 아예 생략하지
     말고 명시적으로 null을 넣으세요(by를 생략하면 기본값인 'sector'로 그룹이 나뉘어
     섹터별로 따로 표준화되므로 의도와 달라집니다). method='zscore'와 함께 쓰면
     '{{field}}_neutral'에 전체 시장 기준 z-score 값이 담기고, 그 값을 histogram_buckets/
     scatter_data/quantile_bucket_means 등 field 파라미터를 받는 어떤 후속 연산에도 그대로
     넘길 수 있습니다(field 이름만 바꾸면 pbr/roe 등 다른 지표에도 동일하게 적용).
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
   end_year 기본값(질문에 종료연도 미명시 시)은 위 공통 규칙 참고.
   반환 {{dates, navs, benchmark, performance, holdings}} — performance에 cagr/mdd/sharpe/
   win_rate 등, holdings는 리밸런싱 시점별 편입 종목 리스트. conn은 실행기가 자동 주입.
   국내(KOSPI/KOSDAQ) 종목만 대상으로 하며 벤치마크는 실제 코스피 지수입니다.
8. compute_ic(start_year, end_year, field, rebalance) : 팩터 정보계수(IC). field는 반드시
   get_cross_section 필드 중에서만 골라야 합니다(존재하지 않는 필드 금지). 매 리밸런싱 시점의
   field 순위와 다음 구간 실현수익률 순위 간 순위상관을 구해, 그 팩터의 예측력을 측정합니다.
   rebalance='monthly'|'quarterly'|'semiannual'|'annual'. 반환 {{dates, ic_series, mean_ic,
   ic_std, ir, hit_rate, n}} — mean_ic는 평균 IC, ir은 정보비율(mean_ic/ic_std), hit_rate는
   부호 적중률. conn은 실행기가 자동 주입. end_year 기본값은 위 공통 규칙 참고.
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
    적용되는 고정값입니다(candidates만 탐색 대상). end_year 기본값은 위 공통 규칙 참고.
    constraints=[{{"metric":"mdd","op":">=",
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
    · 동시에 보유하는 종목엔 균등가중. 신호는 t일 종가로 확정되나
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
    exit_rule은 11번과 완전히 동일한 형식) — **최대 20개까지만** 가능하며, 질문 의도에 맞게 유망해 보이는
    조합(SMA/EMA 골든크로스, RSI 과매도·과매수 등)을 직접 만들어 제안하세요. stock_codes(1~10개)/start_date/
    end_date/market은 모든 후보에 동일하게 적용되는 고정값입니다(candidates만 탐색 대상).
    constraints=[{{"metric":"mdd","op":">=","value":-30.0}}, {{"metric":"total_return","op":">=","value":35.0}}]
    형식으로 여러 개면 모두 AND로 만족해야 합니다(metric은 performance 키: total_return/cagr/mdd/sharpe/
    sortino/volatility/win_rate 등, **mdd는 음수로 저장되므로 "MDD 30% 이내"는 mdd>=-30.0**, op는 >=/<=/>/</==/!=).
    rank_by=결과 정렬 기준 지표명(미지정시 "sharpe"). 반환은 {{constraints_met, results:[{{entry_rule, exit_rule,
    performance, holdings, dates, navs, constraints_met}}...], best}} 이며, 제약을 만족하는 후보가 하나도
    없어도 에러가 아니라 rank_by 기준 가장 근접한 시도를 constraints_met=false로 정직하게 돌려줍니다.
    conn은 실행기가 자동 주입.
13. winsorize(rows, field, k) : field 값 중 [Q1-k·IQR, Q3+k·IQR] 범위 밖의 극단치만 경계로
    눌러 붙입니다(행 삭제 없음, k 미지정시 1.5). combine/zscore의 극단치 제어를 강화하고
    싶을 때, combine/zscore 이전 단계에서 먼저 씁니다. 원본 field는 그대로 두고
    '{{field}}_winsorized'를 새로 추가하므로, 다음 단계(zscore/combine)의 key는 반드시
    '{{field}}_winsorized'로 지정해야 눌린 값이 실제로 쓰입니다. "이상치를 눌러서/극단치를
    완화해서/윈저라이즈해서" 같은 표현이 질문에 있을 때만 씁니다(명시 없으면 안 씀 — combine의
    기본 zscore/rank_sum으로 충분).
14. correlation(rows, field_x, field_y) : rows에서 두 필드의 값끼리 피어슨 상관계수를 구합니다.
    "PBR과 GPA의 상관관계" 처럼 두 지표가 서로 얼마나 같이 움직이는지 물을 때 씁니다
    (compute_ic처럼 미래수익률과 비교하는 게 아니라, 같은 시점 두 필드끼리 비교). 반환
    {{correlation, n}} — correlation은 -1~1(None이면 한쪽 값이 전부 동일해 정의 불가), n은
    사용된 유효 표본 수. field_x/field_y는 get_cross_section 필드 중에서만 고릅니다.
15. quantile_bucket_means(rows, bucket_field, value_field, n) : bucket_field(예: pbr) 기준
    오름차순으로 정렬해 동일 개수로 n분위(미지정시 5, 즉 5분위=퀸타일)로 나누고, 분위별
    value_field(예: gp_a)의 평균을 구합니다. "PBR 분위수별 평균 GPA" 같은 팩터 분석 질문에
    씁니다. 반환은 [{{bucket, count, bucket_range:[최소,최대], mean_value}}, ...] 리스트이며
    bucket=1이 bucket_field가 가장 낮은 분위, bucket=n이 가장 높은 분위입니다.
16. remove_outliers(rows, field, k) : field 값이 [Q1-k·IQR, Q3+k·IQR] 범위를 벗어나는 row를
    통째로 제거해 나머지를 반환합니다(winsorize=값을 누르기와 달리, 이건 이상치 행을 실제로
    걷어냄, k 미지정시 1.5). "산점도에서 이상치를 빼고/제거하고 그려줘" 같은 요청에서 산점도
    데이터를 만들기 전에 씁니다. field가 None인 row는 유지하며, 원본 필드는 그대로 둡니다.
17. scatter_data(rows, x_field, y_field) : rows에서 두 필드를 산점도용 (x, y, labels)로 뽑아
    {{x, y, labels, x_field, y_field}} dict로 반환합니다. 두 지표의 관계를 점으로 뿌리는
    산점도(예: "이익수익률과 투하자본수익률 산점도")의 마지막 단계로 씁니다. x·y 두 좌표가
    모두 있는 종목만 포함됩니다(하나라도 None이면 제외). x_field/y_field는 get_cross_section
    필드 중에서만 고릅니다.
18. histogram_buckets(rows, field, num_buckets) : field(예: pbr) 값을 num_buckets개의 균등폭
    구간으로 나눠(quantile_bucket_means의 "동일 개수" 분위수와 다름 — 이건 "동일 폭" 구간)
    구간별 표본 개수를 센다. "히스토그램"/"분포"/"구간별 빈도" 질문에 씁니다. num_buckets
    미지정시 10. 반환 {{bucket_edges:[n+1개], counts:[n개], n}}. field는 get_cross_section
    필드 중에서만 고릅니다.
19. get_cross_section_qvm(asof, markets) : get_cross_section과 같지만 각 종목에 12-1 모멘텀
    (momentum_12_1: 최근 1개월 제외 12개월 수익률%)을 배치로 얹어 반환합니다. "퀄리티 밸류
    모멘텀(QVM)" 전략 스크리닝은 get_cross_section 대신 반드시 이걸로 시작해 compute_qvm_scores로
    넘깁니다. conn은 실행기가 자동 주입.
20. compute_qvm_scores(rows, quality_fields, value_source_fields, momentum_field, min_sector_n,
    winsorize_lower, winsorize_upper, max_missing, category_weights) : get_cross_section_qvm의 rows에
    사용자 확정 QVM 파이프라인을 적용해 각 종목에 qvm_score(최종 점수)를 얹어 반환합니다.
    내부 순서: 가치역수(E/P·B/P·S/P) → 1%/99% winsorize → 섹터 z-score(<5표본 전체폴백)
    → 카테고리 합성(Quality=roe/gp_a/cfo_ratio, Value=E/P·B/P·S/P, Momentum=12-1) → 결측필터
    (raw 7개 중 3개 이상 결측 제외) → 2차 z-score → 최종점수(등가중). 파라미터는 모두 선택이며
    기본값이 사용자 확정값(1%/99% winsorize, 섹터<5표본 폴백, 등가중 등)과 이미 정확히
    일치합니다 — 질문이 표준 QVM 명세를 그대로 서술하고 있을 뿐 특별히 다른 값을 요구하지
    않으면 파라미터를 생략하고 기본값을 그대로 쓰세요(질문 문장에 나온 숫자를 일일이 다시
    옮겨 적지 마세요). category_weights를 지정할 때는 반드시 [퀄리티가중치, 밸류가중치,
    모멘텀가중치] 순서의 숫자 배열이어야 합니다(예: [0.33, 0.33, 0.34]) — 절대
    {{"quality":..,"value":..,"momentum":..}} 같은 객체/딕셔너리로 쓰지 마세요(오류가 납니다).
    결과 rows를 combine(criteria=[{{"key":"qvm_score","direction":"high"}}], n=원하는수)으로
    넘겨 상위 종목을 뽑습니다.
21. run_qvm_backtest(start_year, end_year, n, rebalance, markets, quality_fields,
    value_source_fields, category_weights, with_benchmark) : QVM 전략으로 리밸런싱 백테스트를
    실행합니다(19·20을 매 리밸런싱 시점에 적용해 qvm_score 상위 n종목을 동일가중 보유). "QVM/
    퀄리티밸류모멘텀 전략으로 백테스트/리밸런싱" 질문에 씁니다. rebalance='monthly'|'quarterly'|
    'semiannual'|'annual', n=편입 종목수(기본 20). 반환 {{dates, navs, benchmark, performance,
    holdings}} — run_backtest와 동일 형식. conn은 실행기가 자동 주입. end_year 기본값은 위 공통 규칙 참고.
22~26. (QVM 저수준 빌딩블록 — 보통은 20번 compute_qvm_scores가 내부에서 자동으로 조립하므로
    직접 쓸 필요가 없습니다) invert_field(rows, field, out_field: 가치지표 역수 1/field),
    winsorize_pct(rows, field, lower_pct, upper_pct: 1%/99% 분위 클리핑),
    sector_zscore_with_fallback(rows, field, min_sector_n: 섹터 z-score, 표본부족시 전체폴백),
    composite_score(rows, fields, out_field, weights: z-score 가중평균, 결측 제외),
    drop_missing_factors(rows, fields, max_missing: 결측 초과 행 제거).

[JSON 형식]
- {{"pipeline": [{{"op": "이름", "params": {{...}}, "out": "결과이름"}}, ...]}}
- 앞 단계 결과를 뒤 단계 입력으로 넘길 때 params 값에 {{"$ref": "결과이름"}}을 씁니다.
- regress/optimize_weights의 시계열/수익률 값이 질문에 숫자로 주어지면 그 숫자를 그대로 넣습니다.
- 한 파이프라인이 서로 다른 out을 여러 개 만들어도(예: 상관계수+분위별평균+산점도 데이터를
  각각 별도 단계로) **여러 개의 최종 산출물이 모두 보존**되어 응답에 실립니다 — 뒤 단계의
  $ref로 한 번도 참조되지 않은 out은 전부 최종 산출물로 취급되므로, 질문이 여러 결과물을
  요구하면(예: "상관계수 구하고 산점도로, 분위별 평균은 막대그래프로") 주저 없이 필요한
  단계를 전부 넣으세요. 하나만 골라 나머지를 생략하지 마세요.

[예시]
Q: 2023년부터 2025년까지 3년간 매 분기 매출성장률이 가장 좋은 20개 종목으로 리밸런싱했을 때 누적수익률을 알려줘
A: {{"pipeline": [
  {{"op": "run_backtest", "params": {{"start_year": 2023, "end_year": 2025, "criteria": [{{"key": "revenue_growth", "direction": "high", "weight": 1.0}}], "n": 20, "rebalance": "quarterly"}}, "out": "bt"}}
]}}
(이 예시는 질문에 종료연도(2025)가 명시된 경우입니다. 질문에 종료연도가 없으면 위 공통
규칙대로 end_year는 오늘({today})이 속한 연도를 씁니다 — 이 예시의 2025를 그대로
베끼지 마세요.)
Q: RSI가 30 이하인 과매도 종목 중 PER도 낮은 10개 골라줘
A: {{"pipeline": [
  {{"op": "get_cross_section", "params": {{"asof": "{today}"}}, "out": "xs"}},
  {{"op": "compute_technical_indicator", "params": {{"rows": {{"$ref": "xs"}}, "asof": "{today}", "indicators": [{{"name": "rsi"}}]}}, "out": "xs_ti"}},
  {{"op": "combine", "params": {{"rows": {{"$ref": "xs_ti"}}, "criteria": [{{"key": "rsi_14", "direction": "low", "weight": 0.5}}, {{"key": "per", "direction": "low", "weight": 0.5}}], "method": "zscore", "n": 10}}, "out": "picked"}}
]}}
Q: ROE 극단치를 눌러서(윈저라이즈) 완화한 뒤 ROE 높은 20개 골라줘
A: {{"pipeline": [
  {{"op": "get_cross_section", "params": {{"asof": "{today}"}}, "out": "xs"}},
  {{"op": "winsorize", "params": {{"rows": {{"$ref": "xs"}}, "field": "roe"}}, "out": "xs_w"}},
  {{"op": "combine", "params": {{"rows": {{"$ref": "xs_w"}}, "criteria": [{{"key": "roe_winsorized", "direction": "high", "weight": 1.0}}], "method": "zscore", "n": 20}}, "out": "picked"}}
]}}
Q: PBR과 GPA의 상관관계 구하고 산점도로 보여줘. PBR을 5분위수로 나눠서 분위수별 평균 GPA도 막대그래프로 보여줘
A: {{"pipeline": [
  {{"op": "get_cross_section", "params": {{"asof": "{today}"}}, "out": "xs"}},
  {{"op": "correlation", "params": {{"rows": {{"$ref": "xs"}}, "field_x": "pbr", "field_y": "gp_a"}}, "out": "corr"}},
  {{"op": "quantile_bucket_means", "params": {{"rows": {{"$ref": "xs"}}, "bucket_field": "pbr", "value_field": "gp_a", "n": 5}}, "out": "buckets"}},
  {{"op": "scatter_data", "params": {{"rows": {{"$ref": "xs"}}, "x_field": "pbr", "y_field": "gp_a"}}, "out": "scatter"}}
]}}
(상관계수만 묻고 시각화 언급이 없으면 scatter_data/quantile_bucket_means 단계는 질문에 실제로 필요한 것만 넣으세요 — 이 예시는 상관관계+산점도+분위수+막대그래프를 모두 요구한 경우입니다.)
Q: 이익수익률과 투하자본수익률 산점도 그려줘, 이상치는 빼고
A: {{"pipeline": [
  {{"op": "get_cross_section", "params": {{"asof": "{today}"}}, "out": "xs"}},
  {{"op": "remove_outliers", "params": {{"rows": {{"$ref": "xs"}}, "field": "earnings_yield"}}, "out": "xs1"}},
  {{"op": "remove_outliers", "params": {{"rows": {{"$ref": "xs1"}}, "field": "roc"}}, "out": "xs2"}},
  {{"op": "scatter_data", "params": {{"rows": {{"$ref": "xs2"}}, "x_field": "earnings_yield", "y_field": "roc"}}, "out": "scatter"}}
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
(이 예시는 질문에 시작·종료연도(2024~2026)가 명시된 경우입니다. 질문에 종료연도가
없으면 위 공통 규칙대로 end_year는 오늘({today})이 속한 연도를 씁니다 — 이 예시의
2026을 "기본값"으로 오해해 그대로 베끼지 마세요.)
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
Q: 코스피 PBR을 100구간으로 쪼개서 히스토그램 그려줘
A: {{"pipeline": [
  {{"op": "get_cross_section", "params": {{"asof": "{today}", "markets": ["KOSPI"]}}, "out": "xs"}},
  {{"op": "histogram_buckets", "params": {{"rows": {{"$ref": "xs"}}, "field": "pbr", "num_buckets": 100}}, "out": "hist"}}
]}}
Q: 코스피 전종목 PER을 전체 시장 기준 z-score로 바꿔서 10구간 히스토그램으로 보여줘
A: {{"pipeline": [
  {{"op": "get_cross_section", "params": {{"asof": "{today}", "markets": ["KOSPI"]}}, "out": "xs"}},
  {{"op": "neutralize", "params": {{"rows": {{"$ref": "xs"}}, "field": "per", "by": null, "method": "zscore"}}, "out": "xs_z"}},
  {{"op": "histogram_buckets", "params": {{"rows": {{"$ref": "xs_z"}}, "field": "per_neutral", "num_buckets": 10}}, "out": "hist"}}
]}}
(by에 반드시 null을 명시하세요 — 생략하면 기본값 'sector'로 섹터별 z-score가 됩니다. 같은
패턴으로 field만 "pbr"/"roe" 등으로 바꾸거나, histogram_buckets 대신 scatter_data/
quantile_bucket_means 등 다른 후속 연산에 xs_z의 '{{field}}_neutral'을 넘기면 "PBR z-score
산점도", "ROE z-score 구간별 평균" 같은 다른 조합에도 동일하게 적용됩니다.)
Q: 퀄리티 밸류 모멘텀 전략으로 상위 20종목 뽑아줘
A: {{"pipeline": [
  {{"op": "get_cross_section_qvm", "params": {{"asof": "{today}"}}, "out": "xs"}},
  {{"op": "compute_qvm_scores", "params": {{"rows": {{"$ref": "xs"}}}}, "out": "scored"}},
  {{"op": "combine", "params": {{"rows": {{"$ref": "scored"}}, "criteria": [{{"key": "qvm_score", "direction": "high", "weight": 1.0}}], "method": "zscore", "n": 20}}, "out": "picked"}}
]}}
Q: 퀄리티 밸류 모멘텀 전략으로 2024년부터 2026년까지 분기별 리밸런싱 백테스트해줘
A: {{"pipeline": [
  {{"op": "run_qvm_backtest", "params": {{"start_year": 2024, "end_year": 2026, "n": 20, "rebalance": "quarterly"}}, "out": "bt"}}
]}}
(이 예시는 질문에 시작·종료연도(2024~2026)가 명시된 경우입니다. 질문에 종료연도가
없으면 위 공통 규칙대로 end_year는 오늘({today})이 속한 연도를 씁니다 — 이 예시의
2026을 "기본값"으로 오해해 그대로 베끼지 마세요.)

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


def validate_pipeline_steps(steps: list[dict]) -> list[str]:
    """steps를 run_pipeline에 넘기기 전에 정적으로 검사한다(실행 없이).

    run_pipeline은 steps를 순서대로 실행하므로, 검증 없이는 파이프라인 뒷부분(예: 20단계
    중 15번째)의 오류가 앞부분의 무거운 연산(DB 조회 등)을 이미 다 실행한 뒤에야 드러난다
    (이 머신엔 실거래봇 quant_trader가 상주하므로 낭비된 연산도 비용이다). 여기서는 실행
    없이 확인 가능한 두 가지만 미리 잡는다: (1) LLM이 지어낸 알 수 없는 연산, (2) 아직
    정의되지 않은 결과를 가리키는 "$ref"(오타 등). 발견된 오류 메시지 리스트를 반환하며,
    비어있으면 유효하다는 뜻이다.
    """
    if not isinstance(steps, list):
        return ["pipeline steps는 list여야 합니다"]

    errors: list[str] = []
    defined: set[str] = set()
    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errors.append(f"스텝 {i}: 각 스텝은 객체(dict)여야 합니다")
            continue
        op = step.get("op")
        if op not in PRIMITIVE_OPS:
            errors.append(f"스텝 {i}: 알 수 없는 연산 '{op}'")
        params = step.get("params", {})
        if isinstance(params, dict):
            for val in params.values():
                if isinstance(val, dict) and set(val.keys()) == {"$ref"}:
                    ref = val["$ref"]
                    if ref not in defined:
                        errors.append(f"스텝 {i}: 참조 '{ref}'가 아직 정의되지 않았습니다")
        out = step.get("out")
        if out:
            defined.add(out)
    return errors


def _infer_requested_count(steps: list[dict]) -> int | None:
    """파이프라인 steps에서 최종 선정 개수(n)를 추정한다.

    검증/종합결론 LLM 프롬프트가 긴 결과 리스트를 앞부분 5개로 축약할 때
    (supervisor._truncate_for_prompt), 도메인 결과 dict에 top_n이 있으면 그 개수까지는
    자르지 않는다 — kr/us 스크리닝은 이미 top_n을 담아 이 혜택을 받지만, backtest 도메인은
    담지 않아 "QVM 상위 20종목" 같은 요청도 5개로 잘려 검증 LLM이 "일부만 있다"고
    오판하는 사례가 실서버에서 재현됐다. combine 등 선정형 프리미티브의 n 파라미터를
    스캔해 재사용한다(여러 단계에 n이 있으면 파이프라인상 더 뒤(최종 선정에 가까운) 값을
    우선). 못 찾으면 None(기존 head=5 그대로, 동작 불변)."""
    found: int | None = None
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        params = step.get("params")
        if isinstance(params, dict) and isinstance(params.get("n"), int):
            found = params["n"]
    return found


def _infer_cross_section_asof(steps: list[dict]) -> str | None:
    """steps에서 get_cross_section(_qvm)의 asof 파라미터를 찾는다.

    compute_qvm_scores 자체는 asof를 모른다(그 앞 단계인 get_cross_section_qvm의
    파라미터로만 존재) — _infer_requested_count와 동일한 관례로 steps(JSON)를 실행 없이
    정적으로 스캔한다. 여러 단계에 있으면 마지막(파이프라인상 더 뒤) 값을 우선한다.
    QVM 파이프라인에 한정되지 않고 correlation/quantile_bucket_means 등 get_cross_section을
    쓰는 모든 파이프라인의 data_asof(아래 _build_data_asof)에도 재사용한다."""
    found: str | None = None
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        if step.get("op") in ("get_cross_section_qvm", "get_cross_section"):
            params = step.get("params")
            if isinstance(params, dict) and isinstance(params.get("asof"), str):
                found = params["asof"]
    return found


def _build_data_asof(steps: list[dict], conn=None) -> dict | None:
    """get_cross_section(_qvm)의 요청 asof를 kr/us 도메인과 동일한 data_asof 키로 노출한다.

    실서버 재현: "코스피 전종목 pbr/gpa 상관관계 5분위 평균" 같은 correlation/
    quantile_bucket_means 파이프라인은 집계값만 반환해 result에 시점 정보가 전혀 남지
    않는다(_qvm_scored_rows처럼 종목별 quarter를 보존하지 않음) — steps에서 정적으로
    추출한 요청 asof를 kr/us 도메인이 쓰는 {"price_date": ...} 형태로 그대로 노출하면,
    supervisor.py의 종합결론 로직(도메인 무관하게 data_asof를 언급하도록 이미 배선됨)이
    별도 분기 없이 재사용된다. get_cross_section을 쓰지 않는 파이프라인(run_backtest 등)은
    None을 반환해 result_payload에 키 자체를 붙이지 않는다(top_n/qvm_summary와 동일한
    하위호환 관례).

    conn이 주어지면 mode_financial_quarter_at(data_access.py)으로 "재무 기준분기"도 함께
    붙인다(price_date만으로는 사용자가 재무제표 시점을 검증할 수 없다는 실사용 리포트
    대응). conn이 없거나(단위테스트 등) 재무데이터가 전혀 없으면 price_date만 반환한다
    (kr/us의 "값이 있는 키만" 관례와 동일)."""
    asof = _infer_cross_section_asof(steps)
    if not asof:
        return None
    result: dict = {"price_date": asof}
    if conn is not None:
        quarter = mode_financial_quarter_at(conn, asof)
        if quarter:
            result["financial_quarter"] = quarter
    return result


def _find_qvm_scored_rows(result) -> list[dict] | None:
    """result(파이프라인 leaf 산출물)에서 compute_qvm_scores가 채점한 rows를 찾는다.

    qvm_summary 계산용 — compute_qvm_scores가 각 row에 남기는 내부 마커
    '_qvm_excluded_count'로 식별한다(combine 등 후속 선정 단계도 원본 dict 참조를 그대로
    넘기므로, top_n으로 걸러진 뒤에도 이 마커는 살아남는다 — select_stocks._filter_valid가
    새 dict를 만들지 않기 때문). result이 list(단일 leaf, 흔한 경우)든 dict(다중 leaf,
    {"out이름": 값})든 찾아낸다. 못 찾으면 None(QVM 파이프라인이 아니거나, 결측필터로
    전부 제외돼 마커를 남길 행이 하나도 없던 경우)."""

    def _matches(v) -> bool:
        return (
            isinstance(v, list) and len(v) > 0
            and all(isinstance(r, dict) for r in v)
            and "_qvm_excluded_count" in v[0]
        )

    if _matches(result):
        return result
    if isinstance(result, dict):
        for v in result.values():
            if _matches(v):
                return v
    return None


def _build_qvm_summary(steps: list[dict], result) -> dict | None:
    """QVM 스크리닝 파이프라인이면 사용자 요청 [출력] 절의 요약 3종(asof/excluded_count/
    sector_distribution)을 만든다. steps에 compute_qvm_scores 단계가 없으면(QVM 파이프라인이
    아님) None을 반환해 result_payload에 아무 것도 추가하지 않는다(하위호환 — kr/us의
    top_n과 동일한 관례).

    excluded_count/sector_distribution은 _find_qvm_scored_rows로 찾은 '최종 결과 rows'
    (combine으로 top_n까지 걸러졌으면 그 이후 rows, 그런 단계가 없으면 compute_qvm_scores의
    전체 스코어 유니버스)에서 계산한다 — 있는 그대로의 최종 결과셋을 그대로 반영하므로
    top_n 반영 여부와 무관하게 자연스럽게 동작한다."""
    if not any(isinstance(s, dict) and s.get("op") == "compute_qvm_scores" for s in steps or []):
        return None
    rows = _find_qvm_scored_rows(result)
    if rows is None:
        return None
    sector_distribution: dict = {}
    for r in rows:
        sector = r.get("sector")
        sector_distribution[sector] = sector_distribution.get(sector, 0) + 1
    return {
        "asof": _infer_cross_section_asof(steps),
        "excluded_count": rows[0]["_qvm_excluded_count"],
        "sector_distribution": sector_distribution,
    }


def _extract_backtest_holdings(result) -> list | None:
    """result(파이프라인 leaf 산출물)에서 백테스트 holdings 리스트를 찾는다.

    단일 백테스트면 result은 {dates,navs,...,holdings} dict라 바로 꺼내고, 다중 leaf
    (dict of {out이름: 값})면 holdings를 가진 dict를 찾아 그 안에서 꺼낸다. 못 찾으면 None.
    """
    def _holdings_of(v):
        if isinstance(v, dict) and isinstance(v.get("holdings"), list):
            return v["holdings"]
        return None

    direct = _holdings_of(result)
    if direct is not None:
        return direct
    if isinstance(result, dict):
        for v in result.values():
            found = _holdings_of(v)
            if found is not None:
                return found
    return None


def _build_rebalance_summary(result) -> str | None:
    """다중 리밸런싱 백테스트면 리밸런싱 시점별 보유종목+구간수익률을 사람이 읽기 좋은
    결정론적 텍스트로 만든다(top_n/qvm_summary와 동일한 순수 추가 필드 관례).

    holdings가 2개 이상(실제로 여러 번 리밸런싱한 경우)일 때만 만든다 — buy&hold(단일
    리밸런싱)엔 구간별 서술이 의미가 없어 None을 반환한다(하위호환: 키 자체가 안 붙음).
    supervisor가 이 텍스트를 최종 결론에 그대로 덧붙여 LLM 요약 재량과 무관하게 '항상'
    포함을 보장한다(사용자 요구: "반기마다 어떤 종목이 있었는지·반기별 수익률도 같이 항상").
    """
    holdings = _extract_backtest_holdings(result)
    if not holdings or len(holdings) < 2:
        return None
    lines = ["리밸런싱 구간별 보유종목·구간수익률:"]
    for h in holdings:
        if not isinstance(h, dict):
            continue
        date_str = h.get("date", "?")
        names = h.get("names") or h.get("codes") or []
        held = ", ".join(str(x) for x in names) if names else "(보유 없음)"
        pr = h.get("period_return")
        if isinstance(pr, (int, float)):
            lines.append(f"- {date_str}: {held} (구간수익률 {pr * 100:+.2f}%)")
        else:
            lines.append(f"- {date_str}: {held}")
    return "\n".join(lines)


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
        market: "KR"(국내 전용). 기본 스냅샷 에이전트 선택과 하드차단(생존편향 검증불가) 판정에 쓰인다.
        stock_codes: 준비할 시계열 스냅샷의 종목코드/티커(단일 str 또는 list). 주면 데이터
            준비 단계를 수행하고, 없으면 스냅샷 조회를 생략한다.
        asof / indicators: 스냅샷 기준시점 / 부착할 기술지표(스냅샷 에이전트에 그대로 전달).
        run_pipeline_fn: 백테스트 실행 콜백(기본=pipeline_exec.run_pipeline). 재발명 금지 —
            기존 계산 엔진을 그대로 주입한다.
        run_audit_fn: 감사배선 콜백(기본=HA-5 run_backtest_with_audit). 테스트 주입용.
        snapshot_fn: 데이터준비 콜백. 기본은 국내 스냅샷 에이전트(get_price_snapshot_kr).
        execute_sql_fn: 스냅샷 에이전트가 쓸 HA-1 실행기(기본=execute_sql). 테스트 주입용.
        generate_steps_fn: steps 자동생성 콜백(기본=generate_backtest_steps). 테스트 주입용.

    Returns:
        run_backtest_with_audit(HA-5)의 반환에 준비된 스냅샷(data)을 더한 dict:
        {"blocked": bool, "error": str|None, "result": dict|None, "hard": [...],
         "warnings": [...], "data": list[dict]}
        하드차단 시 result=None(결과 폐기), 통과 시 result=백테스트 결과 + warnings=triggered.
        result이 list이고 steps에 combine 등 선정형 n 파라미터가 있으면 top_n도 더한다.
        steps에 compute_qvm_scores 단계가 있으면(QVM 스크리닝) qvm_summary={"asof":
        str|None, "excluded_count": int, "sector_distribution": {섹터명: 종목수}}도 더한다
        (QVM 파이프라인이 아니면 이 키 자체가 없다 — 하위호환).
        steps에 get_cross_section(_qvm) 단계가 있으면 data_asof={"price_date": str,
        "financial_quarter": str}(재무 기준분기는 재무데이터가 있을 때만)도 더한다
        (kr 도메인과 동일 키 — supervisor.py 종합결론이 도메인 무관하게 노출).
        get_cross_section을 쓰지 않는 파이프라인(run_backtest 등)은 이 키가 없다.
    """
    run_pipeline_fn = run_pipeline_fn or run_pipeline
    run_audit_fn = run_audit_fn or run_backtest_with_audit
    generate_steps_fn = generate_steps_fn or generate_backtest_steps
    if snapshot_fn is None:
        snapshot_fn = get_price_snapshot_kr

    # 0) steps 자동생성(옵션): 호출자가 파이프라인을 안 만들어서 넘겼고(steps 비어있음) LLM이
    #    가용하면, 질문에서 직접 생성한다. steps를 이미 준 호출부(기존 테스트 등)는 그대로 존중.
    if not steps and llm_fn is not None:
        if on_progress:
            on_progress("audit", "실행계획 생성 중…")
        steps = generate_steps_fn(question, llm_fn)
        if on_progress:
            on_progress(
                "audit", f"실행계획 생성 완료({len(steps)}단계)",
                detail={"kind": "backtest_pipeline", "steps": steps},
            )

    # 0.5) 사전검증: run_pipeline이 실제로 실행하기 전에 구조적 오류(알 수 없는 연산/
    #      미정의 참조)를 잡는다. 통과하지 못하면 데이터 준비/감사배선 모두 건너뛰고
    #      즉시 불확실 응답으로 처리한다 — 잘못된 파이프라인의 앞부분만 실행하고
    #      뒷부분에서야 실패하는 연산 낭비를 막는다.
    validation_errors = validate_pipeline_steps(steps)
    if validation_errors:
        if on_progress:
            on_progress("audit", f"실행계획 검증 실패({len(validation_errors)}건)")
        return {
            "blocked": True,
            "error": "실행계획이 유효하지 않습니다: " + "; ".join(validation_errors),
            "result": None,
            "hard": [],
            "warnings": [],
            "data": [],
        }

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

    result_payload = {**audit, "data": data}
    if isinstance(audit.get("result"), list):
        requested_n = _infer_requested_count(steps)
        if requested_n is not None:
            result_payload["top_n"] = requested_n
    qvm_summary = _build_qvm_summary(steps, audit.get("result"))
    if qvm_summary is not None:
        result_payload["qvm_summary"] = qvm_summary
    data_asof = _build_data_asof(steps, conn=conn)
    if data_asof is not None:
        result_payload["data_asof"] = data_asof
    rebalance_summary = _build_rebalance_summary(audit.get("result"))
    if rebalance_summary is not None:
        result_payload["rebalance_summary"] = rebalance_summary
    return result_payload
