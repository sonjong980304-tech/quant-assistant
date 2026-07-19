"""LLM 프롬프트 (질문 정제 / SQL 생성 / Judge)."""
from __future__ import annotations

# --------------------------------------------------------------------------
# refine_node
# --------------------------------------------------------------------------
REFINE_SYSTEM = (
    "당신은 한국 주식 재무 데이터 질의를 명확하게 다듬는 어시스턴트입니다. "
    "모호한 자연어 질문을 SQL로 옮기기 좋은 구체적 질문으로 바꾸세요."
)

REFINE_USER = """다음 질문을 SQL 변환에 적합하도록 명확히 정제하세요.

규칙:
- 정렬 방향, 개수(N), 대상 지표를 명시적으로 표현 (예: "PER 낮은거" → "PER이 낮은 순으로 상위 10개 회사").
- 개수가 없으면 상위 10개로 가정.
- 의미는 절대 바꾸지 말 것. 새로운 조건을 임의로 추가하지 말 것.
- 한 문장의 한국어 질문만 출력. 설명/따옴표 금지.

원본 질문: {question}

정제된 질문:"""


# --------------------------------------------------------------------------
# sql_gen_node
# --------------------------------------------------------------------------
SQL_SYSTEM = (
    "당신은 한국 상장사 재무/주가 SQLite 데이터베이스를 위한 정확한 Text-to-SQL 엔진입니다. "
    "오직 표준 SQLite 문법의 SELECT 문 하나만 생성합니다."
)

SQL_USER = """아래 스키마로 질문에 답하는 SQLite SELECT 문 하나를 작성하세요.

[스키마]
{schema}

[테이블 구조]
- company(stock_code, name, market, sector)
  · sector는 KRX 업종분류 29종만 존재: IT 서비스/건설/금속/금융/기계·장비/기타금융/기타제조/
    농업 임업 및 어업/보험/부동산/비금속/섬유·의류/오락·문화/운송·창고/운송장비·부품/유통/은행/
    음식료·담배/의료·정밀기기/일반서비스/전기·가스/전기·가스·수도/전기·전자/제약/종이·목재/증권/
    출판·매체복제/통신/화학
  · '반도체'·'게임'·'자동차' 같은 업종명은 없다 → 각각 '전기·전자', 'IT 서비스' 또는 '오락·문화',
    '운송장비·부품'으로 매핑. 업종 조건은 sector LIKE '%…%' 포괄 매칭을 쓴다(예: '금융'→금융+기타금융).
  · 미분류 업종은 NULL이 아니라 빈 문자열('')로 저장된다. 업종 개수·미분류 제외 질의는
    sector != ''를 쓸 것(sector IS NOT NULL은 빈 문자열을 걸러내지 못해 틀린 개수가 나온다).
- financials(stock_code, quarter, disclosed_date, account_key, amount)
  · account_key 값: revenue(매출), cost_of_sales(매출원가), gross_profit(매출총이익), sga(판관비),
    operating_profit(영업이익), net_income(당기순이익), controlling_net_income(지배기업소유주지분순이익),
    interest_expense(이자비용), operating_cashflow(영업활동현금흐름), depreciation(감가상각비), dividend(배당금),
    total_assets, total_liabilities, total_equity, controlling_equity(지배기업소유주지분자본),
    current_assets, non_current_assets, current_liabilities, non_current_liabilities
  · 손익·현금흐름은 '분기 단독' 값, 자산/부채/자본은 분기말 잔액. 최신분기=(SELECT MAX(quarter) FROM financials)
- prices(stock_code, date, close, market_cap)  · 시가총액=market_cap, 최신일=(SELECT MAX(date) FROM prices)

[지표는 financials/prices 원본에서 직접 계산한다 (사전계산 테이블 없음)]
- 영업이익률 = operating_profit/revenue*100, 순이익률 = net_income/revenue*100  (특정 분기 단독)
- 부채비율 = total_liabilities/total_equity*100, 유동비율 = current_assets/current_liabilities*100
- ROE = controlling_net_income(지배기업소유주지분순이익, TTM 최근4분기합)/controlling_equity(지배기업소유주지분자본)*100
  (비지배지분이 섞인 net_income/total_equity는 쓰지 않는다 — 지배기업 주주 관점의 정확한 자기자본이익률을 위해 분자·분모 모두 지배기업 귀속분으로 맞춘다.)
- PER = market_cap/net_income(TTM), PBR = market_cap/total_equity, PSR = market_cap/revenue(TTM)
- 분모는 NULLIF(..,0)로 0 나눗셈 방지, 분모<=0인 행은 WHERE로 제외(금융/지주사는 매출 NULL→자동 제외).
- 영업이익률/순이익률 질의 시에는 반드시 'op < rev'(이익<매출, 즉 마진<100%) 조건을 둘 것. 지주사·금융은 영업이익에 지분법이익/투자수익이 잡혀 매출보다 커서 비정상 폭발(예: SK스퀘어 2756%)하므로, 이 조건으로 자동 제외해 사업회사만 남긴다.
- ROE 질의 시에는 반드시 'ni.ttm < e.eq'(=ROE<100%, e.eq는 controlling_equity) 조건을 둘 것. ① 자기자본이 매우 작은 종목(자본잠식 직전·회복 구간)은 분모가 작아 ROE가 수백~수천%로 폭발하고 ② 일회성·오류성 거대 손익(TTM순이익이 자기자본을 초과)도 ROE를 왜곡하므로, 이 조건으로 정상 우량기업만 남긴다(정상 ROE는 보통 5~30%). '높은 순' 랭킹처럼 우량기업만 보여줄 때는 ni.ttm>0도 함께 둔다. '낮은 순'처럼 적자 기업(음수 ROE)도 정상적으로 보여줘야 하는 질의는 ni.ttm>0을 넣지 말 것(영업이익률 낮은 순과 동일한 관례).
- PER 질의 시에는 영업이익 정합(op_ttm>0 AND ni_ttm<op_ttm)과 ni_ttm<market_cap(=PER>1) 조건을 둘 것. ① 영업적자인데 순이익만 흑자인 일회성 영업외이익(예: 한창·유니온)과 ② 지주사의 연결순이익 vs 단독시총 미스매치(예: 다우데이타)로 PER이 1 미만으로 비현실적으로 폭락하는 것을 막는다(정상 PER은 보통 5~30).
- 순위/스크리닝('가장 ~한 N개', '높은/낮은 순') 질의는 **현재 거래 중인 상장사만** 대상으로 한다. DART는 상장폐지·비상장 외감법인 재무도 주므로, financials만 쓰는 지표(ROE·영업이익률·부채비율 등)라도 'JOIN prices p ON p.stock_code=...stock_code AND p.date=(SELECT MAX(date) FROM prices)'로 최신 거래일 주가가 있는 종목만 남겨, 주가 없는 상폐/비상장 종목(예: 대선조선)을 자동 제외한다.
- **'금액' 지표(영업이익·순이익·매출·시가총액 등 절댓값)를 묻는 질의에는 상한(operating_profit<=N 같은) 조건을 절대 두지 말 것.** 금액은 회사 규모에 따라 수천억~수조 원까지 정상이라, 상한을 두면 대기업이 통째로 누락된다(예: '영업이익 높은 회사'에 ≤50억 상한을 걸면 진짜 1위가 빠짐). 상한·정합 가드는 비율 지표(PER·ROE·영업이익률 등)에만 적용하고, 금액 랭킹은 'WHERE 금액>0 ORDER BY 금액 DESC'만 쓴다.
- 단, **영업이익·순이익 '금액' 랭킹**은 매출도 함께 집계해 'rev>0 AND op<rev'(사업회사: 영업이익<매출) 조건을 둔다. 지주사·금융이 지분법이익/투자수익을 영업이익에 잡아 매출보다 크게 부풀리는 것(예: SK스퀘어 영업이익 8.28조인데 매출 0.30조)을 빼고 순수 사업회사만 남기기 위함이다. (매출·시가총액 금액 랭킹에는 적용하지 않는다.)
- **주의: 위 정합성 가드(op<rev, ni.ttm<e.eq, ni_ttm<op_ttm 등)는 여러 회사를 비교하는 순위/스크리닝 질의에만 적용한다. 특정 회사 1곳을 지목한 단일조회 질의에는 적용하지 않는다.** 이 가드는 데이터 오류성 이상치가 순위 1위를 차지하는 것을 막기 위한 것이지, 그 회사의 실제 지표를 숨기기 위한 것이 아니다 — 일부 우량 기업은 자사주매입 등으로 ROE가 100%를 실제로 넘기도 하므로, 단일회사 조회는 있는 그대로 보여준다.

[SQLite 주의]
- SQLite 전용. YEAR()/대괄호[] 금지. 분기는 'YYYYQN' 문자열('2026Q1'). "26년 1분기"="2026Q1".
- financials는 (account_key, amount) 구조. 계정은 WHERE account_key=... 로 거르고, 여러 계정/비율은 MAX(CASE WHEN ..) 피벗.
- "영업이익률"(%)≠"영업이익"(금액). '률/율/마진'은 비율(나눗셈), 그 외는 금액.

[규칙]
- SELECT 문 하나만. 회사는 c.name 포함. 정렬/조건에 쓴 값은 SELECT에도 포함. 불필요 컬럼(stock_code) 금지.
- 개수는 COUNT(*). 평균/그룹은 AVG/SUM+GROUP BY.
- LIMIT: 개수 명시 시 그 수 / 단수("가장~한 회사") 1 / 미지정 목록 20.

[예시]
Q: 삼성전자 2026년 1분기 매출과 영업이익
A: SELECT c.name,
     MAX(CASE WHEN f.account_key='revenue' THEN f.amount END) AS revenue,
     MAX(CASE WHEN f.account_key='operating_profit' THEN f.amount END) AS operating_profit
   FROM company c JOIN financials f ON c.stock_code=f.stock_code
   WHERE c.name LIKE '%삼성전자%' AND f.quarter='2026Q1' AND f.account_key IN ('revenue','operating_profit')
   GROUP BY c.name;
Q: 영업이익이 가장 높은 5개 회사
A: SELECT c.name, op AS operating_profit FROM (
     SELECT stock_code,
       MAX(CASE WHEN account_key='operating_profit' THEN amount END) op,
       MAX(CASE WHEN account_key='revenue' THEN amount END) rev
     FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
   ) f JOIN company c ON c.stock_code=f.stock_code
   JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
   WHERE op>0 AND rev>0 AND op<rev ORDER BY op DESC LIMIT 5;
Q: 영업이익률이 가장 높은 10개 회사
A: SELECT c.name, ROUND(op*100.0/rev,2) AS operating_margin FROM (
     SELECT stock_code,
       MAX(CASE WHEN account_key='operating_profit' THEN amount END) op,
       MAX(CASE WHEN account_key='revenue' THEN amount END) rev
     FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
   ) f JOIN company c ON c.stock_code=f.stock_code
   WHERE rev>0 AND op IS NOT NULL AND op < rev ORDER BY operating_margin DESC LIMIT 10;
Q: PER이 가장 낮은 5개 회사
A: SELECT c.name, ROUND(p.market_cap/f.ni_ttm,2) AS per FROM (
     SELECT stock_code,
       SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
       SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
     FROM financials
     WHERE account_key IN ('net_income','operating_profit')
       AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
     GROUP BY stock_code) f
   JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
   JOIN company c ON c.stock_code=f.stock_code
   WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap
   ORDER BY per ASC LIMIT 5;
Q: ROE가 높은 10개 회사
A: SELECT c.name, ROUND(ni.ttm*100.0/e.eq,2) AS roe FROM (
     SELECT stock_code, SUM(amount) ttm FROM financials
     WHERE account_key='controlling_net_income'
       AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
     GROUP BY stock_code) ni
   JOIN (SELECT stock_code, amount eq FROM financials
         WHERE account_key='controlling_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
     ON e.stock_code=ni.stock_code
   JOIN company c ON c.stock_code=ni.stock_code
   JOIN prices p ON p.stock_code=ni.stock_code AND p.date=(SELECT MAX(date) FROM prices)
   WHERE ni.ttm>0 AND e.eq>0 AND ni.ttm < e.eq ORDER BY roe DESC LIMIT 10;
Q: 부채비율이 200%를 넘는 회사
A: SELECT c.name, ROUND(l*100.0/e,2) AS debt_ratio FROM (
     SELECT stock_code,
       MAX(CASE WHEN account_key='total_liabilities' THEN amount END) l,
       MAX(CASE WHEN account_key='total_equity' THEN amount END) e
     FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
   ) f JOIN company c ON c.stock_code=f.stock_code
   WHERE e>0 AND l*100.0/e>200 ORDER BY debt_ratio DESC LIMIT 20;
Q: 업종이 몇 개나 있어
A: SELECT COUNT(DISTINCT sector) AS sector_count FROM company WHERE sector != '';

질문: {question}

SQL:"""


# --------------------------------------------------------------------------
# sql_gen_node — route=="pipeline" (SQL로 표현 불가능한 통계/퀀트 분석)
# --------------------------------------------------------------------------
# **절대 원칙**: LLM은 파이썬 코드를 생성하지 않는다. 사전검증된 프리미티브 6종을 JSON으로
# "조립"만 하고, 결정론적 실행기(src/backtest/pipeline_exec.py)가 고정 dict로 실행한다.
PIPELINE_SYSTEM = (
    "당신은 SQL로 표현 불가능한 통계/퀀트 분석 질문을, 사전검증된 프리미티브 연산블록을 "
    "정해진 순서로 조립한 JSON 파이프라인으로 변환하는 엔진입니다. 파이썬 코드는 절대 "
    "생성하지 않고, 오직 프리미티브 이름+파라미터로 구성된 JSON 하나만 출력합니다."
)

PIPELINE_USER = """아래 질문을 프리미티브 조립 JSON 파이프라인으로 변환하세요. 파이썬 코드 금지, JSON만 출력.

[사용 가능한 프리미티브 10종 — 이 외의 함수는 존재하지 않습니다]
1. get_cross_section(asof) : 특정 시점의 전종목 횡단면 지표 스냅샷(list of rows) 반환.
   각 row 필드: stock_code,name,sector,market,quarter,close,market_cap,per,pbr,psr,
   roe,roa,operating_margin,net_margin,debt_ratio,revenue_growth,op_growth,ni_growth.
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
   국내(KOSPI/KOSDAQ) 종목만 대상으로 하며 벤치마크는 시장 동일가중 유니버스입니다.
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

[JSON 형식]
- {{"pipeline": [{{"op": "이름", "params": {{...}}, "out": "결과이름"}}, ...]}}
- 앞 단계 결과를 뒤 단계 입력으로 넘길 때 params 값에 {{"$ref": "결과이름"}}을 씁니다.
- regress/optimize_weights의 시계열/수익률 값이 질문에 숫자로 주어지면 그 숫자를 그대로 넣습니다.

[예시]
Q: 최근 시점 저PER·고ROE 우량주를 섹터중립으로 20개 골라줘
A: {{"pipeline": [
  {{"op": "get_cross_section", "params": {{"asof": "{today}"}}, "out": "xs"}},
  {{"op": "neutralize", "params": {{"rows": {{"$ref": "xs"}}, "field": "roe", "by": "sector"}}, "out": "xn"}},
  {{"op": "combine", "params": {{"rows": {{"$ref": "xn"}}, "criteria": [{{"key": "per", "direction": "low", "weight": 0.5}}, {{"key": "roe_neutral", "direction": "high", "weight": 0.5}}], "method": "zscore", "n": 20}}, "out": "picked"}}
]}}
Q: 누적수익률 [0, 1.2, 2.1, 2.9, 4.3, 5.0]의 K-ratio를 구해줘
A: {{"pipeline": [
  {{"op": "regress", "params": {{"y": [0, 1.2, 2.1, 2.9, 4.3, 5.0]}}, "out": "reg"}}
]}}
Q: 세 종목 수익률로 최대샤프 포트폴리오 비중을 계산해줘 (A=[0.01,0.02,-0.01], B=[0.0,0.01,0.02], C=[0.02,-0.01,0.03])
A: {{"pipeline": [
  {{"op": "optimize_weights", "params": {{"returns": {{"A": [0.01, 0.02, -0.01], "B": [0.0, 0.01, 0.02], "C": [0.02, -0.01, 0.03]}}, "method": "max_sharpe"}}, "out": "w"}}
]}}
Q: 2023년부터 2025년까지 3년간 매 분기 매출성장률이 가장 좋은 20개 종목으로 리밸런싱했을 때 누적수익률을 알려줘
A: {{"pipeline": [
  {{"op": "run_backtest", "params": {{"start_year": 2023, "end_year": 2025, "criteria": [{{"key": "revenue_growth", "direction": "high", "weight": 1.0}}], "n": 20, "rebalance": "quarterly"}}, "out": "bt"}}
]}}
Q: 2023년부터 2025년까지 매출성장률 팩터의 분기별 IC를 알려줘
A: {{"pipeline": [
  {{"op": "compute_ic", "params": {{"start_year": 2023, "end_year": 2025, "field": "revenue_growth", "rebalance": "quarterly"}}, "out": "ic"}}
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
      [{{"key": "roe", "direction": "high", "weight": 1.0}}],
      [{{"key": "per", "direction": "low", "weight": 0.5}}, {{"key": "roe", "direction": "high", "weight": 0.5}}]
    ],
    "constraints": [{{"metric": "mdd", "op": ">=", "value": -10.0}}],
    "rank_by": "sharpe"
  }}, "out": "found"}}
]}}

질문: {question}

JSON:"""


# --------------------------------------------------------------------------
# eval_node (Layer2 LLM-as-Judge)
# --------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "당신은 Text-to-SQL 결과를 평가하는 엄격한 채점관입니다. "
    "생성된 SQL이 질문 의도를 정확히 반영하는지 1~5점으로 평가합니다."
)

JUDGE_USER = """질문 의도에 비춰 생성된 SQL의 적절성을 1~5점으로 평가하세요.

채점 기준:
5 = 의도를 완벽히 반영(올바른 지표/정렬/필터/개수)
4 = 대체로 정확하나 사소한 차이
3 = 핵심은 맞으나 일부 조건 누락/오류
2 = 부분적으로만 관련
1 = 의도와 무관하거나 잘못됨

[스키마 요약]
{schema}

[질문]
{question}

[생성된 SQL]
{sql}

[실행 결과 요약]
행 수: {row_count}, 컬럼: {columns}
샘플: {sample}

JSON으로만 답하세요: {{"score": <1-5 정수>, "reason": "<간단한 한국어 사유>"}}"""


# ---------------------------------------------------------------------------
# 진단 (diagnose_node) — 결과 이상의 '원인'을 분류
# ---------------------------------------------------------------------------
DIAGNOSE_SYSTEM = (
    "당신은 Text-to-SQL 결과의 이상 원인을 진단하는 분석가입니다. "
    "코드가 수집한 증거(조건별 분해표·이상치 원본값)에 근거해, 결과가 이상한 원인을 정확히 "
    "한 가지로 분류합니다. 데이터를 직접 보지 못하므로 반드시 주어진 증거에만 근거하고, "
    "증거가 불충분하면 추측하지 말고 data(사람 확인)로 보냅니다."
)

DIAGNOSE_USER = """생성된 SQL이 실행은 됐으나 결과가 의심스럽거나(빈결과/개수부족/이상치) 실행에 실패했습니다. 원인을 분류하세요.

[원인 4가지]
- sql   : SQL 자체 오류(문법/잘못된 조건/이상치 가드 누락 등). SQL을 고치면 해결. fixable=true.
- refine: 질문 해석이 빗나감(엉뚱한 지표·대상). 질문 재정제로 해결. fixable=true.
- data  : 데이터가 없거나 오염됨(미수집·단위오류). SQL로 못 고침. 사람 확인 필요. fixable=false.
- none  : 사실 정상임. 결과가 적거나 특이해도 그게 올바른 답(예: 조건이 까다로워 N개뿐).

[판단 지침]
- 조건별 분해표: 특정 조건에서 행이 급감하면, '데이터가 비어서'면 data, '값 기준이 까다로워서'면 none.
  · 예) 'market_cap IS NOT NULL'에서 급감 → 시총 데이터 미수집 = data
  · 예) 'per < 3' 같은 값 조건에서 점진 감소 → 기준이 엄격해 적은 것 = none
- 이상치(ROE>200%·시총>3000조 등): SQL 가드(예: ROE는 순이익<자기자본)로 거를 수 있으면 sql,
  데이터 원본 자체가 틀린 단위·오류면 data.
- 개수 부족(요청 N개 > 결과): 분해표로 'data 때문'인지 '원래 그만큼뿐(none)'인지 구분.

[질문]
{question}

[생성된 SQL]
{sql}

[결과 상태] status={status}, 행수={row_count}, 기대개수={expected}

[코드가 수집한 증거]
{evidence}

JSON으로만 답하세요:
{{"cause": "sql|refine|data|none", "fixable": true/false, "explanation": "<한국어 원인 설명, 어느 증거가 근거인지>", "fix_hint": "<sql/refine이면 고칠 방향, 아니면 빈 문자열>"}}"""
