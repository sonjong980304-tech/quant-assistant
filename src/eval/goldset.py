"""정답셋 (financials/prices 원본 스키마 기준으로 재작성).

- 각 항목: {id, question, sql, tags}
- 정답 SQL은 metrics 사전계산 테이블이 아니라, SQL 생성 프롬프트(graph/prompts.py)와
  동일하게 financials/prices에서 직접 계산 + 동일한 이상치 가드를 적용한다.
  (예전 버전은 metrics 테이블을 참조했는데, 실제 SQL 생성 경로는 QUERYABLE_TABLES에
  metrics가 없어 financials/prices만 쓰므로 정의가 어긋나 있었다.)
- company.sector는 KRX 업종분류(29종, scripts/backfill_sector_krx.py로 적재) 기준이다.
  sector 조건은 LIKE 포괄 매칭을 쓴다('금융' → '금융'+'기타금융' 포괄).
  KRX 분류엔 '반도체'·'게임'·'전자부품' 업종이 없으므로(각각 '전기·전자',
  'IT 서비스'/'오락·문화'에 흡수) 해당 문항은 실존 KRX 업종으로 교체했다.
- ROE 의존 문항(id 10,11,12,13,35,41,44,49)은 DART 원문 재수집으로
  controlling_net_income(지배기업소유주지분순이익)이 백필된 뒤 GOLDSET에 병합했다.
  ROE = controlling_net_income(TTM)/controlling_equity — 비지배지분이 섞인
  net_income/total_equity가 아니라 지배기업 주주 귀속분끼리 맞춘 값이다
  (src/graph/prompts.py의 SQL 생성 프롬프트와 동일한 정의).
"""
from __future__ import annotations

GOLDSET: list[dict] = [
    # ---------- PER ----------
    {"id": 1, "question": "PER이 가장 낮은 10개 회사를 알려줘",
     "sql": """SELECT c.name, ROUND(p.market_cap/f.ni_ttm,2) AS per FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
  JOIN company c ON c.stock_code=f.stock_code
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap
ORDER BY per ASC LIMIT 10""", "tags": "PER"},
    {"id": 2, "question": "PER이 가장 높은 5개 회사",
     "sql": """SELECT c.name, ROUND(p.market_cap/f.ni_ttm,2) AS per FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
  JOIN company c ON c.stock_code=f.stock_code
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap
ORDER BY per DESC LIMIT 5""", "tags": "PER"},
    {"id": 3, "question": "PER이 낮은 상위 3개 종목",
     "sql": """SELECT c.name, ROUND(p.market_cap/f.ni_ttm,2) AS per FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
  JOIN company c ON c.stock_code=f.stock_code
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap
ORDER BY per ASC LIMIT 3""", "tags": "PER"},
    {"id": 4, "question": "PER이 10 이하인 회사는 몇 개야?",
     "sql": """SELECT COUNT(*) AS cnt FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap
  AND p.market_cap/f.ni_ttm <= 10""", "tags": "PER,집계"},
    {"id": 5, "question": "전기·전자 업종에서 PER이 가장 낮은 5개 회사",
     "sql": """SELECT c.name, ROUND(p.market_cap/f.ni_ttm,2) AS per FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
  JOIN company c ON c.stock_code=f.stock_code
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap
  AND c.sector LIKE '%전기·전자%'
ORDER BY per ASC LIMIT 5""", "tags": "PER,업종"},

    # ---------- PBR ----------
    {"id": 6, "question": "PBR이 가장 낮은 10개 회사",
     "sql": """SELECT c.name, ROUND(p.market_cap/e.eq,2) AS pbr FROM (
  SELECT stock_code, amount eq FROM financials
  WHERE account_key='total_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
JOIN prices p ON p.stock_code=e.stock_code AND p.date=(SELECT MAX(date) FROM prices)
JOIN company c ON c.stock_code=e.stock_code
WHERE e.eq>0 ORDER BY pbr ASC LIMIT 10""", "tags": "PBR"},
    {"id": 7, "question": "PBR이 가장 높은 5개 회사",
     "sql": """SELECT c.name, ROUND(p.market_cap/e.eq,2) AS pbr FROM (
  SELECT stock_code, amount eq FROM financials
  WHERE account_key='total_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
JOIN prices p ON p.stock_code=e.stock_code AND p.date=(SELECT MAX(date) FROM prices)
JOIN company c ON c.stock_code=e.stock_code
WHERE e.eq>0 ORDER BY pbr DESC LIMIT 5""", "tags": "PBR"},
    {"id": 8, "question": "PBR이 1 미만인 저평가 회사 목록",
     "sql": """SELECT c.name, ROUND(p.market_cap/e.eq,2) AS pbr FROM (
  SELECT stock_code, amount eq FROM financials
  WHERE account_key='total_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
JOIN prices p ON p.stock_code=e.stock_code AND p.date=(SELECT MAX(date) FROM prices)
JOIN company c ON c.stock_code=e.stock_code
WHERE e.eq>0 AND p.market_cap/e.eq < 1 ORDER BY pbr ASC LIMIT 20""", "tags": "PBR"},
    {"id": 9, "question": "금융 업종에서 PBR이 가장 낮은 3개 회사",
     "sql": """SELECT c.name, ROUND(p.market_cap/e.eq,2) AS pbr FROM (
  SELECT stock_code, amount eq FROM financials
  WHERE account_key='total_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
JOIN prices p ON p.stock_code=e.stock_code AND p.date=(SELECT MAX(date) FROM prices)
JOIN company c ON c.stock_code=e.stock_code
WHERE e.eq>0 AND c.sector LIKE '%금융%' ORDER BY pbr ASC LIMIT 3""", "tags": "PBR,업종"},

    # ---------- ROE ----------
    {"id": 10, "question": "ROE가 가장 높은 10개 회사",
     "sql": """SELECT c.name, ROUND(ni.ttm*100.0/e.eq,2) AS roe FROM (
  SELECT stock_code, SUM(amount) ttm FROM financials
  WHERE account_key='controlling_net_income'
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code) ni
JOIN (SELECT stock_code, amount eq FROM financials
      WHERE account_key='controlling_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
  ON e.stock_code=ni.stock_code
JOIN company c ON c.stock_code=ni.stock_code
JOIN prices p ON p.stock_code=ni.stock_code AND p.date=(SELECT MAX(date) FROM prices)
WHERE ni.ttm>0 AND e.eq>0 AND ni.ttm < e.eq
ORDER BY roe DESC LIMIT 10""", "tags": "ROE"},
    {"id": 11, "question": "ROE가 가장 낮은 5개 회사",
     "sql": """SELECT c.name, ROUND(ni.ttm*100.0/e.eq,2) AS roe FROM (
  SELECT stock_code, SUM(amount) ttm FROM financials
  WHERE account_key='controlling_net_income'
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code) ni
JOIN (SELECT stock_code, amount eq FROM financials
      WHERE account_key='controlling_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
  ON e.stock_code=ni.stock_code
JOIN company c ON c.stock_code=ni.stock_code
JOIN prices p ON p.stock_code=ni.stock_code AND p.date=(SELECT MAX(date) FROM prices)
WHERE e.eq>0 AND ni.ttm < e.eq
ORDER BY roe ASC LIMIT 5""", "tags": "ROE"},
    {"id": 12, "question": "ROE가 10% 이상인 회사 수",
     "sql": """SELECT COUNT(*) AS cnt FROM (
  SELECT stock_code, SUM(amount) ttm FROM financials
  WHERE account_key='controlling_net_income'
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code) ni
JOIN (SELECT stock_code, amount eq FROM financials
      WHERE account_key='controlling_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
  ON e.stock_code=ni.stock_code
WHERE e.eq>0 AND ni.ttm < e.eq AND ni.ttm*100.0/e.eq >= 10""", "tags": "ROE,집계"},
    {"id": 13, "question": "운송장비·부품 업종에서 ROE가 가장 높은 3개 회사",
     "sql": """SELECT c.name, ROUND(ni.ttm*100.0/e.eq,2) AS roe FROM (
  SELECT stock_code, SUM(amount) ttm FROM financials
  WHERE account_key='controlling_net_income'
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code) ni
JOIN (SELECT stock_code, amount eq FROM financials
      WHERE account_key='controlling_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
  ON e.stock_code=ni.stock_code
JOIN company c ON c.stock_code=ni.stock_code
JOIN prices p ON p.stock_code=ni.stock_code AND p.date=(SELECT MAX(date) FROM prices)
WHERE ni.ttm>0 AND e.eq>0 AND ni.ttm < e.eq
  AND c.sector LIKE '%운송장비·부품%'
ORDER BY roe DESC LIMIT 3""", "tags": "ROE,업종"},

    # ---------- 영업이익률 ----------
    {"id": 14, "question": "영업이익률이 가장 높은 10개 회사",
     "sql": """SELECT c.name, ROUND(op*100.0/rev,2) AS operating_margin FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='operating_profit' THEN amount END) op,
    MAX(CASE WHEN account_key='revenue' THEN amount END) rev
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code
WHERE rev>0 AND op IS NOT NULL AND op < rev ORDER BY operating_margin DESC LIMIT 10""", "tags": "영업이익률"},
    {"id": 15, "question": "영업이익률이 가장 낮은 5개 회사",
     "sql": """SELECT c.name, ROUND(op*100.0/rev,2) AS operating_margin FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='operating_profit' THEN amount END) op,
    MAX(CASE WHEN account_key='revenue' THEN amount END) rev
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code
WHERE rev>0 AND op IS NOT NULL AND op < rev ORDER BY operating_margin ASC LIMIT 5""", "tags": "영업이익률"},
    {"id": 16, "question": "영업이익률이 15% 이상인 회사 목록",
     "sql": """SELECT c.name, ROUND(op*100.0/rev,2) AS operating_margin FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='operating_profit' THEN amount END) op,
    MAX(CASE WHEN account_key='revenue' THEN amount END) rev
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code
WHERE rev>0 AND op IS NOT NULL AND op < rev AND op*100.0/rev >= 15
ORDER BY operating_margin DESC LIMIT 20""", "tags": "영업이익률"},

    # ---------- 부채비율 ----------
    {"id": 17, "question": "부채비율이 가장 높은 10개 회사",
     "sql": """SELECT c.name, ROUND(l*100.0/e,2) AS debt_ratio FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='total_liabilities' THEN amount END) l,
    MAX(CASE WHEN account_key='total_equity' THEN amount END) e
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code
WHERE e>0 ORDER BY debt_ratio DESC LIMIT 10""", "tags": "부채"},
    {"id": 18, "question": "부채비율이 가장 낮은 5개 회사",
     "sql": """SELECT c.name, ROUND(l*100.0/e,2) AS debt_ratio FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='total_liabilities' THEN amount END) l,
    MAX(CASE WHEN account_key='total_equity' THEN amount END) e
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code
WHERE e>0 ORDER BY debt_ratio ASC LIMIT 5""", "tags": "부채"},
    {"id": 19, "question": "부채비율이 200%를 초과하는 회사 목록",
     "sql": """SELECT c.name, ROUND(l*100.0/e,2) AS debt_ratio FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='total_liabilities' THEN amount END) l,
    MAX(CASE WHEN account_key='total_equity' THEN amount END) e
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code
WHERE e>0 AND l*100.0/e > 200 ORDER BY debt_ratio DESC LIMIT 20""", "tags": "부채"},
    {"id": 20, "question": "부채비율이 100% 미만인 안정적인 회사 수",
     "sql": """SELECT COUNT(*) AS cnt FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='total_liabilities' THEN amount END) l,
    MAX(CASE WHEN account_key='total_equity' THEN amount END) e
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f WHERE e>0 AND l*100.0/e < 100""", "tags": "부채,집계"},

    # ---------- 시가총액 ----------
    {"id": 21, "question": "시가총액이 가장 큰 10개 회사",
     "sql": "SELECT c.name, p.market_cap FROM prices p JOIN company c ON c.stock_code=p.stock_code WHERE p.date=(SELECT MAX(date) FROM prices) AND p.market_cap IS NOT NULL AND p.market_cap>0 ORDER BY p.market_cap DESC LIMIT 10", "tags": "시가총액"},
    {"id": 22, "question": "시가총액이 가장 작은 5개 회사",
     "sql": "SELECT c.name, p.market_cap FROM prices p JOIN company c ON c.stock_code=p.stock_code WHERE p.date=(SELECT MAX(date) FROM prices) AND p.market_cap IS NOT NULL AND p.market_cap>0 ORDER BY p.market_cap ASC LIMIT 5", "tags": "시가총액"},
    {"id": 23, "question": "시가총액 상위 3개 회사의 이름과 종가",
     "sql": "SELECT c.name, p.close, p.market_cap FROM prices p JOIN company c ON c.stock_code=p.stock_code WHERE p.date=(SELECT MAX(date) FROM prices) AND p.market_cap IS NOT NULL AND p.market_cap>0 ORDER BY p.market_cap DESC LIMIT 3", "tags": "시가총액,종가"},

    # ---------- 매출/영업이익/순이익 (financials) ----------
    {"id": 24, "question": "최근 분기 매출액이 가장 큰 10개 회사",
     "sql": "SELECT c.name, f.amount FROM financials f JOIN company c ON c.stock_code=f.stock_code WHERE f.account_key='revenue' AND f.quarter=(SELECT MAX(quarter) FROM financials) AND f.amount>0 ORDER BY f.amount DESC LIMIT 10", "tags": "매출"},
    {"id": 25, "question": "최근 분기 영업이익이 가장 큰 5개 회사",
     "sql": """SELECT c.name, op AS operating_profit FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='operating_profit' THEN amount END) op,
    MAX(CASE WHEN account_key='revenue' THEN amount END) rev
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code
WHERE op>0 AND rev>0 AND op<rev ORDER BY op DESC LIMIT 5""", "tags": "영업이익"},
    {"id": 26, "question": "최근 분기 당기순이익이 가장 큰 10개 회사",
     "sql": """SELECT c.name, ni AS net_income FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='net_income' THEN amount END) ni,
    MAX(CASE WHEN account_key='operating_profit' THEN amount END) op,
    MAX(CASE WHEN account_key='revenue' THEN amount END) rev
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code
WHERE ni>0 AND rev>0 AND op<rev ORDER BY ni DESC LIMIT 10""", "tags": "순이익"},
    {"id": 27, "question": "최근 분기 당기순이익이 적자인(0 미만) 회사 목록",
     "sql": "SELECT c.name, f.amount FROM financials f JOIN company c ON c.stock_code=f.stock_code WHERE f.account_key='net_income' AND f.quarter=(SELECT MAX(quarter) FROM financials) AND f.amount < 0 ORDER BY f.amount ASC LIMIT 20", "tags": "순이익"},
    {"id": 28, "question": "최근 분기 매출액이 가장 작은 5개 회사",
     "sql": "SELECT c.name, f.amount FROM financials f JOIN company c ON c.stock_code=f.stock_code WHERE f.account_key='revenue' AND f.quarter=(SELECT MAX(quarter) FROM financials) AND f.amount>0 ORDER BY f.amount ASC LIMIT 5", "tags": "매출"},

    # ---------- 업종 / 회사 메타 ----------
    {"id": 29, "question": "금융 업종 회사들의 이름을 알려줘",
     "sql": "SELECT name FROM company WHERE sector LIKE '%금융%' ORDER BY name LIMIT 20", "tags": "업종"},
    {"id": 30, "question": "업종이 몇 개나 있어?",
     "sql": "SELECT COUNT(DISTINCT sector) AS cnt FROM company WHERE sector != ''", "tags": "업종,집계"},
    {"id": 31, "question": "업종별 회사 수를 많은 순으로 알려줘",
     "sql": "SELECT sector, COUNT(*) AS cnt FROM company WHERE sector != '' GROUP BY sector ORDER BY cnt DESC LIMIT 20", "tags": "업종,집계"},
    {"id": 32, "question": "제약 업종 회사 목록",
     "sql": "SELECT name FROM company WHERE sector LIKE '%제약%' ORDER BY name LIMIT 20", "tags": "업종"},

    # ---------- 특정 회사 ----------
    {"id": 33, "question": "삼성전자의 PER은 얼마야?",
     "sql": """SELECT c.name, ROUND(p.market_cap/f.ni_ttm,2) AS per FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
  JOIN company c ON c.stock_code=f.stock_code
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap
  AND c.name='삼성전자'""", "tags": "PER,회사"},
    {"id": 34, "question": "기아의 부채비율을 알려줘",
     "sql": """SELECT c.name, ROUND(l*100.0/e,2) AS debt_ratio FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='total_liabilities' THEN amount END) l,
    MAX(CASE WHEN account_key='total_equity' THEN amount END) e
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code WHERE c.name='기아'""", "tags": "부채,회사"},
    {"id": 35, "question": "현대자동차의 ROE와 영업이익률",
     "sql": """SELECT c.name, ROUND(ni.ttm*100.0/e.eq,2) AS roe, ROUND(om.op*100.0/om.rev,2) AS operating_margin
FROM company c
JOIN (SELECT stock_code, SUM(amount) ttm FROM financials
      WHERE account_key='controlling_net_income'
        AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
      GROUP BY stock_code) ni ON ni.stock_code=c.stock_code
JOIN (SELECT stock_code, amount eq FROM financials
      WHERE account_key='controlling_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
  ON e.stock_code=c.stock_code
JOIN (SELECT stock_code,
        MAX(CASE WHEN account_key='operating_profit' THEN amount END) op,
        MAX(CASE WHEN account_key='revenue' THEN amount END) rev
      FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code) om
  ON om.stock_code=c.stock_code
WHERE c.name='현대자동차'""", "tags": "ROE,영업이익률,회사"},
    {"id": 36, "question": "NAVER의 시가총액은?",
     "sql": "SELECT c.name, p.market_cap FROM prices p JOIN company c ON c.stock_code=p.stock_code WHERE c.name='NAVER' AND p.date=(SELECT MAX(date) FROM prices)", "tags": "시가총액,회사"},
    {"id": 37, "question": "삼성전자의 최근 분기 매출액",
     "sql": "SELECT c.name, f.amount FROM financials f JOIN company c ON c.stock_code=f.stock_code WHERE c.name='삼성전자' AND f.account_key='revenue' AND f.quarter=(SELECT MAX(quarter) FROM financials)", "tags": "매출,회사"},

    # ---------- 집계 / 고급 (LLM 변별) ----------
    {"id": 38, "question": "전체 회사의 평균 PER은?",
     "sql": """SELECT ROUND(AVG(p.market_cap/f.ni_ttm),2) AS avg_per FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap""", "tags": "PER,집계"},
    {"id": 39, "question": "업종별 평균 PER을 낮은 순으로",
     "sql": """SELECT c.sector, ROUND(AVG(p.market_cap/f.ni_ttm),2) AS avg_per FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
  JOIN company c ON c.stock_code=f.stock_code
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap
GROUP BY c.sector ORDER BY avg_per ASC LIMIT 20""", "tags": "PER,업종,집계"},
    {"id": 40, "question": "업종별 평균 부채비율을 높은 순으로",
     "sql": """SELECT c.sector, ROUND(AVG(l*100.0/e),2) AS avg_debt FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='total_liabilities' THEN amount END) l,
    MAX(CASE WHEN account_key='total_equity' THEN amount END) e
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code
WHERE e>0 GROUP BY c.sector ORDER BY avg_debt DESC LIMIT 20""", "tags": "부채,업종,집계"},
    {"id": 41, "question": "ROE가 평균보다 높은 회사 목록",
     "sql": """SELECT c.name, ROUND(ni.ttm*100.0/e.eq,2) AS roe FROM (
  SELECT stock_code, SUM(amount) ttm FROM financials
  WHERE account_key='controlling_net_income'
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code) ni
JOIN (SELECT stock_code, amount eq FROM financials
      WHERE account_key='controlling_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
  ON e.stock_code=ni.stock_code
JOIN company c ON c.stock_code=ni.stock_code
JOIN prices p ON p.stock_code=ni.stock_code AND p.date=(SELECT MAX(date) FROM prices)
WHERE ni.ttm>0 AND e.eq>0 AND ni.ttm < e.eq
  AND ni.ttm*100.0/e.eq > (
    SELECT AVG(ni2.ttm*100.0/e2.eq) FROM (
      SELECT stock_code, SUM(amount) ttm FROM financials
      WHERE account_key='controlling_net_income'
        AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
      GROUP BY stock_code) ni2
    JOIN (SELECT stock_code, amount eq FROM financials
          WHERE account_key='controlling_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e2
      ON e2.stock_code=ni2.stock_code
    WHERE ni2.ttm>0 AND e2.eq>0 AND ni2.ttm < e2.eq
  )
ORDER BY roe DESC LIMIT 20""", "tags": "ROE,집계"},
    {"id": 42, "question": "PER이 가장 낮은 회사 한 곳",
     "sql": """SELECT c.name, ROUND(p.market_cap/f.ni_ttm,2) AS per FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
  JOIN company c ON c.stock_code=f.stock_code
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap
ORDER BY per ASC LIMIT 1""", "tags": "PER"},
    {"id": 43, "question": "영업이익률이 가장 높은 회사 한 곳은 어디야?",
     "sql": """SELECT c.name, ROUND(op*100.0/rev,2) AS operating_margin FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='operating_profit' THEN amount END) op,
    MAX(CASE WHEN account_key='revenue' THEN amount END) rev
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code
WHERE rev>0 AND op IS NOT NULL AND op < rev ORDER BY operating_margin DESC LIMIT 1""", "tags": "영업이익률"},
    {"id": 44, "question": "PBR이 1 미만이면서 ROE가 5% 이상인 회사",
     "sql": """SELECT c.name, ROUND(p.market_cap/e.eq,2) AS pbr, ROUND(ni.ttm*100.0/ce.eq,2) AS roe FROM (
  SELECT stock_code, SUM(amount) ttm FROM financials
  WHERE account_key='controlling_net_income'
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code) ni
JOIN (SELECT stock_code, amount eq FROM financials
      WHERE account_key='total_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
  ON e.stock_code=ni.stock_code
JOIN (SELECT stock_code, amount eq FROM financials
      WHERE account_key='controlling_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) ce
  ON ce.stock_code=ni.stock_code
JOIN prices p ON p.stock_code=ni.stock_code AND p.date=(SELECT MAX(date) FROM prices)
JOIN company c ON c.stock_code=ni.stock_code
WHERE e.eq>0 AND ce.eq>0 AND ni.ttm>0 AND ni.ttm < ce.eq
  AND p.market_cap/e.eq < 1 AND ni.ttm*100.0/ce.eq >= 5
ORDER BY roe DESC LIMIT 20""", "tags": "PBR,ROE"},
    {"id": 45, "question": "PER이 낮은 순으로 5개, 회사명과 PER, PBR을 함께",
     "sql": """SELECT c.name, ROUND(p.market_cap/f.ni_ttm,2) AS per, ROUND(p.market_cap/e.eq,2) AS pbr FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
  JOIN company c ON c.stock_code=f.stock_code
  JOIN (SELECT stock_code, amount eq FROM financials WHERE account_key='total_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e ON e.stock_code=f.stock_code
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap AND e.eq>0
ORDER BY per ASC LIMIT 5""", "tags": "PER,PBR"},
    {"id": 46, "question": "시가총액이 1조원 이상인 회사 수",
     "sql": "SELECT COUNT(*) AS cnt FROM prices WHERE date=(SELECT MAX(date) FROM prices) AND market_cap >= 1000000000000", "tags": "시가총액,집계"},
    {"id": 47, "question": "화학 업종의 평균 영업이익률",
     "sql": """SELECT ROUND(AVG(op*100.0/rev),2) AS avg_om FROM (
  SELECT stock_code,
    MAX(CASE WHEN account_key='operating_profit' THEN amount END) op,
    MAX(CASE WHEN account_key='revenue' THEN amount END) rev
  FROM financials WHERE quarter=(SELECT MAX(quarter) FROM financials) GROUP BY stock_code
) f JOIN company c ON c.stock_code=f.stock_code
WHERE rev>0 AND op IS NOT NULL AND op < rev AND c.sector LIKE '%화학%'""", "tags": "영업이익률,업종,집계"},
    {"id": 48, "question": "당기순이익이 흑자인 회사 수",
     "sql": "SELECT COUNT(*) AS cnt FROM financials WHERE account_key='net_income' AND quarter=(SELECT MAX(quarter) FROM financials) AND amount > 0", "tags": "순이익,집계"},
    {"id": 49, "question": "ROE 상위 3개 회사의 평균 PER",
     "sql": """SELECT ROUND(AVG(p.market_cap/f.ni_ttm),2) AS avg_per FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f
JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
JOIN (
  SELECT ni.stock_code FROM (
    SELECT stock_code, SUM(amount) ttm FROM financials
    WHERE account_key='controlling_net_income'
      AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
    GROUP BY stock_code) ni
  JOIN (SELECT stock_code, amount eq FROM financials
        WHERE account_key='controlling_equity' AND quarter=(SELECT MAX(quarter) FROM financials)) e
    ON e.stock_code=ni.stock_code
  JOIN prices p2 ON p2.stock_code=ni.stock_code AND p2.date=(SELECT MAX(date) FROM prices)
  WHERE ni.ttm>0 AND e.eq>0 AND ni.ttm < e.eq
  ORDER BY ni.ttm*100.0/e.eq DESC LIMIT 3
) top3 ON top3.stock_code=f.stock_code
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap""", "tags": "ROE,PER,집계"},
    {"id": 50, "question": "오락·문화 업종에서 PER이 가장 낮은 회사",
     "sql": """SELECT c.name, ROUND(p.market_cap/f.ni_ttm,2) AS per FROM (
  SELECT stock_code,
    SUM(CASE WHEN account_key='net_income' THEN amount END) ni_ttm,
    SUM(CASE WHEN account_key='operating_profit' THEN amount END) op_ttm
  FROM financials
  WHERE account_key IN ('net_income','operating_profit')
    AND quarter IN (SELECT DISTINCT quarter FROM financials ORDER BY quarter DESC LIMIT 4)
  GROUP BY stock_code
) f JOIN prices p ON p.stock_code=f.stock_code AND p.date=(SELECT MAX(date) FROM prices)
  JOIN company c ON c.stock_code=f.stock_code
WHERE f.ni_ttm>0 AND f.op_ttm>0 AND f.ni_ttm < f.op_ttm AND f.ni_ttm < p.market_cap
  AND c.sector LIKE '%오락·문화%'
ORDER BY per ASC LIMIT 1""", "tags": "PER,업종"},

]


def validate_goldset(db_path: str | None = None) -> dict:
    """정답 SQL이 모두 실행 가능하고 비어있지 않은지 검증."""
    from ..db import connect
    from ..sql_exec import run_select

    conn = connect(db_path)
    try:
        ok, empty, failed = 0, [], []
        for item in GOLDSET:
            r = run_select(conn, item["sql"])
            if not r["ok"]:
                failed.append((item["id"], r["error"]))
            elif r["row_count"] == 0:
                empty.append(item["id"])
            else:
                ok += 1
        return {"total": len(GOLDSET), "ok": ok, "empty": empty, "failed": failed}
    finally:
        conn.close()


if __name__ == "__main__":
    print(validate_goldset())
