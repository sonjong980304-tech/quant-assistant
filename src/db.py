"""SQLite 연결/초기화 + 스키마 카탈로그(Text-to-SQL 프롬프트용).

컬럼명은 영어 snake_case로 두되, 각 컬럼의 한글 의미를 schema_catalog로
LLM에 제공한다. 한글 컬럼명은 SQL 인용 처리 이슈가 있어 피한다.

Phase1에서 광범위 계정/지표 컬럼/백테스트 테이블로 확장됨. 기존 DB는
init_db()의 _migrate()가 ALTER ADD COLUMN으로 무중단 이행한다.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import CONFIG

# ---------------------------------------------------------------------------
# 정규화된 계정 키 (회사별 비표준 계정과목명 → 표준 키). 백테스트 지표 산출용 광범위 수집.
# ---------------------------------------------------------------------------
ACCOUNT_KEYS = {
    "revenue": "매출액(영업수익)",
    "cost_of_sales": "매출원가",
    "gross_profit": "매출총이익",
    "sga": "판매비와관리비",
    "operating_profit": "영업이익",
    "non_operating_income": "영업외손익",
    "interest_expense": "이자비용",
    "tax_expense": "법인세비용",
    "net_income": "당기순이익",
    "controlling_net_income": "지배기업소유주귀속 당기순이익",
    "current_assets": "유동자산",
    "non_current_assets": "비유동자산",
    "current_liabilities": "유동부채",
    "non_current_liabilities": "비유동부채",
    "total_assets": "자산총계",
    "total_liabilities": "부채총계",
    "total_equity": "자본총계",
    "controlling_equity": "지배기업소유주지분",
    "cash": "현금및현금성자산",
    "depreciation": "감가상각비",
    "operating_cashflow": "영업활동현금흐름",
    "shares_outstanding": "발행주식수",
    "dividend": "배당금",
}

# metrics 파생 지표 컬럼 (key, SQL타입). per/pbr/roe/operating_margin/debt_ratio는
# 최초 스키마에도 있으나 일괄 관리를 위해 여기 모은다.
METRIC_COLUMNS = {
    "market_cap": "REAL",
    "per": "REAL", "pbr": "REAL", "psr": "REAL", "pcr": "REAL",
    "ev_ebitda": "REAL", "peg": "REAL",
    "roe": "REAL", "roa": "REAL", "operating_margin": "REAL",
    "net_margin": "REAL", "gp_a": "REAL",
    "debt_ratio": "REAL", "current_ratio": "REAL", "interest_coverage": "REAL",
    "revenue_growth": "REAL", "op_growth": "REAL", "ni_growth": "REAL",
    "dividend_yield": "REAL", "momentum": "REAL",
}

# metric_def 시드 (UI 자동생성용): (key, label, category, direction, description)
# direction: 'low'=낮을수록 우수, 'high'=높을수록 우수, 'neutral'=방향선택
METRIC_DEFS = [
    ("per", "PER", "밸류", "low", "주가수익비율 = 시총/당기순이익(TTM)"),
    ("pbr", "PBR", "밸류", "low", "주가순자산비율 = 시총/자본총계"),
    ("psr", "PSR", "밸류", "low", "주가매출비율 = 시총/매출(TTM)"),
    ("pcr", "PCR", "밸류", "low", "주가현금흐름비율 = 시총/영업활동현금흐름(TTM)"),
    ("ev_ebitda", "EV/EBITDA", "밸류", "low", "기업가치/EBITDA"),
    ("peg", "PEG", "밸류", "low", "PER/순이익성장률"),
    ("roe", "ROE", "수익성", "high", "자기자본이익률(%) = 순이익(TTM)/자본총계"),
    ("roa", "ROA", "수익성", "high", "총자산이익률(%) = 순이익(TTM)/총자산"),
    ("operating_margin", "영업이익률", "수익성", "high", "영업이익/매출(%)"),
    ("net_margin", "순이익률", "수익성", "high", "순이익/매출(%)"),
    ("gp_a", "GP/A", "수익성", "high", "매출총이익/총자산(%)"),
    ("debt_ratio", "부채비율", "안정성", "low", "부채총계/자본총계(%)"),
    ("current_ratio", "유동비율", "안정성", "high", "유동자산/유동부채(%)"),
    ("interest_coverage", "이자보상배율", "안정성", "high", "영업이익/이자비용"),
    ("revenue_growth", "매출성장률", "성장", "high", "전년동기比 매출 증가율(%)"),
    ("op_growth", "영업이익성장률", "성장", "high", "전년동기比 영업이익 증가율(%)"),
    ("ni_growth", "순이익성장률", "성장", "high", "전년동기比 순이익 증가율(%)"),
    ("market_cap", "시가총액", "기타", "neutral", "시가총액(원)"),
    ("return_12m", "가격모멘텀", "기타", "high", "최근 12개월 주가 수익률(%)"),
]

_METRIC_BASE_COLS = (
    "market_cap REAL, per REAL, pbr REAL, psr REAL, pcr REAL, ev_ebitda REAL, peg REAL, "
    "roe REAL, roa REAL, operating_margin REAL, net_margin REAL, gp_a REAL, "
    "debt_ratio REAL, current_ratio REAL, interest_coverage REAL, "
    "revenue_growth REAL, op_growth REAL, ni_growth REAL, dividend_yield REAL, momentum REAL"
)

SCHEMA_DDL = f"""
CREATE TABLE IF NOT EXISTS company (
    stock_code     TEXT PRIMARY KEY,   -- 종목코드 (6자리 문자열, 예: '005930')
    name           TEXT NOT NULL,      -- 회사명 (예: '삼성전자')
    market         TEXT,               -- 시장구분 ('KOSPI' | 'KOSDAQ')
    sector         TEXT                -- 업종 (KRX 분류 29종, 예: '전기·전자','IT 서비스','화학','제약','기타금융')
);

CREATE TABLE IF NOT EXISTS financials (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code     TEXT NOT NULL,      -- 종목코드
    quarter        TEXT NOT NULL,      -- 기준분기 (예: '2025Q1')
    disclosed_date TEXT,               -- 공시일 (YYYY-MM-DD)
    account_key    TEXT NOT NULL,      -- 정규화 계정키 (revenue/operating_profit/net_income/total_assets/total_liabilities/total_equity 등)
    account_name   TEXT,               -- 원본 계정과목명 (회사별 비표준)
    amount         REAL,               -- 금액 (원 단위, 손익은 분기 단독)
    UNIQUE(stock_code, quarter, account_key)
);

CREATE TABLE IF NOT EXISTS prices (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code     TEXT NOT NULL,      -- 종목코드
    date           TEXT NOT NULL,      -- 날짜 (YYYY-MM-DD, 거래일)
    close          REAL,               -- 종가 (원)
    market_cap     REAL,               -- 시가총액 (원) = 종가 × 상장주식수
    open           REAL,               -- 시가 (원, 네이버 수정주가 기준)
    high           REAL,               -- 고가 (원, 네이버 수정주가 기준)
    low            REAL,               -- 저가 (원, 네이버 수정주가 기준)
    volume         REAL,               -- 거래량 (주)
    UNIQUE(stock_code, date)
);

-- FnGuide 재무지표(밸류에이션·수익성·컨센서스 목표주가 등, "다 가져오는" 폭넓은 지표라
-- 지표가 계속 늘어날 수 있음) — DART 계산치 전용 metrics 테이블과 완전히 분리된 EAV 스키마.
CREATE TABLE IF NOT EXISTS fnguide_metrics (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code     TEXT NOT NULL,      -- 종목코드
    as_of_date     TEXT NOT NULL,      -- 기준일 (YYYY-MM-DD)
    metric_key     TEXT NOT NULL,      -- 지표 키 (예: roe, per, consensus_target_price)
    metric_value   REAL,               -- 지표 값
    source         TEXT,               -- 출처 (예: 'fnguide')
    collected_at   TEXT,               -- 수집 시각 (ISO)
    UNIQUE(stock_code, as_of_date, metric_key)
);

CREATE TABLE IF NOT EXISTS metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code       TEXT NOT NULL,    -- 종목코드
    quarter          TEXT NOT NULL,    -- 재무 기준분기 (예: '2025Q1')
    price_date       TEXT,             -- 종가 기준일 (주가 기반 지표용, YYYY-MM-DD)
    {_METRIC_BASE_COLS},
    UNIQUE(stock_code, quarter, price_date)
);

-- 미국 시장 데이터 플레인 (한국용 company/prices/financials/metrics와 완전 분리,
-- 컬럼·제약 공유 없음. Text-to-SQL 연결은 별도 작업으로 미룸 — QUERYABLE_TABLES
-- 미포함). .omc/specs/brainstorming-us-market-data-plane.md 참고.
CREATE TABLE IF NOT EXISTS us_company (
    stock_code     TEXT PRIMARY KEY,   -- 티커 심볼 (예: 'AAPL')
    name           TEXT NOT NULL,      -- 회사명
    exchange       TEXT,               -- 'NASDAQ' | 'NYSE' | 'NYSE Amex'
    sector         TEXT,               -- investing.com 원본 taxonomy 그대로(재분류 없음)
    market_cap     REAL,               -- 시가총액 (달러, 파싱된 숫자)
    security_type  TEXT,               -- 증권종류 캐시('common'|'warrant'|'adr'|'preferred'|'unit'|'right'|'other'|NULL=미분류).
                                        -- scripts/backfill_us_security_type.py가 회사명을 LLM으로
                                        -- 배치 분류해 채운다(스크리닝 런타임은 이 캐시만 읽음).
    financial_currency TEXT,           -- us_financials 원본 보고통화(yfinance financialCurrency,
                                        -- 예:'USD'|'KRW'). 주가(us_prices)는 항상 달러 거래지만
                                        -- 외국기업 ADR은 재무제표 자체를 본국통화로 보고하는 경우가
                                        -- 있어(예: SK텔레콤 SKM='KRW') 둘이 다를 수 있다. NULL=미수집.
                                        -- scripts/backfill_us_financial_currency.py가 채운다.
    updated_at     TEXT                -- 마지막 수집 시각 (ISO)
);

CREATE TABLE IF NOT EXISTS us_prices (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code     TEXT NOT NULL,      -- 티커 심볼
    date           TEXT NOT NULL,      -- 날짜 (YYYY-MM-DD)
    open           REAL,               -- 시가 (yfinance 수정주가 기준)
    high           REAL,               -- 고가
    low            REAL,               -- 저가
    close          REAL,               -- 종가
    volume         REAL,               -- 거래량
    UNIQUE(stock_code, date)
);

CREATE TABLE IF NOT EXISTS us_financials (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code     TEXT NOT NULL,      -- 티커 심볼
    as_of_date     TEXT NOT NULL,      -- 기준일 (YYYY-MM-DD)
    period_type    TEXT NOT NULL,      -- 'annual' | 'quarterly'
    statement_type TEXT NOT NULL,      -- 'income_stmt' | 'balance_sheet' | 'cashflow'
    item_key       TEXT NOT NULL,      -- yfinance 원본 항목명 (예: 'Total Revenue')
    item_value     REAL,               -- 항목 값
    disclosed_date TEXT,               -- 공시일 근사 (YYYY-MM-DD, quarterly=기말+45일/annual=기말+90일, look-ahead 방지용)
    source         TEXT,               -- 출처 (예: 'yfinance')
    collected_at   TEXT,               -- 수집 시각 (ISO)
    UNIQUE(stock_code, as_of_date, period_type, statement_type, item_key)
);

-- SEC EDGAR XBRL 재무데이터 플레인 (yfinance 기반 us_financials와 완전 분리된 신규 테이블).
-- SEC companyfacts(무료 공식)의 원시 XBRL 팩트를 축소 없이 그대로 저장한다 — 계산에 당장
-- 쓰는 15~20개로 줄이지 않고(향후 신규 지표 추가 시 재수집 불필요), 태그·값·단위·회계연도(fy)·
-- 회계분기(fp)·양식(form)·제출일(filed)·기간(start/end)·프레임·접수번호까지 보관한다.
-- disclosed_date 근사(us_financials의 기말+45/90일)와 달리 SEC 실제 filed(제출일)를 그대로
-- 저장해 look-ahead(미래참조) 방지를 더 정확히 한다. 자연어 SQL 노출 대상 아님(QUERYABLE_TABLES
-- 미포함 — metrics/macro와 동일 관례). .omc/specs/brainstorming-sec-edgar-us-financials-backfill.md 참고.
CREATE TABLE IF NOT EXISTS us_financials_sec (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code     TEXT NOT NULL,      -- 티커 심볼 (us_company와 연결)
    cik            TEXT NOT NULL,      -- SEC 회사고유번호 (10자리 zero-pad, 예: '0000320193')
    tag            TEXT NOT NULL,      -- XBRL 태그명 (예: 'Revenues', 'NetIncomeLoss', 'Assets')
    taxonomy       TEXT,               -- 분류체계 ('us-gaap' | 'dei' | 'ifrs-full' 등)
    unit           TEXT,               -- 단위 ('USD' | 'shares' | 'USD/shares' 등)
    value          REAL,               -- 팩트 값
    period_start   TEXT,               -- 기간 시작일 (duration 팩트, YYYY-MM-DD; instant 팩트는 NULL)
    period_end     TEXT NOT NULL,      -- 기간 종료일 / 기준일 (YYYY-MM-DD) — as_of_date에 대응
    fy             INTEGER,            -- 회계연도 (예: 2009)
    fp             TEXT,               -- 회계분기 ('Q1'|'Q2'|'Q3'|'FY')
    form           TEXT,               -- 제출 양식 ('10-Q'|'10-K'|'10-K/A'|'10-Q/A' 등)
    filed          TEXT,               -- 실제 제출일 (YYYY-MM-DD) — disclosed_date로 사용(look-ahead 방지)
    frame          TEXT,               -- CY 프레임 식별자 (있을 때만, 예: 'CY2009Q1')
    accn           TEXT,               -- 접수번호 (예: '0001193125-09-214859')
    source         TEXT,               -- 출처 ('sec_companyfacts_zip' | 'sec_companyfacts_api')
    collected_at   TEXT,               -- 수집 시각 (ISO)
    UNIQUE(stock_code, tag, unit, period_start, period_end, form, accn)
);

-- LLM Wiki: 질문 → SQL (SQL 캐시 / 시점 무관)
CREATE TABLE IF NOT EXISTS wiki (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    question           TEXT NOT NULL,
    raw_question       TEXT,
    question_embedding BLOB,
    sql                TEXT NOT NULL,
    route              TEXT,
    data_version       TEXT,
    result_json        TEXT,
    verified           INTEGER DEFAULT 0,
    tags               TEXT,
    use_count          INTEGER DEFAULT 0,
    model              TEXT,            -- 이 SQL을 생성한 LLM (모델별 캐시 분리; 검증된 항목은 모델 무관 공유)
    created_at         TEXT,
    updated_at         TEXT
);

-- 결과 캐시: (SQL + data_version) → 결과 (시점 의존)
CREATE TABLE IF NOT EXISTS result_cache (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sql_hash     TEXT NOT NULL,
    data_version TEXT NOT NULL,
    sql          TEXT,
    result_json  TEXT,
    row_count    INTEGER,
    created_at   TEXT,
    UNIQUE(sql_hash, data_version)
);

CREATE TABLE IF NOT EXISTS ingest_meta (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT
);

-- 상장폐지 (생존편향 제거용)
CREATE TABLE IF NOT EXISTS delisting (
    stock_code     TEXT PRIMARY KEY,   -- 종목코드
    name           TEXT,               -- 회사명
    delisting_date TEXT                -- 상장폐지일 (YYYY-MM-DD)
);

-- 미국 상장폐지 (구간 기반, 생존편향 검증용). KR delisting(stock_code 단일 PK + 날짜 하나)과
-- 달리 미국은 상장폐지된 티커가 나중에 다른 회사에 재사용되는 경우가 흔하다(예: TWTR). 단순
-- "delisting_date보다 뒤면 무조건 죽음"으로 판정하면 재상장 이후 시점을 오탐(잘못된 하드차단)
-- 한다 → 티커 하나에 여러 행(상장~폐지 구간 여러 개)을 허용하고, 판정은 asof가 어느 구간에
-- 드는지로 한다(data_access_us._is_alive_us). 소스=FMP delisted-companies(symbol/companyName/
-- exchange/ipoDate/delistedDate). 상장일 미상이면 listing_date=''(빈문자열 센티널) — SQLite
-- UNIQUE가 NULL을 매번 다른 값으로 취급해 upsert 멱등성이 깨지는 것을 막는다(us_financials_sec
-- period_start 관례와 동일). .omc/specs/brainstorming-us-delisting-survivorship.md 참고.
CREATE TABLE IF NOT EXISTS us_delisting (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code     TEXT NOT NULL,      -- 티커 심볼 (us_company와 간접 연결, PK 아님)
    company_name   TEXT,               -- 회사명 (FMP companyName)
    exchange       TEXT,               -- 거래소 (FMP exchange)
    listing_date   TEXT,               -- 상장일 (FMP ipoDate, YYYY-MM-DD; 미상이면 '')
    delisting_date TEXT,               -- 상장폐지일 (FMP delistedDate, YYYY-MM-DD)
    updated_at     TEXT,               -- 마지막 수집 시각 (ISO)
    UNIQUE(stock_code, listing_date, delisting_date)
);

-- KR 관리종목/매매거래정지 상태 이력 (구간 기반, 스냅샷 누적). KRX/KIND 는 과거 조회일을
-- 무시하고 '오늘 현재' 스냅샷만 반환해 과거 이력을 무료로 구할 수 없다 → 매 실행(매일 1회)
-- 마다 '오늘 현재' 관리종목/거래정지 목록을 받아 직전 실행 스냅샷과 diff 해서 앞으로의
-- 지정~해제 구간을 우리 쪽에서 누적으로 쌓는다. '직전 스냅샷'은 별도 저장 없이 이 테이블의
-- 열린 구간(end_date IS NULL)에서 유도한다: 현재 목록에 새로 나타난 종목=지정 개시(start_date=
-- 관측일, end_date=NULL), 사라진 종목=해제(end_date=관측일), 계속 있는 종목=구간 유지.
-- us_delisting 과 동일한 구간+멱등 upsert 사상 — 한 종목이 지정→해제→재지정을 반복하면
-- 여러 행(각 구간 1행)을 가진다. status_type 판별자로 관리종목/거래정지를 한 테이블에 담는다
-- (macro_indicators.indicator, us_financials.statement_type 판별자 관례와 동일 — diff 알고리즘이
-- 두 종류에 동일해 코드 중복을 없앤다). start_date 는 KRX 최초지정일이 아니라 '우리가 스냅샷에서
-- 처음 관측한 날'이다(과거 이력이 없으니 정직하게 관측 시점만 기록). KRX 원본 최초지정일(관리)/
-- 지정일시(정지)는 krx_designated_date 에 참고용으로 보관한다.
-- ⚠️ 이 데이터는 backtest look-ahead 경로(metrics_at 등)에 절대 연결하지 않는다 — 과거 이력이
-- 없어 연결하면 look-ahead/생존편향을 오히려 악화시킨다. '현재 시점' 라이브 필터링 전용
-- (is_currently_administrative_or_halted). 자연어 SQL 질의 대상 아님(QUERYABLE_TABLES 미포함).
CREATE TABLE IF NOT EXISTS kr_trading_status (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code         TEXT NOT NULL,   -- 종목코드 (6자리 단축코드, KRX ISU_SRT_CD/ISU_CD 정규화)
    status_type        TEXT NOT NULL,   -- 'admin'(관리종목) | 'halt'(매매거래정지)
    company_name       TEXT,            -- 종목명 (KRX ISU_NM)
    market             TEXT,            -- 시장구분 (KRX MKT_NM, 예: 'KOSPI'|'KOSDAQ'|'KONEX'; 정지목록은 없을 수 있음)
    reason             TEXT,            -- 지정/정지 사유 (관리=LIST_BZ_RSN_NM, 정지=HALT_RSN_NM)
    start_date         TEXT NOT NULL,   -- 상태 시작 관측일 (우리가 스냅샷에서 처음 관측한 날, YYYY-MM-DD)
    end_date           TEXT,            -- 상태 해제 관측일 (스냅샷에서 사라진 걸 관측한 날; NULL=현재 진행 중)
    krx_designated_date TEXT,           -- KRX 원본 최초지정일/지정일시의 날짜부 (참고용, YYYY-MM-DD; 미상 NULL)
    updated_at         TEXT,            -- 마지막 스냅샷 반영 시각 (ISO)
    UNIQUE(stock_code, status_type, start_date)
);

-- 지표 정의 (백테스트 UI 자동생성용)
CREATE TABLE IF NOT EXISTS metric_def (
    key         TEXT PRIMARY KEY,      -- metrics 컬럼명
    label       TEXT,                  -- 표시명 (PER 등)
    category    TEXT,                  -- 밸류/수익성/안정성/성장/기타
    direction   TEXT,                  -- 'low'|'high'|'neutral'
    description TEXT
);

-- 백테스트 실행 기록
CREATE TABLE IF NOT EXISTS backtest_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,                  -- 전략명
    params_json TEXT,                  -- 리밸런싱/종목수/지표선정/조합방식/산업필터 등
    cost_json   TEXT,                  -- 거래비용 설정
    start_year  INTEGER,
    end_year    INTEGER,
    result_json TEXT,                  -- CAGR/MDD/샤프/소르티노/승률/회전율 등
    created_at  TEXT
);

-- 재무제표 정정공시 이력 (append-only). financials 는 UNIQUE(stock_code,quarter,account_key)라
-- (종목,분기,계정)당 1행뿐이라 DART 정정공시(재무제표 재작성) 재수집 시 예전 값이 덮어써져
-- "정정 전엔 얼마였는지" 역사가 사라진다 → 백테스트가 과거 asof 시점에 "그때 알 수 있었던 값"
-- 대신 "나중에야 알 수 있었던 정정값"을 쓰는 look-ahead(미래참조) 편향. 이 테이블은 적재마다
-- 새 행을 INSERT(같은 rcept_no 재수집은 멱등)해 모든 버전을 보존한다. 백테스트 look-ahead
-- 크리티컬 경로만(effective_quarter_at/_fin/_sum_ttm/_yoy 등) 이 테이블에서 "disclosed_date<=asof
-- 중 가장 최근 disclosed_date" 버전을 고른다(SEC us_financials_sec 의 filed 상한과 동일 사상).
-- 기존 financials 테이블과 그 소비자(goldset/legacy/data_financial 등)는 전혀 건드리지 않는다.
-- rcept_no: DART 접수번호(공시 고유 id). 같은 공시 재수집=같은 rcept_no → UNIQUE 로 멱등,
-- 진짜 새 rcept_no(정정공시)만 새 행. 접수번호 미상(추정 공시일 폴백)이면 disclosed_date 를
-- 센티널로 넣어(us_delisting listing_date='' 관례와 동일) NULL 로 인한 멱등성 깨짐을 막는다.
CREATE TABLE IF NOT EXISTS financials_revision (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code     TEXT NOT NULL,      -- 종목코드
    quarter        TEXT NOT NULL,      -- 기준분기 (예: '2024Q1')
    disclosed_date TEXT,               -- 공시일 (YYYY-MM-DD, 이 버전이 공시된 날)
    account_key    TEXT NOT NULL,      -- 정규화 계정키
    account_name   TEXT,               -- 원본 계정과목명
    amount         REAL,               -- 금액 (원 단위)
    rcept_no       TEXT NOT NULL,      -- DART 접수번호(공시 고유 id) / 미상이면 공시일 센티널
    UNIQUE(stock_code, quarter, account_key, rcept_no)
);

CREATE INDEX IF NOT EXISTS idx_fin_code_q   ON financials(stock_code, quarter);
CREATE INDEX IF NOT EXISTS idx_fin_key      ON financials(account_key);
-- 값 조회(_fin/_sum_ttm/_yoy): (종목,분기,계정) 필터 후 disclosed_date DESC 로 asof 이하 최신 버전 선택.
CREATE INDEX IF NOT EXISTS idx_fin_rev_lookup ON financials_revision(stock_code, quarter, account_key, disclosed_date);
-- 유효분기 판정(effective_quarter_at/mode_financial_quarter_at): (종목) 필터 후 disclosed_date<=asof 중 최신 quarter.
CREATE INDEX IF NOT EXISTS idx_fin_rev_code_disc ON financials_revision(stock_code, disclosed_date, quarter);
CREATE INDEX IF NOT EXISTS idx_price_code_d ON prices(stock_code, date);
CREATE INDEX IF NOT EXISTS idx_price_date   ON prices(date);
CREATE INDEX IF NOT EXISTS idx_metrics_code ON metrics(stock_code, quarter);
CREATE INDEX IF NOT EXISTS idx_metrics_full ON metrics(stock_code, quarter, price_date);

-- 원본 재무제표 응답 보관 (재수집 없이 재파싱용). payload = zlib.compress(json.dumps(list)).
-- 새 계정이 필요해지면 재수집 대신 이 원본을 재파싱해 financials를 다시 만든다.
CREATE TABLE IF NOT EXISTS raw_reports (
    stock_code  TEXT NOT NULL,
    bsns_year   INTEGER NOT NULL,
    reprt_code  TEXT NOT NULL,      -- 11013/11012/11014/11011
    fs_div      TEXT NOT NULL,      -- CFS(연결) | OFS(별도)
    payload     BLOB,               -- zlib 압축된 원본 list JSON
    fetched_at  TEXT,
    PRIMARY KEY (stock_code, bsns_year, reprt_code, fs_div)
);

-- 매크로 지표 에이전트 (장단기금리차 + 공포지수). 순수 규칙기반, LLM 미사용.
-- .omc/specs/brainstorming-macro-indicator-agent.md 참고. KR/US 종목 데이터 플레인과
-- 완전 분리이며 자연어 SQL 질의 대상 아님 — QUERYABLE_TABLES 미포함(metrics와 동일 관례).
CREATE TABLE IF NOT EXISTS macro_indicators (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator  TEXT NOT NULL,      -- 지표 키 ('T10Y2Y'=장단기금리차 | 'VIXCLS'=VIX | 'CNN_FNG'=CNN 공포탐욕지수)
    date       TEXT NOT NULL,      -- 데이터 기준일 (YYYY-MM-DD, T+1 — 장중 실시간 아님)
    value      REAL,               -- 지표 값 (금리차=%p, VIX=지수, CNN=0~100 정수)
    source     TEXT,               -- 출처 ('FRED' | 'CNN')
    UNIQUE(indicator, date)
);

-- 신호 판정 이력. 종합신호(overall)는 오직 금리차 레짐에서만 결정되며(정상→GREEN/
-- 평탄화→YELLOW/역전→RED), cnn/vix 밴드는 참고 표시 전용(신호 계산에 관여하지 않음).
-- 날짜별 1행 append(UPDATE 아님) — 이력 추적용. spread_regime='데이터없음'이면 금리차
-- 수집 실패로 직전 신호(overall=prev_overall)를 유지한 상태다.
CREATE TABLE IF NOT EXISTS macro_signal (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of         TEXT NOT NULL,   -- 판정 기준일 (YYYY-MM-DD)
    spread        REAL,            -- 장단기금리차 값(%p), 수집 실패 시 NULL
    spread_regime TEXT,            -- 금리차 레짐 ('정상'|'평탄화'|'역전'|'데이터없음')
    cnn_value     REAL,            -- CNN 공포탐욕지수(0~100), 참고 표시 전용
    cnn_band      TEXT,            -- CNN 밴드('극단공포'|'공포'|'중립'|'탐욕'|'극단탐욕'), 참고용
    vix_value     REAL,            -- VIX 값, 참고 표시 전용
    vix_band      TEXT,            -- VIX 밴드('안정'|'보통'|'경계'|'공포'), 참고용
    overall       TEXT,            -- 종합신호('GREEN'|'YELLOW'|'RED') — 금리차 레짐 단독 결정
    prev_overall  TEXT,            -- 직전 판정의 overall (알림 변경감지·이력용)
    created_at    TEXT             -- 판정 생성 시각 (ISO)
);

-- 올웨더 포트폴리오 모니터링 스냅샷 이력. 매달 1일 배치가 실제 10년 walk-forward 백테스트로
-- 계산한 결과(비중/MDD/CAGR/누적수익률/샤프)를 월별 1행 append(UPDATE 아님)로 쌓는다 —
-- 텔레그램 델타(직전 달 대비 비중 변경분) 계산이 이 이력에 의존한다. 화면은 이 저장값을
-- 읽기만 하고 즉석 재계산하지 않는다. 자연어 SQL 질의 대상 아님(QUERYABLE_TABLES 미포함 —
-- macro/metrics와 동일 관례). .omc/specs/brainstorming-all-weather-portfolio.md 참고.
CREATE TABLE IF NOT EXISTS all_weather_snapshot (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at       TEXT NOT NULL,   -- 배치 실행일 (YYYY-MM-DD)
    weights           TEXT,            -- 종목별 목표비중 (JSON 오브젝트, 티커→비중)
    cagr              REAL,            -- 연평균 복리 수익률 (소수, walk-forward 곡선 기준)
    mdd               REAL,            -- 최대낙폭 (소수, 0 이하)
    sharpe            REAL,            -- 샤프비율 (실현 곡선 기준, 시점별 ^IRX 무위험이자율 반영)
    cumulative_return REAL,            -- 누적수익률 (소수)
    backtest_curve    TEXT,            -- 자산곡선 (JSON 배열, 각 원소 date/nav)
    created_at        TEXT             -- 생성 시각 (ISO)
);
"""

# 읽기 전용 쿼리에 노출되는 테이블 (Text-to-SQL 대상).
# metrics(사전계산 스냅샷)는 제외 — 모든 지표는 financials/prices 원본에서 질의 시점에 계산한다.
# us_company/us_prices/us_financials: .omc/specs/brainstorming-us-nl-sql-integration.md —
# 질문 내용과 무관하게 항상 KR+US 스키마를 동시 노출한다(Round6, 별도 사전판단 단계 없음).
QUERYABLE_TABLES = ["company", "financials", "prices", "us_company", "us_prices", "us_financials"]


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or CONFIG.db_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)  # 락 시 최대 15초 대기
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("PRAGMA journal_mode = WAL")  # 동시 read/write 허용(백필 중 질의)
    except sqlite3.OperationalError:
        pass
    return conn


def connect_readonly(db_path: str | None = None) -> sqlite3.Connection:
    """읽기전용(mode=ro) 연결 — LLM이 생성한 신뢰불가 SQL 실행 전용.

    엔진 레벨에서 모든 쓰기/DDL을 거부하며(attempt to write a readonly database),
    PRAGMA로 되돌릴 수 없다(핸들 레벨). is_safe_select 정규식 필터를 우회당해도
    데이터가 변조되지 않도록 하는 최종 방어층(defense-in-depth)이다.
    앱 내부의 정상 쓰기(wiki 저장 등)는 별도의 쓰기 가능 connect()를 쓴다.
    """
    path = db_path or CONFIG.db_path
    # check_same_thread=False: 파이프라인 실행기(pipeline_exec)가 타임아웃 상한을 강제하려고
    # 워커 스레드에서 프리미티브를 실행하므로 이 읽기전용 연결을 다른 스레드에서 써야 한다.
    # 읽기전용 + 직렬 접근(메인 스레드는 future 완료까지 대기)이라 동시성 문제가 없다.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 15000")  # 락 대기(읽기 전용 설정, DB 미변경)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """기존 DB 무중단 이행: metrics 지표 컬럼 + wiki.model + prices OHLV 컬럼 ALTER ADD."""
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(metrics)")}
    for col, typ in METRIC_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE metrics ADD COLUMN {col} {typ}")
    wcols = {r["name"] for r in conn.execute("PRAGMA table_info(wiki)")}
    if "model" not in wcols:
        conn.execute("ALTER TABLE wiki ADD COLUMN model TEXT")
    pcols = {r["name"] for r in conn.execute("PRAGMA table_info(prices)")}
    for col in ("open", "high", "low", "volume"):
        if col not in pcols:
            conn.execute(f"ALTER TABLE prices ADD COLUMN {col} REAL")
    ucols = {r["name"] for r in conn.execute("PRAGMA table_info(us_financials)")}
    if "disclosed_date" not in ucols:
        conn.execute("ALTER TABLE us_financials ADD COLUMN disclosed_date TEXT")
    ucompany_cols = {r["name"] for r in conn.execute("PRAGMA table_info(us_company)")}
    if "security_type" not in ucompany_cols:
        conn.execute("ALTER TABLE us_company ADD COLUMN security_type TEXT")
    if "financial_currency" not in ucompany_cols:
        conn.execute("ALTER TABLE us_company ADD COLUMN financial_currency TEXT")
    if "cik" not in ucompany_cols:
        # SEC 티커↔CIK 매핑 결과 저장용(us_financials_sec 수집·조회의 조인 키).
        conn.execute("ALTER TABLE us_company ADD COLUMN cik TEXT")
    # us_financials(EAV, 700만+ 행)는 UNIQUE(stock_code, as_of_date, period_type,
    # statement_type, item_key) 자동 인덱스만 있었다 — as_of_date가 period_type/
    # statement_type/item_key보다 앞이라 backtest/data_access_us.py의
    # effective_quarter_at_us/_ttm_us/_yoy_us(모두 stock_code+period_type[+statement_type
    # +item_key]로 필터 후 as_of_date로 정렬/범위조회) 패턴에 안 맞아 종목당 풀스캔이
    # 발생했다(백테스트 US 12개 지표 확장 후 metrics_at_us가 다년치 분기 백테스트에서
    # 수십 초 걸려 브라우저 fetch가 "Load failed"로 끊기는 원인). disclosed_date 컬럼이
    # 이 함수 위쪽에서 이미 보장된 뒤라 여기서 만들어야 구 스키마 DB에서도 안전하다
    # (SCHEMA_DDL 쪽에 두면 disclosed_date ALTER 전에 실행돼 컬럼없음 에러가 남).
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_us_fin_quarter ON us_financials"
        "(stock_code, period_type, disclosed_date, as_of_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_us_fin_ttm ON us_financials"
        "(stock_code, period_type, statement_type, item_key, as_of_date)"
    )
    # us_financials_sec 조회 패턴 인덱스: data_access_us_sec 의 effective_quarter_at_sec/
    # _ttm_sec/_yoy_sec 는 (stock_code, tag[, form]) 로 필터 후 period_end 로 정렬/범위조회하고
    # filed 로 look-ahead 를 거른다. UNIQUE 자동 인덱스는 앞 컬럼이 tag/unit 이라 이 패턴에
    # 안 맞으므로 별도로 만든다(us_financials 인덱스 추가와 동일 취지).
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_us_sec_ttm ON us_financials_sec"
        "(stock_code, tag, period_end)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_us_sec_filed ON us_financials_sec"
        "(stock_code, filed, period_end)"
    )
    conn.commit()


def seed_metric_defs(conn: sqlite3.Connection) -> None:
    """METRIC_DEFS를 metric_def 테이블에 upsert하고, 거기서 빠진(제거된) key는 지운다.

    upsert만 하면 이미 시드된 DB에 init_db를 다시 돌려도 옛 지표(예: 계산 로직이 없어
    UI에서 뺀 dividend_yield)가 그대로 남아 체크박스가 되살아난다 — DELETE로 항상
    METRIC_DEFS와 정확히 일치시킨다.
    """
    keys = [key for key, *_ in METRIC_DEFS]
    for key, label, cat, direction, desc in METRIC_DEFS:
        conn.execute(
            "INSERT OR REPLACE INTO metric_def(key,label,category,direction,description) "
            "VALUES(?,?,?,?,?)",
            (key, label, cat, direction, desc),
        )
    placeholders = ",".join("?" * len(keys))
    conn.execute(f"DELETE FROM metric_def WHERE key NOT IN ({placeholders})", keys)
    conn.commit()


def init_db(db_path: str | None = None) -> None:
    """스키마 생성 + 마이그레이션 + metric_def 시드 (idempotent)."""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_DDL)
        conn.commit()
        _migrate(conn)
        seed_metric_defs(conn)
    finally:
        conn.close()


def schema_catalog(db_path: str | None = None) -> str:
    """LLM 프롬프트에 넣을 스키마 설명 (쿼리 대상 테이블 DDL + 계정키 안내)."""
    lines: list[str] = []
    blocks = SCHEMA_DDL.split("CREATE TABLE IF NOT EXISTS ")
    for block in blocks[1:]:
        name = block.split("(", 1)[0].strip()
        if name in QUERYABLE_TABLES:
            lines.append("CREATE TABLE " + block.split(";", 1)[0].strip() + ";")
    catalog = "\n\n".join(lines)
    account_help = "\n".join(f"  - {k}: {v}" for k, v in ACCOUNT_KEYS.items())
    catalog += "\n\n-- financials.account_key 가능한 값 (정규화된 계정):\n" + account_help
    return catalog


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM ingest_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    from .version import now_iso

    conn.execute(
        "INSERT INTO ingest_meta(key, value, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, now_iso()),
    )
    conn.commit()


if __name__ == "__main__":
    init_db()
    print("DB 초기화 완료:", CONFIG.db_path)
    print("\n=== schema_catalog ===\n")
    print(schema_catalog())
