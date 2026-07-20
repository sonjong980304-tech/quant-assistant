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

-- KR 관리종목/매매거래정지 상태 이력 (구간 기반, 스냅샷 누적). KRX/KIND 는 과거 조회일을
-- 무시하고 '오늘 현재' 스냅샷만 반환해 과거 이력을 무료로 구할 수 없다 → 매 실행(매일 1회)
-- 마다 '오늘 현재' 관리종목/거래정지 목록을 받아 직전 실행 스냅샷과 diff 해서 앞으로의
-- 지정~해제 구간을 우리 쪽에서 누적으로 쌓는다. '직전 스냅샷'은 별도 저장 없이 이 테이블의
-- 열린 구간(end_date IS NULL)에서 유도한다: 현재 목록에 새로 나타난 종목=지정 개시(start_date=
-- 관측일, end_date=NULL), 사라진 종목=해제(end_date=관측일), 계속 있는 종목=구간 유지.
-- 구간 기반 멱등 upsert 사상 — 한 종목이 지정→해제→재지정을 반복하면
-- 여러 행(각 구간 1행)을 가진다. status_type 판별자로 관리종목/거래정지를 한 테이블에 담는다
-- (macro_indicators.indicator 판별자 관례와 동일 — diff 알고리즘이
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

-- KR 기업 주요 변동이력 (상호/업종/액면 변경). pykrx get_stock_major_changes(ticker) 가 종목별로
-- 날짜별 상호변경전/후·업종변경전/후·액면변경전/후·대표이사변경전/후를 돌려준다(1975년부터의
-- 전체 이력을 시점조회 없이 한 번에 반환 — kr_trading_status 의 '오늘 스냅샷만' 제약과 달리 과거
-- 이력이 통째로 온다). 이 중 스크리닝/백테스트/회사명 매칭에 쓰는 상호·업종·액면만 저장하고
-- 대표이사변경은 저장하지 않는다(YAGNI). 핵심 활용: 자연어→SQL 질의에서 회사명 매칭이 현재
-- 사명(company.name)으로 실패할 때 예전 사명(name_before/name_after)으로도 종목코드를 찾는다
-- (domain_kr.find_stock_code 폴백). 액면변경(분할/병합)은 향후 수동 분할보정 도구가 "언제 분할이
-- 있었나" 참고자료로 쓸 수 있어 저장만 해둔다(이번 스코프에서 그 도구엔 연결하지 않음).
-- '없음' 표기는 pykrx 원본이 텍스트 "-"/액면 0 이며, 적재 시 모두 NULL 로 정규화한다. 한 종목이
-- 여러 번 변경하면 (종목,날짜)별로 여러 행을 가진다 → UNIQUE(stock_code, changed_at) 로 재수집
-- 멱등(INSERT OR REPLACE 로 값 정정도 반영). 자연어 SQL 질의 대상 아님(QUERYABLE_TABLES 미포함 —
-- 회사명 매칭 폴백 전용으로 domain_kr 이 execute_sql 경유로만 읽는다).
CREATE TABLE IF NOT EXISTS kr_stock_changes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code    TEXT NOT NULL,   -- 종목코드 (6자리 단축코드)
    changed_at    TEXT NOT NULL,   -- 변경일 (YYYY-MM-DD, pykrx 날짜 인덱스)
    name_before   TEXT,            -- 상호변경전 (없으면 NULL)
    name_after    TEXT,            -- 상호변경후 (없으면 NULL)
    sector_before TEXT,            -- 업종변경전 (없으면 NULL)
    sector_after  TEXT,            -- 업종변경후 (없으면 NULL)
    par_before    INTEGER,         -- 액면변경전 (원, 없으면 NULL — pykrx 0 을 NULL 로)
    par_after     INTEGER,         -- 액면변경후 (원, 없으면 NULL)
    updated_at    TEXT,            -- 마지막 수집 시각 (ISO)
    UNIQUE(stock_code, changed_at)
);

-- KR 관리종목/매매거래정지 '진짜 과거 이력' (구간 기반). kr_trading_status 는 KRX 가 '오늘 스냅샷'
-- 만 줘서 미래 구간만 쌓이지만, 이 테이블은 DART OpenDART list.json(회사별 공시목록)이 실제로
-- 과거 조회(bgn_de=20150101)를 지원하는 점을 이용해 과거까지 소급 복원한 이력이다. 소스가 완전히
-- 다르므로(KRX 스냅샷 vs DART 공시) kr_trading_status 와 별개 테이블로 둔다 — 그 테이블엔 '미래
-- 전용, 백테스트 연결 금지' 주석이 박혀 있어 혼동을 피하려 분리했다.
-- 복원 방식: DART 공시 제목(report_nm)을 순수 분류(kr_admin_status_history.classify_disclosure)해
-- 관리종목 지정/해제·매매거래정지 시작/해제 이벤트만 확정하고, 이를 시간순으로 짝지어 구간을
-- 만든다(build_status_intervals). report_nm 이 방향을 문자로 담지 않는 애매한 공시(KOSPI
-- '매매거래정지및정지해제' 결합형, '주권매매거래정지기간변경')는 구간으로 만들지 않고 review 로
-- 보류·보고한다(추측성 오분류 방지). start_date/end_date 는 DART 접수일(rcept_dt)이라 kr_trading_
-- status 의 '관측일'과 달리 실제 사건일에 가깝다. start_report_nm/end_report_nm 에 트리거 공시
-- 제목을 남겨 사람이 각 경계를 감사·검증할 수 있게 한다.
-- kr_trading_status 와 동일한 구간+멱등 upsert 사상 — 한 종목이 지정→해제→재지정을
-- 반복하면 여러 행(각 구간 1행). UNIQUE(stock_code, status_type, start_date) 로 재수집 멱등이고,
-- 열린 구간(end_date NULL)이 나중에 해제 공시로 닫히면 그 행의 end_date 만 갱신한다.
-- ⚠️ 이번 스코프에선 backtest(data_access asof)에 연결하지 않는다 — 데이터 품질을 사용자가 검증한
-- 뒤 별도로 연결 여부를 결정한다. 자연어 SQL 질의 대상 아님(QUERYABLE_TABLES 미포함).
CREATE TABLE IF NOT EXISTS kr_admin_status_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code      TEXT NOT NULL,   -- 종목코드 (6자리 단축코드)
    status_type     TEXT NOT NULL,   -- 'admin'(관리종목) | 'halt'(매매거래정지)
    start_date      TEXT NOT NULL,   -- 지정/정지 시작일 (DART rcept_dt, YYYY-MM-DD)
    end_date        TEXT,            -- 해제일 (YYYY-MM-DD; NULL=미해제/진행 중)
    start_report_nm TEXT,            -- 시작 트리거 공시 제목 (감사·검증용)
    end_report_nm   TEXT,            -- 해제 트리거 공시 제목 (감사·검증용; 미해제면 NULL)
    source          TEXT,            -- 출처 태그 ('dart_list')
    updated_at      TEXT,            -- 마지막 수집 시각 (ISO)
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
-- 중 가장 최근 disclosed_date" 버전을 고른다(공시일 상한 기반 look-ahead 방지).
-- 기존 financials 테이블과 그 소비자(goldset/legacy/data_financial 등)는 전혀 건드리지 않는다.
-- rcept_no: DART 접수번호(공시 고유 id). 같은 공시 재수집=같은 rcept_no → UNIQUE 로 멱등,
-- 진짜 새 rcept_no(정정공시)만 새 행. 접수번호 미상(추정 공시일 폴백)이면 disclosed_date 를
-- 센티널로 넣어(빈문자열 센티널 관례) NULL 로 인한 멱등성 깨짐을 막는다.
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
CREATE INDEX IF NOT EXISTS idx_kr_stock_changes_code ON kr_stock_changes(stock_code);
CREATE INDEX IF NOT EXISTS idx_kr_admin_status_hist_code ON kr_admin_status_history(stock_code, status_type);

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
QUERYABLE_TABLES = ["company", "financials", "prices"]


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


def schema_catalog(db_path: str | None = None, include_metrics: bool = False) -> str:
    """LLM 프롬프트에 넣을 스키마 설명 (쿼리 대상 테이블 DDL + 계정키 안내).

    include_metrics=False(기본): 기존 동작 그대로 QUERYABLE_TABLES만 노출한다.
    include_metrics=True: 사전계산 스냅샷 테이블 metrics DDL을 추가로 포함한다 —
        자유코드 폴백(exec_fallback)처럼 "지금 기준" 파생지표를 재계산하지 않고
        이미 계산된 값을 바로 쓰는 게 더 정확·안전한 opt-in 경로에서만 켠다.
    """
    tables = QUERYABLE_TABLES + ["metrics"] if include_metrics else QUERYABLE_TABLES
    lines: list[str] = []
    blocks = SCHEMA_DDL.split("CREATE TABLE IF NOT EXISTS ")
    for block in blocks[1:]:
        name = block.split("(", 1)[0].strip()
        if name in tables:
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
