"""파생 지표 계산 (financials + prices → metrics).

밸류
- PER = 시총 / 당기순이익(TTM) · PBR = 시총 / 지배주주지분
- PSR = 시총 / 매출(TTM) · PCR = 시총 / 영업현금흐름(TTM)
- EV/EBITDA = (시총+부채-현금) / (영업이익+감가상각, TTM) · PEG = PER / 순이익성장률
수익성
- ROE = 순이익(TTM)/자본 · ROA = 순이익(TTM)/총자산 · 영업이익률 · 순이익률 · GP/A = 매출총이익(TTM)/총자산
안정성
- 부채비율 = 부채/자본 · 유동비율 = 유동자산/유동부채 · 이자보상배율 = 영업이익(TTM)/이자비용(TTM)
성장(YoY 분기)
- 매출/영업이익/순이익 성장률 = (당분기 - 전년동기)/|전년동기|
기타
- 배당수익률 = 배당금/시총 · 모멘텀 = 최근 6개월 주가 상승률

광범위 계정(매출총이익/현금흐름/유동자산·부채/이자비용/배당)이 아직 없으면 해당
지표는 NULL로 두고, 광범위 수집 후 재계산 시 자동으로 채워진다.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

from ..version import effective_price_date, effective_quarter, shift_quarter

_PCT = 100.0

# 이상치/오류 가드 임계값(단일 정의처 — src/backtest/data_access.py의 metrics_at()이
# 질의 경로와 동일 기준을 쓰도록 여기서 import해 공유한다. son-checker 이슈 #23 IMP-C:
# 예전엔 두 파일에 거의 동일하게 매직넘버가 복붙돼 있어 나중에 따로 놀 위험이 있었다).
CONTROLLING_EQUITY_MAX_RATIO = 1.05  # 지배주주지분이 자본총계의 이 배수를 넘으면 수집 오류로 간주
CONTROLLING_EQUITY_MIN_RATIO = 0.20  # 지배주주지분이 자본총계의 이 비율 미만이면 수집 오류로 간주
NI_TO_EQUITY_MAX_RATIO = 20          # 순이익이 자기자본의 이 배수를 넘으면(Q4 차분 폭발 등) 무효화
CAP_TO_EQUITY_MAX_RATIO = 100        # 시총이 자기자본의 이 배수를 넘으면(PBR 극단) 시총 기반 지표 무효화
PER_MAX_REASONABLE = 10000  # PER이 이 배수를 넘으면 분모(지배주주순이익)가 사실상 0에 가까운
                             # 이상치로 간주해 무효화(실측: 유수홀딩스가 DART 비표준계정 오류로
                             # 지배주주순이익이 100~600원대로 잘못 잡혀 PER이 1억9600만배까지
                             # 치솟아 코스피 전체 PER 히스토그램을 오염시킨 사례)


def _fin(conn, code, quarter, key, asof=None):
    """(종목,분기,계정)의 금액.

    asof=None(기본, compute_metrics 등 "현재 최신값" 경로): 기존대로 financials 에서 읽는다.
    asof 지정(백테스트 look-ahead 크리티컬 경로): financials_revision 에서 disclosed_date<=asof
    인 버전 중 가장 최근 disclosed_date 값을 고른다(SEC filed_max 와 동일 사상 — 정정공시로
    나중에야 알게 된 값을 과거 시점에 앞당겨 쓰지 않는다). 그 (종목,분기,계정)에 정정이력이
    하나라도 있으면 asof 이전 버전이 없을 때 None(look-ahead 방지). 이력이 아예 없으면
    (정정테이블 도입 전 적재분) 기존 financials 로 폴백해 과거 데이터 백테스트가 깨지지 않게 한다.
    """
    if asof is not None:
        revs = conn.execute(
            "SELECT amount, disclosed_date FROM financials_revision "
            "WHERE stock_code=? AND quarter=? AND account_key=? "
            "ORDER BY disclosed_date DESC",
            (code, quarter, key),
        ).fetchall()
        if revs:  # 정정이력 존재 → asof 이하 최신 버전만 사용(없으면 None, 폴백 안 함)
            for r in revs:
                if r["disclosed_date"] is not None and r["disclosed_date"] <= asof:
                    return r["amount"] if r["amount"] is not None else None
            return None
        # 정정이력 없음(도입 전 적재분) → 기존 financials 폴백
    row = conn.execute(
        "SELECT amount FROM financials WHERE stock_code=? AND quarter=? AND account_key=?",
        (code, quarter, key),
    ).fetchone()
    return row["amount"] if row and row["amount"] is not None else None


def _sum_ttm(conn, code, quarter, key, asof=None):
    """최근 4분기 합(TTM).

    SoT [METRIC]: TTM은 '실제 최근 4개 분기 합'만 사용한다.
    SoT [DATA]: 날짜/일부값으로 분기를 추정하지 않는다.
    → 4분기 중 하나라도 누락되면 추정(평균×4)하지 않고 None을 반환한다.
    asof 는 _fin 으로 그대로 전달(look-ahead: 각 분기값도 그 시점 유효 버전으로 조회).
    """
    vals = [_fin(conn, code, shift_quarter(quarter, -i), key, asof=asof) for i in range(4)]
    if any(v is None for v in vals):
        return None
    return sum(vals)


# ── 시점별 상장주식수 조회 (시가총액 = 종가 × 그 시점 상장주식수) ──────────────────
# 시점별 상장주식수를 disclosed_date<=asof 기준으로 고르는 KR 버전.
# 기존 ingest 는 _collect_shares 로 "오늘 기준 최신 주식수" 상수 하나를 과거 전 기간에 곱해,
# 유상증자·자사주소각·액면분할 등으로 주식수가 바뀐 종목의 과거 시총이 부정확했다.
# financials.shares_outstanding(분기별, disclosed_date)에서 asof 시점까지 공시된 최신값을 골라
# look-ahead 를 막는다(_fin/effective_quarter_at 과 동일 규약). shares_outstanding 은
# _ingest_shares 가 financials 에만 UPSERT 하고 financials_revision 을 거치지 않으므로
# (라이브 DB 실측 0행) financials 만 조회한다.
_SHARES_STALE_DAYS = 400     # 마지막 공시가 asof 대비 이보다 오래되면 stale(US _SHARES_STALE_DAYS 와 동일)
_SHARES_ANOMALY_RATIO = 100  # 종목 median 대비 이 배수 이상 이탈하면 단위오류로 제외(dart._SHARES_JUMP_RATIO 와 동일)


def _effective_shares_outstanding(series, asof):
    """series=[(disclosed_date'YYYY-MM-DD', amount), ...] 공시일 오름차순 → asof 유효 상장주식수.

    · look-ahead 방지: disclosed_date<=asof 중 가장 최근 공시값(없으면 None).
    · staleness: 그 공시가 asof 대비 _SHARES_STALE_DAYS 초과로 오래됐으면 None(상수 폴백 유도).
    · 이상치 가드: 종목 전체 shares median 대비 _SHARES_ANOMALY_RATIO 배 이상 이탈한 값은
      단위오류(×1000 원복형 스파이크 — 003240·226400 등 실측 54종목)로 보고 None 반환.
      액면분할(삼성 50:1 등, 지속적 계단변화)은 median 이 분할 후 값 쪽으로 수렴해 배수가 100
      미만이라 죽지 않는다(실측 최대 분할이 50:1). insert 시점 _shares_jump_anomalous 가드는
      백필 중 분기 역순 삽입으로 원복형 스파이크를 놓쳤어서, read 시점에 한 번 더 방어한다
      (prices.market_cap·text-to-SQL 랭킹처럼 자기자본 가드가 없는 소비자를 오염에서 보호).
    """
    if not series:
        return None
    chosen = chosen_disc = None
    for disc, amt in series:
        if disc <= asof:
            chosen, chosen_disc = amt, disc
        else:
            break  # 공시일 오름차순이라 이후는 전부 미래 공시(look-ahead)
    if chosen is None:
        return None
    if (date.fromisoformat(asof) - date.fromisoformat(chosen_disc)).days > _SHARES_STALE_DAYS:
        return None
    amounts = sorted(a for _, a in series)
    median = amounts[len(amounts) // 2]
    if median > 0 and (chosen > median * _SHARES_ANOMALY_RATIO or chosen < median / _SHARES_ANOMALY_RATIO):
        return None
    return chosen


def _shares_outstanding_at(conn, code, asof):
    """asof 시점(disclosed_date<=asof) 유효 상장주식수. 없거나 stale/이상치면 None.

    _effective_shares_outstanding 의 DB 조회 래퍼 — financials 에서 종목의 shares_outstanding
    시계열을 공시일 오름차순으로 읽어 넘긴다. disclosed_date 결측/빈문자 행은 시점을 알 수 없어
    제외한다(look-ahead 방지). ingest(krx)·마이그레이션(backfill_marketcap)이 이 함수를 공유해
    동일 규약을 쓴다.
    """
    rows = conn.execute(
        "SELECT disclosed_date, amount FROM financials "
        "WHERE stock_code=? AND account_key='shares_outstanding' "
        "AND amount IS NOT NULL AND amount>0 AND disclosed_date IS NOT NULL AND disclosed_date!='' "
        "ORDER BY disclosed_date",
        (code,),
    ).fetchall()
    return _effective_shares_outstanding([(r["disclosed_date"], r["amount"]) for r in rows], asof)


def controlling_equity(conn, code, quarter, asof=None):
    """지배주주지분(지배기업소유주지분). 미수집/이상값이면 자본총계로 폴백.

    SoT [METRIC]: ROE/PBR 분모는 '지배주주지분'을 쓴다(자본총계 아님).
    단독재무 등 비지배지분이 없는 기업은 지배주주지분=자본총계이므로 폴백이 안전하다.

    sanity 폴백(임시 방어 — 근본 해결은 normalize.py 수정 후 재수집):
      · 결측
      · 자본총계 초과(논리상 불가 → 수집 오류, 예: 이큐셀 85경)
      · 자본총계의 20% 미만(보고서별 표기 흔들림으로 비정상적으로 작게 수집, 예: 삼성 47조)
    → 위 경우 자본총계로 폴백한다.
    asof 는 _fin 으로 그대로 전달(백테스트 look-ahead 경로에서 그 시점 유효 버전으로 조회).
    """
    v = _fin(conn, code, quarter, "controlling_equity", asof=asof)
    total = _fin(conn, code, quarter, "total_equity", asof=asof)
    if total is not None and total > 0:
        if (
            v is None
            or v > total * CONTROLLING_EQUITY_MAX_RATIO
            or v < total * CONTROLLING_EQUITY_MIN_RATIO
        ):
            return total
    elif v is None:
        return total
    return v


def avg_controlling_equity(conn, code, quarter, asof=None):
    """ROE 분모용 자기자본 평균 = (4분기 전 기말 + 최근 기말 지배주주지분)/2.

    SoT [METRIC]: ROE 분자(TTM, 4분기)와 분모(자본 평균)의 기간을 일치시킨다.
    4분기 전 값이 없으면(신규상장 등) 최근 기말 단독값으로 폴백한다.
    asof 는 controlling_equity 로 그대로 전달(look-ahead 경로에서 그 시점 유효 버전으로 조회).
    """
    cur = controlling_equity(conn, code, quarter, asof=asof)
    prev = controlling_equity(conn, code, shift_quarter(quarter, -4), asof=asof)
    if cur is None:
        return None
    if prev is None:
        return cur
    return (cur + prev) / 2.0


def _yoy(conn, code, quarter, key, asof=None):
    """전년동기 대비 성장률(%). 분기 단독값 기준.

    asof 는 _fin 으로 그대로 전달(look-ahead 경로에서 당분기·전년동기 모두 그 시점 유효 버전).
    """
    cur = _fin(conn, code, quarter, key, asof=asof)
    prev = _fin(conn, code, shift_quarter(quarter, -4), key, asof=asof)
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev) * _PCT


def _momentum(conn, code, price_date, months=6):
    """최근 months개월 주가 상승률(%). 과거 종가가 없으면 None."""
    cur = conn.execute(
        "SELECT close FROM prices WHERE stock_code=? AND date=?", (code, price_date)
    ).fetchone()
    if not cur or not cur["close"]:
        return None
    target = (date.fromisoformat(price_date) - timedelta(days=months * 30)).isoformat()
    prev = conn.execute(
        "SELECT close FROM prices WHERE stock_code=? AND date<=? ORDER BY date DESC LIMIT 1",
        (code, target),
    ).fetchone()
    if not prev or not prev["close"]:
        return None
    return (cur["close"] / prev["close"] - 1) * _PCT


def _div(a, b, pct=False):
    if a is None or b is None or b == 0:
        return None
    v = a / b
    return v * _PCT if pct else v


def compute_metrics(conn: sqlite3.Connection, d=None) -> int:
    """metrics 테이블 전체 지표 재계산. 반환: 갱신 종목 수."""
    quarter = effective_quarter(conn, d)
    pdc = effective_price_date(conn, d)
    price_date = f"{pdc[:4]}-{pdc[4:6]}-{pdc[6:]}"

    companies = conn.execute("SELECT stock_code, name FROM company").fetchall()
    n = 0
    for crow in companies:
        code = crow["stock_code"]
        name = crow["name"] or ""
        cap_row = conn.execute(
            "SELECT market_cap, close FROM prices WHERE stock_code=? AND date=?",
            (code, price_date),
        ).fetchone()
        cap = cap_row["market_cap"] if cap_row else None

        # 시점값(BS)
        equity = _fin(conn, code, quarter, "total_equity")        # 부채비율 분모(자본총계)
        book = controlling_equity(conn, code, quarter)            # PBR 분모(지배주주지분)
        avg_eq = avg_controlling_equity(conn, code, quarter)      # ROE 분모(4분기 평균 지배주주지분)
        liab = _fin(conn, code, quarter, "total_liabilities")
        assets = _fin(conn, code, quarter, "total_assets")
        cur_assets = _fin(conn, code, quarter, "current_assets")
        cur_liab = _fin(conn, code, quarter, "current_liabilities")
        cash = _fin(conn, code, quarter, "cash")  # (광범위에서 추가 시)

        # TTM(손익/현금흐름)
        rev_ttm = _sum_ttm(conn, code, quarter, "revenue")
        op_ttm = _sum_ttm(conn, code, quarter, "operating_profit")
        ni_ttm = _sum_ttm(conn, code, quarter, "net_income")               # 연결 전체 (ROA용)
        # PER·ROE 분자는 '지배주주 귀속 순이익'. 미수집 시 연결 순이익으로 폴백.
        ctrl_ni_ttm = _sum_ttm(conn, code, quarter, "controlling_net_income")
        if ctrl_ni_ttm is None:
            ctrl_ni_ttm = ni_ttm
        # 마진(영업이익률/순이익률)은 해당 분기 단독 기준 (TTM 아님) — 특정 분기 질의에 정확히 답하기 위함
        rev_q = _fin(conn, code, quarter, "revenue")
        op_q = _fin(conn, code, quarter, "operating_profit")
        ni_q = _fin(conn, code, quarter, "net_income")
        gp_ttm = _sum_ttm(conn, code, quarter, "gross_profit")
        ocf_ttm = _sum_ttm(conn, code, quarter, "operating_cashflow")
        dep_ttm = _sum_ttm(conn, code, quarter, "depreciation")
        int_ttm = _sum_ttm(conn, code, quarter, "interest_expense")
        div_ttm = _sum_ttm(conn, code, quarter, "dividend")

        # 성장률(YoY 분기)
        rev_g = _yoy(conn, code, quarter, "revenue")
        op_g = _yoy(conn, code, quarter, "operating_profit")
        ni_g = _yoy(conn, code, quarter, "net_income")

        # ⑤ 이상치/오류 가드 (질의 경로를 백테스트 경로와 동일 기준으로 정렬)
        # - 스팩(기업인수목적회사): 현금성 자산 덩어리라 PER/ROE/PBR 무의미 → 가치·수익성 지표 제외
        is_spac = ("스팩" in name) or ("기업인수목적" in name)
        # - 순이익이 자기자본 대비 비현실적으로 거대(Q4 차분 폭발/원본 오류)면 순이익 기반 지표 무효화
        if ni_ttm is not None and equity and equity > 0 and abs(ni_ttm) > equity * NI_TO_EQUITY_MAX_RATIO:
            ni_ttm = None
        if (
            ctrl_ni_ttm is not None
            and equity
            and equity > 0
            and abs(ctrl_ni_ttm) > equity * NI_TO_EQUITY_MAX_RATIO
        ):
            ctrl_ni_ttm = None
        # - 시총이 자기자본의 100배 초과(PBR>100) → 상장주식수/주가 데이터 오류 의심
        #   (예: 서린바이오 시총 40조 = 종가×주식수 91억주, 자본 853억 → PBR 469배).
        #   시총 기반 지표(PER·PBR·PSR·시총)를 무효화한다.
        if cap is not None and equity and equity > 0 and cap > equity * CAP_TO_EQUITY_MAX_RATIO:
            cap = None

        # 밸류
        per = _div(cap, ctrl_ni_ttm) if (not is_spac and ctrl_ni_ttm and ctrl_ni_ttm > 0) else None  # 분자=지배주주순이익
        pbr = _div(cap, book) if (not is_spac and book and book > 0) else None    # ① 지배주주지분
        psr = _div(cap, rev_ttm) if (not is_spac and rev_ttm and rev_ttm > 0) else None
        pcr = _div(cap, ocf_ttm) if (not is_spac and ocf_ttm and ocf_ttm > 0) else None
        peg = (per / ni_g) if (per is not None and ni_g and ni_g > 0) else None
        ebitda = (op_ttm + dep_ttm) if (op_ttm is not None and dep_ttm is not None) else None
        ev = (cap + liab - (cash or 0)) if (cap is not None and liab is not None) else None
        ev_ebitda = _div(ev, ebitda) if (ebitda and ebitda > 0) else None

        # 수익성
        # ② ROE = 지배주주순이익(TTM) ÷ 4분기 평균 지배주주지분. 분자·분모 모두 지배주주 귀속으로 대응.
        #    부호는 그대로 인정(적자면 음수=유효, net_margin과 동일 관례). 다만 자본 평균 초과(≥100%,
        #    일회성 폭발)는 이상치로 보고 제외한다.
        roe = _div(ctrl_ni_ttm, avg_eq, pct=True) if (
            not is_spac and ctrl_ni_ttm and avg_eq and avg_eq > 0 and ctrl_ni_ttm < avg_eq
        ) else None
        # ROA = 연결 전체 순이익 ÷ 총자산(전체) — 분자·분모 모두 연결 전체로 대응
        roa = _div(ni_ttm, assets, pct=True) if (not is_spac and ni_ttm and assets and assets > 0) else None
        # ③ 영업이익률=단일분기. op<rev(마진<100%)일 때만 — 지주사·지분법이익 폭발 제외
        opm = _div(op_q, rev_q, pct=True) if (rev_q and rev_q > 0 and op_q is not None and op_q < rev_q) else None
        npm = _div(ni_q, rev_q, pct=True) if (rev_q and rev_q > 0 and ni_q is not None) else None
        gp_a = _div(gp_ttm, assets, pct=True) if (assets and assets > 0) else None

        # 안정성
        debt_ratio = _div(liab, equity, pct=True) if (equity and equity > 0) else None
        current_ratio = _div(cur_assets, cur_liab, pct=True) if (cur_liab and cur_liab > 0) else None
        interest_cov = _div(op_ttm, int_ttm) if (int_ttm and int_ttm > 0) else None

        # 기타
        div_yield = _div(div_ttm, cap, pct=True) if (cap and cap > 0 and div_ttm) else None
        momentum = _momentum(conn, code, price_date)

        conn.execute(
            """INSERT INTO metrics
                 (stock_code, quarter, price_date, market_cap, per, pbr, psr, pcr, ev_ebitda, peg,
                  roe, roa, operating_margin, net_margin, gp_a,
                  debt_ratio, current_ratio, interest_coverage,
                  revenue_growth, op_growth, ni_growth, dividend_yield, momentum)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(stock_code, quarter, price_date) DO UPDATE SET
                 market_cap=excluded.market_cap, per=excluded.per, pbr=excluded.pbr,
                 psr=excluded.psr, pcr=excluded.pcr, ev_ebitda=excluded.ev_ebitda, peg=excluded.peg,
                 roe=excluded.roe, roa=excluded.roa, operating_margin=excluded.operating_margin,
                 net_margin=excluded.net_margin, gp_a=excluded.gp_a,
                 debt_ratio=excluded.debt_ratio, current_ratio=excluded.current_ratio,
                 interest_coverage=excluded.interest_coverage,
                 revenue_growth=excluded.revenue_growth, op_growth=excluded.op_growth,
                 ni_growth=excluded.ni_growth, dividend_yield=excluded.dividend_yield,
                 momentum=excluded.momentum""",
            (code, quarter, price_date, _r(cap), _r(per), _r(pbr), _r(psr), _r(pcr),
             _r(ev_ebitda), _r(peg), _r(roe), _r(roa), _r(opm), _r(npm), _r(gp_a),
             _r(debt_ratio), _r(current_ratio), _r(interest_cov),
             _r(rev_g), _r(op_g), _r(ni_g), _r(div_yield), _r(momentum)),
        )
        n += 1
    conn.commit()
    return n


def _r(v, nd=2):
    return round(v, nd) if v is not None else None
