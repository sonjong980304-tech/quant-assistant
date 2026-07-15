"""FnGuide snapshot.init({...}) 임베디드 JSON 파서.

소스: https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode={code}
페이지 내 <script> 안 snapshot.init({...}) JS 객체 리터럴에 snpFinancial(재무
하이라이트, wide-format: 기간별 VAL1..VAL8 컬럼)과 snpTargetChart(애널리스트
컨센서스 목표주가 시계열) 등이 valid JSON 값으로 박혀 있다. 외곽 JS 객체는
unquoted key라 json.loads가 그대로는 안 통하지만, 각 value 자체는 valid JSON
이므로 정규식으로 키 위치를 찾은 뒤 중괄호 스택 매칭으로 값 경계를 구해
json.loads()로 파싱한다(non-greedy 정규식은 중첩된 '}'에서 잘못 끊기므로
브레이스 카운팅이 필요).
"""
from __future__ import annotations

import calendar
import json
import re
from datetime import datetime, timezone

from ..db import connect, init_db
from .http_fetch import ThrottledFetcher
from .notify import send_slack_alert
from .robust import log_ingest

_SVD_MAIN_URL = "https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode={code}"

_TARGET_CHART_METRIC_KEYS = {
    "DEG": "consensus_opinion_score",
    "TRGT_PRC": "consensus_target_price",
    "CLS_PRC": "adjusted_close",
}


def extract_snapshot_json(html_text: str, key: str) -> dict:
    """snapshot.init({...}) 안의 `key: {...}` 값을 브레이스 매칭으로 추출해 파싱."""
    m = re.search(rf'\b{re.escape(key)}\s*:\s*\{{', html_text)
    if not m:
        raise ValueError(f"key not found: {key}")
    start = m.end() - 1  # '{' 위치
    depth = 0
    for i in range(start, len(html_text)):
        ch = html_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(html_text[start : i + 1])
    raise ValueError(f"unterminated object for key: {key}")


def _yymm_to_period_end(yymm: str) -> str:
    """'YYYY/MM' → 해당 월 마지막 날 'YYYY-MM-DD'."""
    year, month = (int(p) for p in yymm.split("/"))
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-{last_day:02d}"


def parse_financial_highlights(snp_financial: dict) -> list[dict]:
    """snpFinancial(wide-format) → fnguide_metrics EAV 행 리스트.

    header[i]["CD"](VAL1..)가 header[i]["YYMM"] 기간에 대응한다. data의 각 행은
    지표 하나(NAME)이며 VAL1..VALn 컬럼에 기간별 값이 들어있다. null 값은
    제외한다.
    """
    period_by_code = {h["CD"]: h["YYMM"] for h in snp_financial.get("header", [])}
    rows: list[dict] = []
    for item in snp_financial.get("data", []):
        name = (item.get("NAME") or "").strip()
        if not name:
            continue
        for code, yymm in period_by_code.items():
            val = item.get(code)
            if val is None:
                continue
            rows.append(
                {
                    "as_of_date": _yymm_to_period_end(yymm),
                    "metric_key": name,
                    "metric_value": float(val),
                }
            )
    return rows


def parse_target_chart(snp_target_chart: dict) -> list[dict]:
    """snpTargetChart(레코드=하루치, TRD_DT+DEG/TRGT_PRC/CLS_PRC) → EAV 행 리스트.

    snpFinancial과 달리 data의 각 원소가 이미 단일 기간 레코드라 wide→long
    변환이 아니라 필드별 펼치기만 하면 된다. null 값은 제외한다.
    """
    rows: list[dict] = []
    for item in snp_target_chart.get("data", []):
        trd_dt = item.get("TRD_DT")
        if not trd_dt:
            continue
        as_of_date = trd_dt.replace("/", "-")
        for code, metric_key in _TARGET_CHART_METRIC_KEYS.items():
            val = item.get(code)
            if val is None:
                continue
            rows.append({
                "as_of_date": as_of_date,
                "metric_key": metric_key,
                "metric_value": float(val),
            })
    return rows


def fetch_snapshot_html(stock_code: str, fetcher: ThrottledFetcher | None = None) -> str:
    """FnGuide SVD_Main 페이지 HTML을 받아온다(추출은 extract_snapshot_json이 담당)."""
    fetcher = fetcher or ThrottledFetcher()
    url = _SVD_MAIN_URL.format(code=stock_code)
    return fetcher.get(url).text


def ingest_fnguide_metrics(db_path: str | None = None, fetcher: ThrottledFetcher | None = None) -> dict:
    """company 테이블 전종목의 재무 하이라이트+컨센서스 목표주가를 FnGuide에서 받아
    fnguide_metrics에 upsert. 실패 종목은 건너뛰고 Slack 알림만 보낸다(AC11).
    """
    fetcher = fetcher or ThrottledFetcher()
    init_db(db_path)
    conn = connect(db_path)
    collected_at = datetime.now(timezone.utc).isoformat()
    try:
        codes = [r["stock_code"] for r in conn.execute("SELECT stock_code FROM company").fetchall()]
        succeeded = 0
        failed: list[str] = []
        metric_rows = 0
        for code in codes:
            try:
                html = fetch_snapshot_html(code, fetcher=fetcher)
                rows = parse_financial_highlights(
                    extract_snapshot_json(html, "snpFinancial")
                ) + parse_target_chart(extract_snapshot_json(html, "snpTargetChart"))
                if not rows:
                    raise ValueError("빈 응답(필드 누락)")
            except Exception as exc:  # noqa: BLE001 — 종목별 실패를 격리해 다음 종목으로 계속
                failed.append(code)
                log_ingest({"source": "fnguide_metrics", "stock_code": code, "status": "fail", "error": str(exc)})
                send_slack_alert(f"[fnguide_metrics] {code} 수집 실패: {exc}")
                continue
            for row in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO fnguide_metrics"
                    "(stock_code, as_of_date, metric_key, metric_value, source, collected_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (code, row["as_of_date"], row["metric_key"], row["metric_value"], "fnguide", collected_at),
                )
                metric_rows += 1
            succeeded += 1
        conn.commit()
        return {"tickers": len(codes), "succeeded": succeeded, "failed": failed, "metric_rows": metric_rows}
    finally:
        conn.close()
