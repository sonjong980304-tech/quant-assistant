"""FnGuide snapshot.init({...}) 임베디드 JSON 파서 회귀 테스트.

.omc/specs/brainstorming-naver-fnguide-crawlers.md FnGuide 크롤러 소스 섹션 참고.
소스: https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode={code}
실제 페이지(2026-07-11 리서치 에이전트가 확보한 fixture)의 <script> 안에
snapshot.init({ ..., snpFinancial: {...}, snpTargetChart: {...}, ... }) 형태로
여러 개의 valid-JSON 값이 unquoted-key JS 객체 리터럴 안에 박혀 있다.
"""
from __future__ import annotations

import src.ingest.fnguide_metrics as fnguide_metrics
from src.db import connect, init_db
from src.ingest.fnguide_metrics import (
    extract_snapshot_json,
    fetch_snapshot_html,
    ingest_fnguide_metrics,
    parse_financial_highlights,
    parse_target_chart,
)
from tests.conftest import FailingFetcher, FakeFetcher, seed_kr_companies


def test_extract_snapshot_json_finds_nested_braces_correctly():
    # snpDebt(단순 flat 객체)가 먼저 나오고, 그 뒤 중첩 배열/객체를 포함한
    # snpTargetChart가 나오는 실제 구조를 흉내낸다 — naive non-greedy 정규식은
    # 중첩된 '}' 에서 잘못 끊기므로 브레이스 스택 매칭이 필요함을 검증.
    html = """
    <script type="text/javascript">
        snapshot.init({
            cmp_cd: '005930',
            snpDebt: {"KIS_DT":"2004/05/25","KIS_GRD":"AAA"},
            snpTargetChart : {"header":[{"ID":"TRD_DT","NM":"일자"}],"data":[{"TRD_DT":"2025/07/11","DEG":4.00}]},
            snpSector: {"CD":"KSE"}
        });
    </script>
    """
    result = extract_snapshot_json(html, "snpTargetChart")
    assert result == {
        "header": [{"ID": "TRD_DT", "NM": "일자"}],
        "data": [{"TRD_DT": "2025/07/11", "DEG": 4.00}],
    }


def test_extract_snapshot_json_missing_key_raises():
    html = "snapshot.init({ cmp_cd: '005930' });"
    try:
        extract_snapshot_json(html, "snpFinancial")
        assert False, "ValueError가 발생해야 함"
    except ValueError:
        pass


def test_parse_financial_highlights_converts_wide_periods_to_eav_rows():
    # 실제 snpFinancial 구조 축약본: 2개 기간(VAL1=2023/12, VAL2=2024/02 윤년)
    # x 2개 지표(매출액, 영업이익). VAL2의 영업이익은 null(추정치 미확정) 케이스 포함.
    snp_financial = {
        "header": [
            {"YYMM": "2023/12", "CD": "VAL1", "NO_TYP": "IFRS"},
            {"YYMM": "2024/02", "CD": "VAL2", "NO_TYP": "IFRS"},
        ],
        "data": [
            {"SEQ": 1, "NAME": "매출액", "VAL1": "2589354.94", "VAL2": "3008709.03"},
            {"SEQ": 2, "NAME": "영업이익", "VAL1": "65669.76", "VAL2": None},
        ],
    }
    rows = parse_financial_highlights(snp_financial)
    assert {
        "as_of_date": "2023-12-31",
        "metric_key": "매출액",
        "metric_value": 2589354.94,
    } in rows
    # 2024-02는 윤년이라 마지막 날이 29일이어야 한다
    assert {
        "as_of_date": "2024-02-29",
        "metric_key": "매출액",
        "metric_value": 3008709.03,
    } in rows
    # null 값(영업이익 VAL2)은 제외돼야 한다
    assert not any(r["metric_key"] == "영업이익" and r["as_of_date"] == "2024-02-29" for r in rows)
    assert len(rows) == 3


def test_parse_target_chart_converts_records_to_eav_rows():
    # 실제 fixture 구조(scratchpad/fnguide_svd_main.html)로 확인: data의 각 원소가
    # 이미 하루치 레코드(TRD_DT + DEG/TRGT_PRC/CLS_PRC)라 snpFinancial과 달리
    # wide-period 변환이 아니라 필드별로 펼치기만 하면 된다.
    snp_target_chart = {
        "header": [
            {"ID": "TRD_DT", "NM": "일자", "DIGIT": -1},
            {"ID": "DEG", "NM": "투자의견(좌)", "DIGIT": 2},
            {"ID": "TRGT_PRC", "NM": "목표주가", "DIGIT": 0},
            {"ID": "CLS_PRC", "NM": "수정주가", "DIGIT": 0},
        ],
        "data": [
            {"TRD_DT": "2025/07/11", "DEG": 4.0, "TRGT_PRC": 75792.0, "CLS_PRC": 62600.0},
        ],
    }
    rows = parse_target_chart(snp_target_chart)
    assert {"as_of_date": "2025-07-11", "metric_key": "consensus_target_price", "metric_value": 75792.0} in rows
    assert {"as_of_date": "2025-07-11", "metric_key": "consensus_opinion_score", "metric_value": 4.0} in rows
    assert {"as_of_date": "2025-07-11", "metric_key": "adjusted_close", "metric_value": 62600.0} in rows
    assert len(rows) == 3


def test_parse_target_chart_skips_records_with_null_field_value():
    snp_target_chart = {
        "header": [{"ID": "TRD_DT", "NM": "일자"}, {"ID": "TRGT_PRC", "NM": "목표주가"}],
        "data": [{"TRD_DT": "2025/07/11", "DEG": None, "TRGT_PRC": 75792.0, "CLS_PRC": None}],
    }
    rows = parse_target_chart(snp_target_chart)
    metric_keys = {r["metric_key"] for r in rows}
    assert metric_keys == {"consensus_target_price"}


def test_fetch_snapshot_html_builds_svd_main_url_with_gicode():
    fetcher = FakeFetcher("<html>snapshot.init({});</html>")
    fetch_snapshot_html("005930", fetcher=fetcher)
    assert fetcher.calls == [
        "https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=005930"
    ]


def test_fetch_snapshot_html_returns_response_text():
    fetcher = FakeFetcher("<html>snapshot.init({});</html>")
    html = fetch_snapshot_html("005930", fetcher=fetcher)
    assert html == "<html>snapshot.init({});</html>"


_SAMPLE_SNAPSHOT_HTML = """
<script type="text/javascript">
snapshot.init({
    cmp_cd: '005930',
    snpFinancial: {"header":[{"YYMM":"2023/12","CD":"VAL1","NO_TYP":"IFRS"}],"data":[{"SEQ":1,"NAME":"매출액","VAL1":"100.0"}]},
    snpTargetChart: {"header":[{"ID":"TRD_DT","NM":"일자"},{"ID":"TRGT_PRC","NM":"목표주가"}],"data":[{"TRD_DT":"2025/07/11","TRGT_PRC":75792.0}]}
});
</script>
"""


def test_ingest_fnguide_metrics_upserts_financial_and_target_chart_rows(tmp_path, monkeypatch):
    db = str(tmp_path / "fg1.db")
    init_db(db)
    conn = connect(db)
    seed_kr_companies(conn, ["005930"])
    conn.close()

    monkeypatch.setattr(fnguide_metrics, "send_slack_alert", lambda *a, **kw: None)
    fetcher = FakeFetcher(_SAMPLE_SNAPSHOT_HTML)
    result = ingest_fnguide_metrics(db_path=db, fetcher=fetcher)

    conn = connect(db)
    rows = conn.execute(
        "SELECT as_of_date, metric_key, metric_value, source FROM fnguide_metrics ORDER BY metric_key"
    ).fetchall()
    conn.close()
    metric_keys = {r["metric_key"] for r in rows}
    assert metric_keys == {"매출액", "consensus_target_price"}
    assert all(r["source"] == "fnguide" for r in rows)
    assert result["succeeded"] == 1
    assert result["failed"] == []


def test_ingest_fnguide_metrics_skips_failing_ticker_and_alerts(tmp_path, monkeypatch):
    db = str(tmp_path / "fg2.db")
    init_db(db)
    conn = connect(db)
    seed_kr_companies(conn, ["005930", "000660"])
    conn.close()

    alerts: list[str] = []
    monkeypatch.setattr(fnguide_metrics, "send_slack_alert", lambda msg, **kw: alerts.append(msg))
    fetcher = FailingFetcher(_SAMPLE_SNAPSHOT_HTML, fail_when="gicode=005930")
    result = ingest_fnguide_metrics(db_path=db, fetcher=fetcher)

    assert result["failed"] == ["005930"]
    assert result["succeeded"] == 1
    assert len(alerts) == 1

    conn = connect(db)
    rows = conn.execute("SELECT DISTINCT stock_code FROM fnguide_metrics").fetchall()
    conn.close()
    assert [r["stock_code"] for r in rows] == ["000660"]
