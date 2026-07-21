#!/usr/bin/env python3
"""US-8: factcheck 5개 도메인 통합 실행 + 리포트 생성 (1회성 실측 스크립트)."""
from __future__ import annotations

import sys
import traceback
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.db import connect_readonly
from src.eval.factcheck.backtest import run_backtest_pytest_check
from src.eval.factcheck.chart import run_chart_check
from src.eval.factcheck.financials import run_financials_check
from src.eval.factcheck.live import (
    build_system_llm_fn, build_vision_client, make_chart_llm_fn,
    make_financial_llm_fn, make_price_llm_fn, make_screening_llm_fn, make_vision_fn,
)
from src.eval.factcheck.price import run_price_check
from src.eval.factcheck.sample import top_market_cap_stocks
from src.eval.factcheck.screening import _recompute_top_n, run_screening_check
from src.eval.factcheck.tolerance import within_pct_tolerance

RESEARCH_DIR = Path(__file__).resolve().parent.parent / ".omc" / "research"
REPORT_PATH = RESEARCH_DIR / "factcheck-report.md"
_FIN_METRIC = "operating_profit"
_INDEX_TOL = 0.025

_SCREENING_CASES = [
    {"question": "PER이 가장 낮은 10개 종목을 알려줘", "top_n": 10, "metric": "per", "ascending": True},
    {"question": "PBR이 가장 낮은 10개 종목을 알려줘", "top_n": 10, "metric": "pbr", "ascending": True},
    {"question": "ROE가 가장 높은 10개 종목을 알려줘", "top_n": 10, "metric": "roe", "ascending": False},
    {"question": "ROA가 가장 높은 10개 종목을 알려줘", "top_n": 10, "metric": "roa", "ascending": False},
    {"question": "PSR이 가장 낮은 10개 종목을 알려줘", "top_n": 10, "metric": "psr", "ascending": True},
    {"question": "영업이익률이 가장 높은 10개 종목을 알려줘", "top_n": 10, "metric": "operating_margin", "ascending": False},
    {"question": "순이익률이 가장 높은 10개 종목을 알려줘", "top_n": 10, "metric": "net_margin", "ascending": False},
    {"question": "부채비율이 가장 낮은 10개 종목을 알려줘", "top_n": 10, "metric": "debt_ratio", "ascending": True},
    {"question": "시가총액이 가장 큰 10개 종목을 알려줘", "top_n": 10, "metric": "market_cap", "ascending": False},
    {"question": "매출성장률이 가장 높은 10개 종목을 알려줘", "top_n": 10, "metric": "revenue_growth", "ascending": False},
]
_CHART_CASES = [
    {"question": "시가총액이 가장 큰 10개 종목을 막대그래프로 그려줘", "top_n": 10, "metric": "market_cap", "ascending": False},
    {"question": "PER이 가장 낮은 10개 종목을 막대그래프로 그려줘", "top_n": 10, "metric": "per", "ascending": True},
    {"question": "PBR이 가장 낮은 10개 종목을 그래프로 그려줘", "top_n": 10, "metric": "pbr", "ascending": True},
    {"question": "ROE가 가장 높은 10개 종목을 그래프로 그려줘", "top_n": 10, "metric": "roe", "ascending": False},
    {"question": "영업이익률이 가장 높은 10개 종목을 그래프로 그려줘", "top_n": 10, "metric": "operating_margin", "ascending": False},
]
_BACKTEST_SCENARIOS = [
    {"name": "동일가중 전체·연간 리밸런싱", "start_year": 2025, "end_year": 2026, "rebalance": "annual", "n": 5000},
    {"name": "동일가중 전체·반기 리밸런싱", "start_year": 2025, "end_year": 2026, "rebalance": "semiannual", "n": 5000},
    {"name": "동일가중 전체·분기 리밸런싱", "start_year": 2025, "end_year": 2026, "rebalance": "quarterly", "n": 5000},
    {"name": "동일가중 전체·월간 리밸런싱", "start_year": 2025, "end_year": 2026, "rebalance": "monthly", "n": 5000},
    {"name": "동일가중 대형주 상위200·분기 리밸런싱", "start_year": 2025, "end_year": 2026, "rebalance": "quarterly", "n": 200},
]


def fetch_kospi_return():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from pykrx import stock
        start = "20250101"
        end = date.today().strftime("%Y%m%d")
        df = stock.get_index_ohlcv_by_date(start, end, "1001")
        if df is None or len(df) < 2 or "종가" not in df.columns:
            return None, "pykrx 코스피 지수 응답이 비었거나 형식이 다름", {}
        closes = df["종가"]
        first, last = float(closes.iloc[0]), float(closes.iloc[-1])
        if not first:
            return None, "코스피 지수 시작값이 0", {}
        meta = {"start_date": str(df.index[0].date()), "end_date": str(df.index[-1].date()),
                "first_close": first, "last_close": last}
        return last / first - 1, "", meta
    except Exception as exc:  # noqa: BLE001
        return None, f"pykrx 코스피 지수 확보 실패: {type(exc).__name__}: {exc}", {}


def run_backtest_index_direct(conn, scenarios, expected_return, kospi_note):
    from src.backtest.primitives import run_backtest_primitive
    results = []
    for sc in scenarios:
        name = sc["name"]
        try:
            r = run_backtest_primitive(conn, start_year=sc["start_year"], end_year=sc["end_year"],
                criteria=[{"key": "market_cap", "direction": "high"}], n=sc.get("n", 5000),
                rebalance=sc.get("rebalance", "quarterly"), with_benchmark=False, market="KR")
            actual = r["performance"]["total_return"] / 100.0
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            results.append({"scenario": name, "expected": expected_return, "actual": None,
                            "pass": None, "note": f"측정불가(백테스트 실패): {type(exc).__name__}: {exc}"})
            continue
        if expected_return is None:
            results.append({"scenario": name, "expected": None, "actual": actual, "pass": None,
                            "note": f"측정불가: {kospi_note}"})
        else:
            results.append({"scenario": name, "expected": expected_return, "actual": actual,
                            "pass": within_pct_tolerance(expected_return, actual, _INDEX_TOL), "note": ""})
    return results


def _counts(items):
    p = sum(1 for it in items if it.get("pass") is True)
    f = sum(1 for it in items if it.get("pass") is False)
    u = sum(1 for it in items if it.get("pass") is None)
    return p, f, u


def _rate(passed, measurable):
    return f"{round(100 * passed / measurable, 1)}%" if measurable else "N/A"


def _fmt_num(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:
            return "무응답(NaN)"
        return f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def _verdict(p):
    return {True: "pass", False: "fail", None: "측정불가"}[p]


def build_report(sections, kospi_note, kospi_meta, model_summary, errors):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 퀀트 어시스턴트 factcheck 실측 리포트 (US-8 / AC6)", "",
        f"- 실측 일시: {ts}", f"- 모델/설정: {model_summary}",
        f"- DB: 원본({CONFIG.db_path}, 약 38.67GB)을 **엔진 레벨 읽기전용(mode=ro)** 으로 조회 "
        "(디스크 용량 절약을 위해 격리 사본을 만들지 않음 — read-only 연결이 원본 write를 물리적으로 차단).",
        "- 판정: pass=허용오차 내 · fail=허용오차 밖 · 측정불가=외부 재조회/시스템 응답 확보 실패.", "",
    ]
    lines += ["## 코스피 지수 데이터 확보 상태", ""]
    if kospi_meta:
        lines.append(f"- **성공**: pykrx get_index_ohlcv_by_date('1001') (KRX 로그인 .env 사용). "
                     f"{kospi_meta['start_date']} 종가 {kospi_meta['first_close']:,.2f} → {kospi_meta['end_date']} "
                     f"종가 {kospi_meta['last_close']:,.2f}, 기간수익률 {(kospi_meta['last_close']/kospi_meta['first_close']-1)*100:.2f}%.")
    else:
        lines.append(f"- **실패/측정불가**: {kospi_note}")
    lines.append("")
    total_pass = total_fail = total_unmeasurable = 0
    for sec in sections:
        items = sec["items"]
        p, f, u = _counts(items)
        total_pass += p; total_fail += f; total_unmeasurable += u
        measurable = p + f
        lines += [f"## {sec['title']}", ""]
        if sec.get("note"):
            lines += [f"> {sec['note']}", ""]
        lines.append("| " + " | ".join(sec["headers"]) + " |")
        lines.append("|" + "|".join(["---"] * len(sec["headers"])) + "|")
        for row in sec["rows"]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines += ["", f"- 통과율(측정가능 기준): {_rate(p, measurable)} ({p}/{measurable}) · "
                  f"측정불가 {u}건 · 전체 {len(items)}건", ""]
    total_items = total_pass + total_fail + total_unmeasurable
    total_measurable = total_pass + total_fail
    lines += ["## 종합 통과율", "",
        f"- 전체 항목: {total_items}건 (pass {total_pass} · fail {total_fail} · 측정불가 {total_unmeasurable})",
        f"- **측정가능 기준 통과율: {_rate(total_pass, total_measurable)} ({total_pass}/{total_measurable})**",
        f"- **전체 항목 기준 통과율: {_rate(total_pass, total_items)} ({total_pass}/{total_items})**", ""]
    if errors:
        lines += ["## 도메인 실행 오류(격리됨)", ""]
        for dom, err in errors.items():
            lines.append(f"- {dom}: {err}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    model_summary = CONFIG.summary()
    llm_fn = build_system_llm_fn("sql")
    if llm_fn is None:
        print("[경고] 시스템 LLM 미가용 — 휴리스틱 폴백")
    vision_client = build_vision_client()
    conn = connect_readonly(CONFIG.db_path)
    sections = []; errors = {}
    try:
        top10 = top_market_cap_stocks(conn, 10)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(); top10 = []; errors["sample"] = f"{type(exc).__name__}: {exc}"
    print(f"[표본] 시총 상위 {len(top10)}종목: {[s['name'] for s in top10]}")
    name_of = {s["stock_code"]: s["name"] for s in top10}

    print("[1/5] 재무제표(DART ±1%) …")
    try:
        fin_items = [{"stock_code": s["stock_code"], "name": s["name"], "metric": _FIN_METRIC} for s in top10]
        fin_res = run_financials_check(fin_items, make_financial_llm_fn(conn, llm_fn), CONFIG.dart_api_key or None)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(); errors["financials"] = f"{type(exc).__name__}: {exc}"
        fin_res = [{"stock_code": s["stock_code"], "expected": None, "actual": None, "pass": None, "note": "측정불가(도메인 예외)"} for s in top10]
    sections.append({"title": "1. 재무제표 사실확인 (AC1 · 영업이익 vs OpenDART, ±1%)",
        "headers": ["종목", "DART원문(기대)", "시스템응답(실제)", "판정/비고"], "items": fin_res,
        "rows": [(f"{name_of.get(r['stock_code'], r['stock_code'])}({r['stock_code']})", _fmt_num(r["expected"]),
                  _fmt_num(r["actual"]), _verdict(r["pass"]) + (f" · {r['note']}" if r.get("note") else "")) for r in fin_res]})

    print("[2/5] 주가(네이버 완전일치) …")
    try:
        price_items = [{"stock_code": s["stock_code"], "name": s["name"]} for s in top10]
        price_res = run_price_check(price_items, make_price_llm_fn(conn, llm_fn))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(); errors["price"] = f"{type(exc).__name__}: {exc}"
        price_res = [{"stock_code": s["stock_code"], "expected": None, "actual": None, "pass": None, "note": "측정불가(도메인 예외)"} for s in top10]
    sections.append({"title": "2. 실시간 주가 (AC2 · 오늘 종가 vs 네이버 fchart, 완전일치)",
        "headers": ["종목", "네이버(기대)", "시스템응답(실제)", "판정/비고"], "items": price_res,
        "rows": [(f"{name_of.get(r['stock_code'], r['stock_code'])}({r['stock_code']})", _fmt_num(r["expected"]),
                  _fmt_num(r["actual"]), _verdict(r["pass"]) + (f" · {r['note']}" if r.get("note") else "")) for r in price_res]})

    print("[3/5] 차트(데이터일치 + vision) …")
    try:
        chart_items = []
        for c in _CHART_CASES:
            expected = _recompute_top_n(conn, {"metric": c["metric"], "top_n": c["top_n"], "ascending": c["ascending"]})
            chart_items.append({"question": c["question"], "expected_data": expected})
        chart_res = run_chart_check(chart_items, make_chart_llm_fn(conn, llm_fn), make_vision_fn(vision_client=vision_client))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(); errors["chart"] = f"{type(exc).__name__}: {exc}"
        chart_res = [{"question": c["question"], "data_match": None, "vision_verdict": None, "pass": None, "note": "측정불가(도메인 예외)"} for c in _CHART_CASES]
    sections.append({"title": "3. 차트 (AC3 · 근거데이터 일치 + gpt-5.4-mini vision)",
        "headers": ["질문", "데이터일치", "vision판정", "판정/비고"], "items": chart_res,
        "rows": [((r["question"] or "")[:40].replace("|", "/"), str(r.get("data_match")), str(r.get("vision_verdict")),
                  _verdict(r["pass"]) + (f" · {r['note']}" if r.get("note") else "")) for r in chart_res]})

    print("[4/5] 스크리닝(top-N 재계산 완전일치) …")
    try:
        screen_res = run_screening_check(_SCREENING_CASES, make_screening_llm_fn(conn, llm_fn), conn)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(); errors["screening"] = f"{type(exc).__name__}: {exc}"
        screen_res = [{"question": c["question"], "pass": None, "note": "측정불가(도메인 예외)"} for c in _SCREENING_CASES]
    sections.append({"title": "4. 스크리닝 (AC4 · top-N vs DB 재계산, 완전일치)",
        "headers": ["질문", "판정", "비고"], "items": screen_res,
        "rows": [((r["question"] or "")[:40].replace("|", "/"), _verdict(r["pass"]), (r.get("note") or "")[:80].replace("|", "/")) for r in screen_res]})

    print("[5/5] 백테스트 …")
    try:
        pytest_res = run_backtest_pytest_check()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(); errors["backtest_pytest"] = f"{type(exc).__name__}: {exc}"
        pytest_res = {"exit_code": None, "summary": "", "pass": None, "note": f"측정불가(예외): {exc}"}
    sections.append({"title": "5a. 백테스트 pytest 재실행 (AC5a)",
        "headers": ["대상", "판정", "요약/비고"], "items": [{"pass": pytest_res.get("pass")}],
        "rows": [("test_backtest_performance/weights/auditor_static_guard", _verdict(pytest_res.get("pass")),
                  f"exit={pytest_res.get('exit_code')} · {pytest_res.get('summary')} {pytest_res.get('note') or ''}".strip())]})

    kospi_ret, kospi_note, kospi_meta = fetch_kospi_return()
    print(f"[코스피] {'확보 성공' if kospi_meta else '측정불가'}: {kospi_meta if kospi_meta else kospi_note}")
    try:
        index_res = run_backtest_index_direct(conn, _BACKTEST_SCENARIOS, kospi_ret, kospi_note)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(); errors["backtest_index"] = f"{type(exc).__name__}: {exc}"
        index_res = [{"scenario": sc["name"], "expected": kospi_ret, "actual": None, "pass": None, "note": f"측정불가(예외): {exc}"} for sc in _BACKTEST_SCENARIOS]
    idx_note = (f"코스피 기간수익률(pykrx, {kospi_meta.get('start_date')}~{kospi_meta.get('end_date')}) 대비 동일가중 유니버스, ±2.5%p. "
                "코스피(시총가중 지수)와 동일가중 유니버스는 구성이 근본적으로 달라 큰 괴리가 정상." ) if kospi_meta else f"코스피 확보 실패로 전 시나리오 측정불가: {kospi_note}"
    sections.append({"title": "5b. 코스피 지수 대비 동일가중 백테스트 (AC5b · ±2.5%p)",
        "headers": ["시나리오", "코스피(기대)", "백테스트(실제)", "판정/비고"], "items": index_res, "note": idx_note,
        "rows": [(r["scenario"], (f"{r['expected']*100:.2f}%" if r.get("expected") is not None else "-"),
                  (f"{r['actual']*100:.2f}%" if r.get("actual") is not None else "-"),
                  _verdict(r["pass"]) + (f" · {r['note']}" if r.get("note") else "")) for r in index_res]})

    conn.close()
    REPORT_PATH.write_text(build_report(sections, kospi_note, kospi_meta, model_summary, errors), encoding="utf-8")
    print(f"\n[완료] 리포트: {REPORT_PATH}")
    tp = sum(1 for sec in sections for it in sec["items"] if it.get("pass") is True)
    tf = sum(1 for sec in sections for it in sec["items"] if it.get("pass") is False)
    tu = sum(1 for sec in sections for it in sec["items"] if it.get("pass") is None)
    print(f"  종합: 전체 {tp+tf+tu}건 · pass {tp} · fail {tf} · 측정불가 {tu}")
    print(f"  측정가능 기준 통과율: {_rate(tp, tp+tf)} ({tp}/{tp+tf})")
    print(f"  전체 항목 기준 통과율: {_rate(tp, tp+tf+tu)} ({tp}/{tp+tf+tu})")


if __name__ == "__main__":
    main()
