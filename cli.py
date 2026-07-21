#!/usr/bin/env python3
"""Quant Assistant — CLI 진입점 (레거시 6노드 SQL 생성 파이프라인 + 질의기록 위키).

사용 예:
  python cli.py setup-dummy                 # 더미 데이터 생성
  python cli.py query "PER이 낮은 5개 회사"   # 질의
  python cli.py eval --limit 10              # 3층 평가
  python cli.py wiki-eff                     # 위키 효율 리포트
  python cli.py wiki list                    # 위키 목록
  python cli.py wiki verify 3                # 위키 항목 검증
"""
from __future__ import annotations

import argparse

from src.config import CONFIG


# ---------------------------------------------------------------------------
# 출력 헬퍼
# ---------------------------------------------------------------------------
# 컬럼명에 이 키워드가 들어있으면 비율/배수 지표로 간주해 통화기호를 붙이지 않는다
# (per/roe 등은 원/달러 어느 시장이든 통화 무관 — US 프롬프트의 currency 컬럼 관례와 짝을 이룸).
_RATIO_COLUMN_HINTS = ("per", "pbr", "psr", "roe", "roa", "rate", "ratio", "margin", "pct", "percent")


def print_table(columns: list[str], rows: list[dict], max_rows: int = 30) -> None:
    if not rows:
        print("  (결과 없음)")
        return
    all_cols = columns or list(rows[0].keys())
    # currency는 표시용 메타 정보일 뿐 그 자체를 열로 보여주지 않는다(값에 $ 기호로 이미 반영됨).
    cols = [c for c in all_cols if c != "currency"]
    widths = {c: len(str(c)) for c in cols}
    shown = rows[:max_rows]
    for r in shown:
        currency = r.get("currency")
        for c in cols:
            cur = currency if not any(h in c.lower() for h in _RATIO_COLUMN_HINTS) else None
            widths[c] = max(widths[c], len(_fmt(r.get(c), currency=cur)))
    header = " | ".join(str(c).ljust(widths[c]) for c in cols)
    print("  " + header)
    print("  " + "-+-".join("-" * widths[c] for c in cols))
    for r in shown:
        currency = r.get("currency")
        cells = []
        for c in cols:
            cur = currency if not any(h in c.lower() for h in _RATIO_COLUMN_HINTS) else None
            cells.append(_fmt(r.get(c), currency=cur).ljust(widths[c]))
        print("  " + " | ".join(cells))
    if len(rows) > max_rows:
        print(f"  ... ({len(rows) - max_rows}행 생략, 총 {len(rows)}행)")


def _fmt(v, currency: str | None = None) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        if currency == "USD":
            return f"${v:,.2f}"
        if abs(v) >= 1e12:
            return f"{v/1e12:.2f}조"
        if abs(v) >= 1e8:
            return f"{v/1e8:.1f}억"
        return f"{v:,.2f}"
    return str(v)


# ---------------------------------------------------------------------------
# 백테스트 결과(performance+holdings 중첩 dict) 전용 출력 포맷
# ---------------------------------------------------------------------------
_PERFORMANCE_LABELS = {
    "total_return": "누적수익률",
    "cagr": "CAGR",
    "mdd": "MDD",
    "volatility": "변동성",
    "sharpe": "샤프",
    "sortino": "소르티노",
    "win_rate": "승률",
    "avg_turnover": "회전율",
    "benchmark_return": "벤치마크수익률",
    "excess_return": "초과수익률",
    "beta": "베타",
}


def _is_backtest_result(row: dict) -> bool:
    """파이프라인 결과 한 행이 run_backtest 프리미티브 결과(성과+보유종목 중첩)인지 판별."""
    return isinstance(row, dict) and "performance" in row and "holdings" in row


_RATIO_KEYS = {"sharpe", "sortino", "beta"}  # 비율(%) 아닌 순수 지표


def format_backtest_result(row: dict) -> str:
    """백테스트 결과를 성과 요약 + 리밸런싱 시점별 보유종목 목록 문자열로 렌더링한다."""
    perf = row.get("performance") or {}
    lines = []
    summary = "  ".join(
        f"{_PERFORMANCE_LABELS.get(k, k)} {v}{'' if k in _RATIO_KEYS else '%'}"
        for k, v in perf.items() if k != "periods"
    )
    lines.append(f"성과: {summary}")
    holdings = row.get("holdings") or []
    if holdings:
        lines.append("\n리밸런싱 종목:")
        for h in holdings:
            names = h.get("names") or h.get("codes") or []
            lines.append(f"  {h.get('date')}: {', '.join(names)}")
    return "\n".join(lines)


# 소프트경고(감지된 위험) 죄악명 한국어 라벨 — 백테스트 감사 레이어(src/backtest/auditor.py)
_SIN_LABELS = {
    "storytelling": "스토리텔링·데이터의 역사",
    "snooping": "데이터마이닝·스누핑",
    "signal_decay": "신호감소·회전율",
    "outlier": "이상치 제어",
    "survivorship": "생존편향 검증불가",  # US 백테스트: 상장폐지 데이터 없음(검증불가)
}


def format_audit_warnings(warnings: list[dict]) -> str:
    """백테스트 감사 소프트경고를 정상 결과 아래에 붙일 '감지된 위험' 섹션으로 렌더링한다(AUD-8).

    하드차단과 달리 정상 결과를 가리거나 대체하지 않고, 검토 권장 경고로만 첨부한다(AC10).
    """
    lines = ["", "감지된 위험 (소프트경고 — 결과는 유효하나 방법론 검토 권장):"]
    for w in warnings:
        label = _SIN_LABELS.get(w.get("sin"), w.get("sin"))
        lines.append(f"  - [{label}] {w.get('message', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 명령 구현
# ---------------------------------------------------------------------------
def cmd_setup_dummy(args):
    from src.ingest.dummy import generate_dummy

    print("더미 데이터 생성 중...")
    info = generate_dummy()
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("완료. 이제 `python cli.py query \"...\"` 로 질의하세요.")


def cmd_query(args):
    from src.factors.fama_french import handle_query
    from src.legacy.pipeline import Pipeline

    print(f"[설정] {CONFIG.summary()}\n")

    factor_result = handle_query(args.question)
    if factor_result is not None:
        print(factor_result)
        return

    p = Pipeline()
    try:
        s = p.run(args.question, do_eval=args.eval)
    finally:
        pass
    print(f"질문(원본) : {s.get('raw_question')}")
    print(f"질문(정제) : {s.get('question')}")
    print(f"라우팅     : {s.get('route')}   data_version: {s.get('data_version')}")
    rec = f"기록 #{s.get('wiki_id')} 저장됨" if s.get("wiki_id") else "기록 안 함"
    print(f"기록 상태  : {rec}   (캐시 도출 없음: 항상 새로 생성)")
    print(f"SQL 출처   : {s.get('sql_source')}")
    print(f"\nSQL:\n  {s.get('sql')}\n")
    if s.get("error"):
        print(f"실행 오류: {s.get('error')}")
    else:
        rows = s.get("rows", [])
        if s.get("sql_source") == "pipeline" and rows and _is_backtest_result(rows[0]):
            print(format_backtest_result(rows[0]))
        else:
            print(f"결과 ({s.get('row_count')}행):")
            print_table(s.get("columns", []), rows)
        # 백테스트 감사 소프트경고는 정상 결과 아래에 별도 '감지된 위험' 섹션으로 첨부(AUD-8/AC10)
        warnings = s.get("audit_warnings")
        if warnings:
            print(format_audit_warnings(warnings))
        if s.get("chart_path"):
            print(f"\n차트: {s.get('chart_path')}")
    if args.eval and s.get("evaluation"):
        ev = s["evaluation"]
        print(f"\n평가: {ev.get('summary')}")
        if ev["layer2"].get("applicable"):
            print(f"  Judge 사유: {ev['layer2'].get('reason')}")
    p.close()


def cmd_eval(args):
    import json

    from src.eval.runner import format_eval_report, run_evaluation

    rep = run_evaluation(limit=args.limit, offline=args.offline)
    if args.json:
        # 러너(donegate)가 execution_accuracy.ex_pct를 파싱할 수 있도록,
        # stdout에는 리포트 dict의 JSON만 출력한다(부가 텍스트 없이).
        print(json.dumps(rep, ensure_ascii=False))
        return
    off = " [offline 강제]" if args.offline else ""
    print(f"[설정] {CONFIG.summary()}{off}")
    print(f"정답셋 평가 실행 중 (limit={args.limit or '전체'})...\n")
    print(format_eval_report(rep))
    if args.verbose:
        print("\n[문항별]")
        for r in rep["rows"]:
            print(f"  #{r['id']:2d} EX={r['ex']} L3={r['l3']} src={r['sql_source']} | {r['question']}")


def cmd_wiki(args):
    from src.db import connect
    from src.wiki.store import WikiStore

    conn = connect()
    store = WikiStore(conn)
    try:
        if args.wiki_cmd == "list":
            pages = store.list_pages(tag=args.tag)
            print(f"기록 {len(pages)}개" + (f" (태그={args.tag})" if args.tag else ""))
            for p in pages:
                mark = "✓검증" if p["verified"] else " 미검증"
                print(f"  #{p['id']:3d} [{mark}] rows={p['result_rows']:3d} [{p['tags']}] {p['question']}")
        elif args.wiki_cmd == "show":
            p = store.get_page(args.id)
            if not p:
                print("없는 항목"); return
            for k, v in p.items():
                print(f"  {k}: {v}")
        elif args.wiki_cmd == "verify":
            ok = store.set_verified(args.id, True)
            print("검증 표시 완료" if ok else "실패: 없는 항목")
        elif args.wiki_cmd == "edit":
            ok = store.update_sql(args.id, args.sql, verified=True)
            print("SQL 수정 + 검증 완료" if ok else "실패: 없는 항목")
        elif args.wiki_cmd == "tag":
            ok = store.set_tags(args.id, args.tags)
            print("태그 수정 완료" if ok else "실패: 없는 항목")
        elif args.wiki_cmd == "delete":
            ok = store.delete(args.id)
            print("삭제 완료" if ok else "실패: 없는 항목")
        elif args.wiki_cmd == "stats":
            for k, v in store.stats().items():
                print(f"  {k}: {v}")
        else:
            print("wiki 하위명령: list | show | verify | edit | tag | delete | stats")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 파서
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant Assistant CLI — 레거시 SQL 생성 파이프라인 + 질의 기록")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup-dummy", help="더미 데이터 생성")

    p_q = sub.add_parser("query", help="자연어 질의")
    p_q.add_argument("question")
    p_q.add_argument("--eval", action="store_true", help="3층 평가도 수행")

    p_e = sub.add_parser("eval", help="정답셋 3층 평가")
    p_e.add_argument("--limit", type=int, default=None)
    p_e.add_argument("--verbose", "-v", action="store_true")
    p_e.add_argument("--offline", action="store_true",
                     help="LLM 강제 비활성화(휴리스틱 폴백만 사용, 결정론 보장)")
    p_e.add_argument("--json", action="store_true",
                     help="평가 리포트(dict)를 JSON으로 stdout 출력")

    p_w = sub.add_parser("wiki", help="기록 관리")
    wsub = p_w.add_subparsers(dest="wiki_cmd", required=True)
    w_list = wsub.add_parser("list"); w_list.add_argument("--tag", default=None)
    w_show = wsub.add_parser("show"); w_show.add_argument("id", type=int)
    w_ver = wsub.add_parser("verify"); w_ver.add_argument("id", type=int)
    w_edit = wsub.add_parser("edit"); w_edit.add_argument("id", type=int); w_edit.add_argument("sql")
    w_tag = wsub.add_parser("tag"); w_tag.add_argument("id", type=int); w_tag.add_argument("tags")
    w_del = wsub.add_parser("delete"); w_del.add_argument("id", type=int)
    wsub.add_parser("stats")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "setup-dummy": cmd_setup_dummy,
        "query": cmd_query,
        "eval": cmd_eval,
        "wiki": cmd_wiki,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
