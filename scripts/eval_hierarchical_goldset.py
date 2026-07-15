"""HA-14 (AC18/AC20) 1회성 실측 스크립트 — goldset 재실행(신규 vs legacy) + 성능 실측.

사용:
  python3 scripts/eval_hierarchical_goldset.py --limit 5            # goldset 5문항 + 성능
  python3 scripts/eval_hierarchical_goldset.py --limit 70 --mode goldset
  python3 scripts/eval_hierarchical_goldset.py --mode perf          # 성능만

동작(AC18):
  - 같은 goldset subset 을 (a) legacy 파이프라인(src/eval/runner.run_evaluation, 손대지 않음)과
    (b) 신규 계층형 그래프(src/agents/graph.run_hierarchical)로 나란히 재실행한다.
  - legacy 는 결과셋 denotation 비교(EX%), 신규는 LLM judge 로 정답 SQL 결과와 대조(judge %).
  - 결과를 .omc/research/hierarchical-goldset-comparison.md 에 문서화한다.

동작(AC20):
  - 대표 질문 세트(KR 재무/주가, US, macro, backtest)를 run_hierarchical 로 실행하며
    응답시간(초)·llm_fn 호출횟수(CountingLLM)를 실측한다. 상한 판정 없이 수치만 기록.
  - .omc/research/hierarchical-performance.md 에 문서화한다.

주의: 실제 OpenAI API 를 호출한다(비용/시간 발생). 원본 DB 는 항상 격리 사본에서만 읽는다.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.graph import run_hierarchical
from src.config import CONFIG
from src.db import connect_readonly
from src.eval.goldset import GOLDSET
from src.eval.hierarchical_runner import CountingLLM, run_hierarchical_eval
from src.eval.runner import _cleanup_copy, _isolated_copy
from src.llm import LLMClient

RESEARCH_DIR = Path(__file__).resolve().parent.parent / ".omc" / "research"

# AC20 대표 질문 세트 — 4개 도메인을 고루 커버(백테스트는 steps 없이 라우팅/검증 경로만 측정).
_PERF_QUESTIONS = [
    ("KR-재무", "삼성전자 PER 알려줘"),
    ("KR-재무", "SK하이닉스 영업이익 알려줘"),
    ("KR-주가", "삼성전자 주가 이동평균 알려줘"),
    ("US", "애플 PER 알려줘"),
    ("US", "AAPL 주가 알려줘"),
    ("US", "테슬라 순이익 알려줘"),
    ("macro", "지금 매크로 신호 알려줘"),
    ("macro", "장단기 금리차 스프레드 레짐이 어때?"),
    ("backtest", "저PER 20개 종목 분기 리밸런싱 전략 백테스트"),
]


def _build_llm_fn(role: str):
    """web/app.py._build_llm_fn 과 동일한 규약(Callable[[str],str]). 미가용 시 None."""
    client = LLMClient()
    if not client.available:
        return None
    return lambda prompt: (client.complete(prompt, role=role).text or "")


def run_goldset(items: list[dict], llm_fn, judge_llm_fn) -> dict:
    """신규 계층형 구조로 goldset subset 재실행 + judge 판정(격리 사본)."""
    return run_hierarchical_eval(
        items, run_hierarchical, llm_fn=llm_fn, judge_llm_fn=judge_llm_fn,
        db_path=CONFIG.db_path, connect_ro_fn=connect_readonly,
    )


def run_legacy(limit: int) -> dict:
    """legacy 파이프라인 EX% baseline(run_evaluation 은 손대지 않고 그대로 호출)."""
    from src.eval.runner import run_evaluation

    return run_evaluation(limit=limit)


def run_perf(llm_fn) -> dict:
    """대표 질문 세트 성능 실측(응답시간/LLM 호출횟수). 격리 사본에서 실행."""
    copy_path = _isolated_copy(CONFIG.db_path)
    counter = CountingLLM(llm_fn)
    rows: list[dict] = []
    try:
        for domain, question in _PERF_QUESTIONS:
            counter.reset()
            conn = connect_readonly(copy_path)
            steps = [{"op": "run_backtest", "params": {}, "out": "bt"}] if domain == "backtest" else None
            t0 = time.perf_counter()
            try:
                hier = run_hierarchical(question, conn, llm_fn=counter, steps=steps)
                err = None
            except Exception as exc:  # noqa: BLE001
                hier = {}
                err = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - t0
            conn.close()
            rows.append({
                "domain": domain, "question": question,
                "latency_s": round(latency, 2), "llm_calls": counter.count,
                "routes": hier.get("routes"), "uncertain": bool(hier.get("uncertain")),
                "error": err,
            })
    finally:
        _cleanup_copy(copy_path)
    lat = [r["latency_s"] for r in rows]
    calls = [r["llm_calls"] for r in rows]
    return {
        "n": len(rows),
        "avg_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
        "avg_llm_calls": round(sum(calls) / len(calls), 2) if calls else None,
        "rows": rows,
    }


def _fmt_goldset_md(hier: dict, legacy: dict, n: int, model: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lex = legacy["execution_accuracy"]
    lines = [
        "# 신규 계층형 vs legacy — goldset 재실행 정확도 비교 (AC18)",
        "",
        f"- 실측 일시: {ts}",
        f"- 모델: {model}",
        f"- 실행 문항 수: {n} (goldset 전체 {len(GOLDSET)}문항 중 앞 {n}문항)",
        f"- DB: 원본({CONFIG.db_path})의 격리 사본에서 실행(원본 보호)",
        "",
        "## 요약",
        "",
        "| 구조 | 정확도 | 판정 방식 | 비고 |",
        "|---|---|---|---|",
        f"| legacy 6단계 파이프라인 | {lex['ex_pct']}% ({lex['match']}/{lex['applicable']}) | "
        "결과셋 denotation 비교(EX) | 정답 SQL 결과와 행 단위 일치 |",
        f"| 신규 계층형(run_hierarchical) | {hier['accuracy_pct']}% ({hier['match']}/{hier['judged']}) | "
        "LLM judge(정답 SQL 결과 vs 종합결론+도메인결과) | judged=판정가능 문항 |",
        "",
        "> 두 수치는 **판정 방식이 다르다**(legacy=행 단위 denotation, 신규=LLM judge). "
        "신규 구조의 산출물은 결과셋이 아니라 종합결론+도메인 원본결과라 직접 행 비교가 불가능해 "
        "judge 로 대체했다. 따라서 완전한 apples-to-apples 는 아니며 '각 구조가 질문에 옳게 답했는가'의 근사 비교다.",
        "",
        f"- 신규 구조 평균 응답시간: {hier['avg_latency_s']}s / 평균 LLM 호출: {hier['avg_llm_calls']}회 (문항당)",
        "",
        "## 문항별 (신규 계층형)",
        "",
        "| id | tags | 질문 | 라우팅 | 불확실 | judge | match | 응답(s) | LLM호출 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in hier["rows"]:
        q = (r["question"] or "").replace("|", "/")[:40]
        lines.append(
            f"| {r['id']} | {r['tags']} | {q} | {r['routes']} | "
            f"{'Y' if r['uncertain'] else ''} | {'O' if r['judge_applicable'] else 'X'} | "
            f"{'✅' if r['match'] else ('❌' if r['match'] is False else '-')} | "
            f"{r['latency_s']} | {r['llm_calls']} |"
        )
    lines += [
        "",
        "## 해석 메모",
        "",
        "- goldset 문항은 대부분 **Top-N 랭킹/스크리닝**(예: 'PER이 가장 낮은 10개 회사')이다.",
        "- 신규 계층형 도메인 에이전트(answer_kr_question/answer_us_question)는 **단일 종목 조회** "
        "지향(질문에서 종목 1개를 찾아 재무/주가를 조회)이라, 전체 유니버스 랭킹을 산출하는 legacy "
        "SQL 생성 경로와 설계 목적이 다르다. 이 구조적 차이가 정확도 격차의 주원인인지 아래 문항별 "
        "라우팅/불확실 컬럼으로 확인할 수 있다.",
    ]
    return "\n".join(lines) + "\n"


def _fmt_perf_md(perf: dict, model: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 신규 계층형 구조 성능 실측 (AC20)",
        "",
        f"- 실측 일시: {ts}",
        f"- 모델: {model}",
        f"- 대표 질문 수: {perf['n']} (KR 재무/주가, US, macro, backtest 도메인 커버)",
        "- 상한 판정 없이 수치만 기록(요청사항 AC20).",
        "",
        f"**평균 응답시간: {perf['avg_latency_s']}s · 평균 LLM 호출: {perf['avg_llm_calls']}회 (질문당)**",
        "",
        "| 도메인 | 질문 | 응답(s) | LLM호출 | 라우팅 | 불확실 | 오류 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in perf["rows"]:
        q = (r["question"] or "").replace("|", "/")[:40]
        lines.append(
            f"| {r['domain']} | {q} | {r['latency_s']} | {r['llm_calls']} | "
            f"{r['routes']} | {'Y' if r['uncertain'] else ''} | {r['error'] or ''} |"
        )
    lines += [
        "",
        "> 비용: 모델 unit cost 가 src/config.py 에 정의돼 있지 않아 금액 추정은 생략한다"
        "(호출횟수만 실측). 백테스트 도메인은 실제 steps 없이 라우팅/검증 경로만 측정한 값이다.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5, help="goldset 재실행 문항 수(앞에서부터)")
    ap.add_argument("--mode", choices=["both", "goldset", "perf"], default="both")
    args = ap.parse_args()

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    llm_fn = _build_llm_fn("sql")
    judge_llm_fn = _build_llm_fn("judge")
    model = CONFIG.summary()
    if llm_fn is None:
        print("[경고] LLM 미가용(키 없음) — 신규 구조는 휴리스틱 폴백으로 실행됨")

    if args.mode in ("both", "goldset"):
        items = GOLDSET[: args.limit]
        print(f"[goldset] 신규 계층형 재실행 {len(items)}문항 …")
        hier = run_goldset(items, llm_fn, judge_llm_fn)
        print(f"[goldset] legacy baseline 재실행 {len(items)}문항 …")
        legacy = run_legacy(args.limit)
        md = _fmt_goldset_md(hier, legacy, len(items), model)
        out = RESEARCH_DIR / "hierarchical-goldset-comparison.md"
        out.write_text(md, encoding="utf-8")
        print(f"[goldset] 기록: {out}")
        print(f"  신규 judge 정확도={hier['accuracy_pct']}%  legacy EX={legacy['execution_accuracy']['ex_pct']}%")

    if args.mode in ("both", "perf"):
        print(f"[perf] 대표 질문 {len(_PERF_QUESTIONS)}개 성능 실측 …")
        perf = run_perf(llm_fn)
        md = _fmt_perf_md(perf, model)
        out = RESEARCH_DIR / "hierarchical-performance.md"
        out.write_text(md, encoding="utf-8")
        print(f"[perf] 기록: {out}")
        print(f"  평균 응답시간={perf['avg_latency_s']}s  평균 LLM호출={perf['avg_llm_calls']}회")


if __name__ == "__main__":
    main()
