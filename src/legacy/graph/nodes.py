"""LangGraph 노드 구현 (순서대로).

1. refine_node    : 질문 LLM 정제
2. router_node    : 데이터 소스 판단(financial/price/both) → data_version 키 형식 결정
3. sql_gen_node   : 정제 질문 → SQL 생성 (LLM, 폴백=휴리스틱)
4. execute_node   : SQLite 실행 (캐시 도출 없음, 항상 새로 실행)
5. record_node    : 성공한 질의를 기록 로그(wiki 테이블)에 항상 저장
6. eval_node      : 3층 평가

설계 변경(기록 전용 전환)
------------------------
과거에는 질문 임베딩 유사도로 이전 SQL/결과를 캐시 재사용했으나, 그 캐시(도출)
기능을 완전히 폐기했다. 이제 wiki 테이블은 "질의 기록 로그"로만 쓰이며,
어떤 질의도 이전 기록에서 SQL/결과를 가져오지 않고 항상 새로 생성·실행한다.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date

from src.ingest.exchange_rate import get_usdkrw_rate, needs_exchange_rate
from src.ingest.price_live import ensure_live_prices, ensure_live_prices_us, needs_live_price
from src.llm import LLM, LLMClient, extract_json, extract_sql
from src.sql_exec import run_select
from src.version import build_data_version, now_iso
from src.wiki.store import WikiStore

from . import prompts
from .heuristic import detect_pipeline, detect_route, heuristic_sql
from .state import GraphState


@dataclass
class Deps:
    conn: sqlite3.Connection
    store: WikiStore
    schema: str
    today: date | None = None
    llm: LLMClient = LLM
    chart_dir: str = "data/charts"


def _note(state: GraphState, msg: str) -> list[str]:
    notes = list(state.get("notes", []))
    notes.append(msg)
    return notes


_BRACKET = re.compile(r"\[([A-Za-z_]\w*)\]")


def _sanitize_sql(sql: str) -> str:
    """로컬 모델이 자주 내는 비SQLite 표기 정리: [col]→col, 백틱 제거."""
    if not sql:
        return sql
    sql = _BRACKET.sub(r"\1", sql)  # f.[revenue] → f.revenue
    return sql.replace("`", "")


def _resolve_holdings_names(result, code_to_name: dict) -> dict:
    """파이프라인 결과(dict)에 holdings가 있으면 종목코드 옆에 이름을 붙인다(US-백테스트UX).

    holdings 키가 없으면 result를 그대로 반환한다(no-op). code_to_name에 없는 코드는
    코드 그대로 폴백한다(web/app.py의 기존 /api/backtest 변환 관례와 동일).
    """
    if not (isinstance(result, dict) and "holdings" in result):
        return result
    holdings = [
        {**h, "names": [code_to_name.get(c, c) for c in h.get("codes", [])]}
        for h in result["holdings"]
    ]
    return {**result, "holdings": holdings}


def _pipeline_result_rows(result) -> tuple[list[dict], list[str]]:
    """파이프라인 최종 결과를 표시용 (rows, columns)로 변환한다.

    - list[dict](횡단면 선정 결과) → 그대로 rows
    - dict(회귀/최적화 결과) → 한 줄 rows
    - dict{asset: weight}처럼 스칼라 매핑 → [{'key','value'}] 행들
    - 스칼라 → 한 줄
    """
    if isinstance(result, list):
        rows = [r for r in result if isinstance(r, dict)]
        cols = list(rows[0].keys()) if rows else []
        return rows, cols
    if isinstance(result, dict):
        # 값이 전부 스칼라(예: {종목:비중}, {slope:.., se_slope:..})면 한 줄로 표시
        rows = [result]
        return rows, list(result.keys())
    return [{"result": result}], ["result"]


def make_nodes(deps: Deps) -> dict:
    # ---- 1. refine ----
    def _with_exchange_rate(question: str) -> str:
        """절대금액(원/달러 혼동 위험) 질문이면 실시간 환율을 명시적으로 덧붙인다.

        SQL 생성 LLM은 실시간 환율을 모르므로 여기서 질문 텍스트에 직접 주입한다
        (.omc/specs/brainstorming-us-nl-sql-integration.md AC7 — refine_node 배선).
        """
        if not needs_exchange_rate(question):
            return question
        try:
            rate = get_usdkrw_rate(deps.conn, deps.today)
        except Exception:  # noqa: BLE001 — 환율 조회 실패는 질의를 막지 않는다(단위 언급 없이 진행)
            return question
        return f"{question} (환율 참고: 1달러={rate}원)"

    def refine_node(state: GraphState) -> dict:
        raw = state["raw_question"]
        if deps.llm.available:
            res = deps.llm.complete(
                prompts.REFINE_USER.format(question=raw),
                system=prompts.REFINE_SYSTEM,
                role="refine",
                max_tokens=200,
            )
            if res.ok and res.text.strip():
                refined = res.text.strip().splitlines()[0].strip().strip('"')
                refined = _with_exchange_rate(refined)
                return {"question": refined, "notes": _note(state, f"refine: LLM 정제")}
        return {"question": _with_exchange_rate(raw), "notes": _note(state, "refine: 원본 사용(LLM 미사용)")}

    # ---- 2. router ----
    def _mentioned_stock_codes(conn, table: str, text: str) -> list[str]:
        """text(질문)에 실제로 name이 언급된 종목의 stock_code만 골라 반환한다.

        실사용 재현 버그: 이 스코핑 없이 테이블 전체(company 3,924건 + us_company
        7,123건, 2026-07-14 실측)를 ensure_live_prices(_us)에 통째로 넘기면 "오늘
        삼성전자 종가"처럼 종목 하나만 필요한 질문에도 전체 종목의 실시간 시세를
        하나씩 외부 API(pykrx/yfinance)로 호출하게 돼 요청이 사실상 멈춘 것처럼
        보일 만큼(수천 건 순차 호출) 느려진다. instr()은 LIKE와 달리 name에 포함된
        와일드카드 문자(%,_) 이스케이프가 필요 없는 안전한 리터럴 부분일치다.
        """
        rows = conn.execute(
            f"SELECT stock_code FROM {table} WHERE name IS NOT NULL AND name != '' "
            "AND instr(?, name) > 0",
            (text,),
        ).fetchall()
        return [r[0] for r in rows]

    def router_node(state: GraphState) -> dict:
        # SQL로 표현 불가능한 통계/퀀트 분석 → route="pipeline"으로 분기(실시간 주가/데이터버전 불필요).
        # refine이 키워드를 지웠을 수 있어 원본(raw_question)도 함께 본다.
        if state.get("route") == "pipeline" or detect_pipeline(state.get("question", "")) \
                or detect_pipeline(state.get("raw_question", "")):
            return {"route": "pipeline",
                    "notes": _note(state, "router: route=pipeline (통계/퀀트 분석 감지)")}
        route = state.get("route") or detect_route(state["question"], state.get("sql"))
        notes = state.get("notes", [])
        # '오늘/현재' 류 주가 질의면 최신 종가를 DB에 보장(DB 우선 → 없으면 실시간 1건 → 저장).
        if needs_live_price(state["question"], route):
            try:
                mention_text = f"{state.get('question', '')} {state.get('raw_question', '')}"
                codes = _mentioned_stock_codes(deps.conn, "company", mention_text)
                info = ensure_live_prices(deps.conn, codes, deps.today)
                us_codes = _mentioned_stock_codes(deps.conn, "us_company", mention_text)
                us_info = ensure_live_prices_us(deps.conn, us_codes, deps.today)
                notes = _note(
                    state,
                    f"live_price: KR 실시간 {info['fetched']}건/캐시 {info['cached']}건, "
                    f"US 실시간 {us_info['fetched']}건/캐시 {us_info['cached']}건 ({info['today']})",
                )
            except Exception as exc:  # noqa: BLE001 — 실시간 호출 실패는 질의를 막지 않는다
                notes = _note(state, f"live_price: 스킵(오류: {exc})")
        data_version = build_data_version(route, deps.conn, deps.today)
        return {
            "route": route,
            "data_version": data_version,
            "notes": _note({"notes": notes}, f"router: route={route} data_version={data_version}"),
        }

    # ---- 3b. pipeline_gen (route=="pipeline"): SQL 대신 프리미티브 JSON 파이프라인 생성 ----
    def _pipeline_gen(state: GraphState) -> dict:
        import json

        q = state["question"]
        new_attempt = state.get("attempt_count", 0) + 1
        diag = state.get("diagnosis") or {}
        diag_feedback = ""
        if diag.get("fix_hint"):
            diag_feedback = (
                f"\n\n[진단 피드백] 직전 파이프라인에 문제가 있었습니다."
                f"\n원인: {diag.get('explanation', '')}\n수정 지시: {diag.get('fix_hint')}"
                f"\n이를 반드시 반영해 다시 작성하세요."
            )
        today = (deps.today or date.today()).isoformat()
        if deps.llm.available:
            prompt = prompts.PIPELINE_USER.format(schema=deps.schema, today=today, question=q) + diag_feedback
            res = deps.llm.complete(prompt, system=prompts.PIPELINE_SYSTEM, role="sql", max_tokens=700)
            if res.ok and res.text.strip():
                data = extract_json(res.text)
                steps = data.get("pipeline") if isinstance(data, dict) else None
                if isinstance(steps, list) and steps:
                    return {"pipeline": steps, "sql": json.dumps(data, ensure_ascii=False),
                            "sql_source": "pipeline", "attempt_count": new_attempt,
                            "notes": _note(state, f"sql_gen: 파이프라인 JSON {len(steps)}단계 생성")}
            return {"pipeline": [], "sql": "", "sql_source": "pipeline", "attempt_count": new_attempt,
                    "notes": _note(state, "sql_gen: 파이프라인 생성 실패(LLM 응답 파싱 불가)")}
        return {"pipeline": [], "sql": "", "sql_source": "pipeline", "attempt_count": new_attempt,
                "notes": _note(state, "sql_gen: 파이프라인 생성 불가(LLM 미사용)")}

    # ---- 3. sql_gen (실행에러 self-correction + diagnose 피드백 반영) ----
    def sql_gen_node(state: GraphState) -> dict:
        if state.get("route") == "pipeline":
            return _pipeline_gen(state)
        q = state["question"]
        route = state.get("route", "both")
        new_attempt = state.get("attempt_count", 0) + 1
        # diagnose가 직전에 sql/refine 원인으로 되돌려보낸 경우, 그 수정 지시를 프롬프트에 반영한다.
        diag = state.get("diagnosis") or {}
        diag_feedback = ""
        if diag.get("fix_hint"):
            diag_feedback = (
                f"\n\n[진단 피드백] 직전 SQL의 결과에 문제가 있었습니다."
                f"\n원인: {diag.get('explanation', '')}\n수정 지시: {diag.get('fix_hint')}"
                f"\n이를 반드시 반영해 다시 작성하세요."
            )
        if deps.llm.available:
            last_err = prev_sql = None
            for attempt in range(3):
                prompt = prompts.SQL_USER.format(schema=deps.schema, route=route, question=q) + diag_feedback
                if last_err:
                    prompt += (
                        f"\n\n[직전 시도 실패] 아래 SQL이 SQLite에서 오류가 났습니다. "
                        f"원인을 고쳐 올바른 SQLite SELECT로 다시 작성하세요."
                        f"\n오류: {last_err}\n실패 SQL: {prev_sql}"
                    )
                res = deps.llm.complete(prompt, system=prompts.SQL_SYSTEM, role="sql", max_tokens=400)
                if not (res.ok and res.text.strip()):
                    break
                sql = _sanitize_sql(extract_sql(res.text))
                if not sql:
                    break
                check = run_select(deps.conn, sql)  # 실행 검증
                if check["ok"]:
                    tag = "LLM 생성" + (f" (재시도 {attempt}회 후 성공)" if attempt else "")
                    if diag.get("fix_hint"):
                        tag += " [진단 피드백 반영]"
                    return {"sql": sql, "sql_source": "generated", "attempt_count": new_attempt,
                            "notes": _note(state, f"sql_gen: {tag}")}
                last_err, prev_sql = check["error"], sql
            sql = heuristic_sql(q, route)
            return {"sql": sql, "sql_source": "fallback", "attempt_count": new_attempt,
                    "notes": _note(state, f"sql_gen: 휴리스틱 폴백(LLM 3회 실패: {last_err})")}
        sql = heuristic_sql(q, route)
        return {"sql": sql, "sql_source": "fallback", "attempt_count": new_attempt,
                "notes": _note(state, "sql_gen: 휴리스틱 폴백(LLM 미사용)")}

    # ---- 4b. pipeline 실행 (route=="pipeline"): 프리미티브 조립 JSON을 결정론적 실행 ----
    def _pipeline_execute(state: GraphState) -> dict:
        from src.backtest import auditor
        from src.backtest.pipeline_exec import run_pipeline

        steps = state.get("pipeline") or []
        if not steps:
            return {"rows": [], "columns": [], "row_count": 0, "error": "빈 파이프라인",
                    "result_hit": False, "notes": _note(state, "execute: 빈 파이프라인")}

        # 백테스트(run_backtest op)가 있는 파이프라인에만 감사 레이어가 자동 개입한다(AUD-7).
        has_backtest = any(s.get("op") == "run_backtest" for s in steps)
        # search_strategy(역백테스트)는 op=="run_backtest"가 아니라 위 조건에 안 걸리므로
        # 별도로 감지해 스누핑 소프트경고만 1회 첨부한다(architect 검토 MAJOR, auditor.
        # audit_search_strategy_result 참고 — 후보별 전체 감사는 LLM 호출이 최대 80회로
        # 치솟아 비용이 크므로 채택하지 않음).
        has_search_strategy = any(s.get("op") == "search_strategy" for s in steps)

        def _hard_block(msg: str) -> dict:
            return {"rows": [], "columns": [], "row_count": 0, "error": msg,
                    "result_hit": False, "notes": _note(state, f"execute: {msg}")}

        # 사전검사(AUD-3/AUD-7): run_backtest 실행 '직전' 공매도(음수 비중) 하드차단
        if has_backtest:
            try:
                pre = auditor.pre_audit(steps, deps.conn, run_pipeline)
            except Exception:  # noqa: BLE001 — 감사 자체 오류는 실행을 막지 않는다(보조 레이어)
                pre = None
            if pre and pre.get("blocked"):
                return _hard_block(auditor.format_hard_block([pre]))

        try:
            result = run_pipeline(steps, conn=deps.conn)
        except Exception as exc:  # noqa: BLE001 — 실패는 diagnose_node가 원인 분석/재시도/HITL로 처리
            return {"rows": [], "columns": [], "row_count": 0,
                    "error": f"파이프라인 실행 실패: {exc}", "result_hit": False,
                    "notes": _note(state, f"execute: 파이프라인 실패: {exc}")}
        if isinstance(result, dict) and "holdings" in result:
            code_to_name = {r["stock_code"]: r["name"] for r in deps.conn.execute("SELECT stock_code,name FROM company")}
            result = _resolve_holdings_names(result, code_to_name)

        # 사후검사(AUD-1/2/6/AUD-7): run_backtest 실행 '직후' 생존/미래참조 하드차단 + 소프트경고 4종
        warnings: list[dict] = []
        llm_fn = (lambda p: (deps.llm.complete(p, role="judge").text or "")) if deps.llm.available else None
        question = state.get("raw_question") or state.get("question", "")
        if has_backtest and isinstance(result, dict):
            # run_backtest 단계의 market(기본 KR)을 감사에 전달 — US면 생존편향이 '검증불가'로 첨부된다.
            market = next(
                ((s.get("params") or {}).get("market", "KR") for s in steps if s.get("op") == "run_backtest"),
                "KR",
            )
            try:
                audit = auditor.post_audit(result, deps.conn, question, llm_fn, market=market)
            except Exception:  # noqa: BLE001 — 감사 자체 오류는 정상 결과를 버리지 않는다
                audit = {"blocked": False, "hard": [], "soft": []}
            if audit.get("blocked"):
                return _hard_block(auditor.format_hard_block(audit.get("hard", [])))
            warnings = [v for v in audit.get("soft", []) if v.get("triggered")]
        elif has_search_strategy and isinstance(result, list):
            try:
                soft = auditor.audit_search_strategy_result(result, question, llm_fn)
            except Exception:  # noqa: BLE001 — 감사 자체 오류는 정상 결과를 버리지 않는다
                soft = []
            warnings = [v for v in soft if v.get("triggered")]

        rows, columns = _pipeline_result_rows(result)
        out = {"rows": rows, "columns": columns, "row_count": len(rows), "error": None,
               "result_hit": False,
               "notes": _note(state, f"execute: 파이프라인 {len(steps)}단계 실행 성공(rows={len(rows)})")}
        if warnings:
            out["audit_warnings"] = warnings
            out["notes"] = _note({"notes": out["notes"]}, f"execute: 감지된 위험 {len(warnings)}건 첨부")
        chart_path = _maybe_save_chart(result)
        if chart_path:
            out["chart_path"] = chart_path
            out["notes"] = _note({"notes": out["notes"]}, f"execute: 차트 저장 {chart_path}")
        return out

    def _maybe_save_chart(result) -> str | None:
        """run_backtest 프리미티브 결과(navs 시계열 포함)면 NAV PNG를 저장한다(US-12)."""
        if not (isinstance(result, dict) and "navs" in result and "dates" in result):
            return None
        import os

        from src.backtest.chart import save_nav_chart

        os.makedirs(deps.chart_dir, exist_ok=True)
        filename = f"backtest_{now_iso().replace(':', '-')}.png"
        path = os.path.join(deps.chart_dir, filename)
        return save_nav_chart(result["dates"], result["navs"], path, benchmark=result.get("benchmark"))

    # ---- 4. execute (캐시 도출 없음: 항상 새로 실행) ----
    def execute_node(state: GraphState) -> dict:
        if state.get("route") == "pipeline":
            return _pipeline_execute(state)
        sql = state["sql"]
        # 결과 캐시 조회(도출)를 영구 제거 → 결과는 늘 SQLite에서 새로 실행한다.
        r = run_select(deps.conn, sql)
        return {
            "rows": r["rows"],
            "columns": r["columns"],
            "row_count": r["row_count"],
            "error": r["error"],
            "result_hit": False,
            "notes": _note(
                state,
                f"execute: 실행 {'성공' if r['ok'] else '실패: '+str(r['error'])} "
                f"(rows={r['row_count']})",
            ),
        }

    # ---- 5. record (execute 뒤에 실행) ----
    def record_node(state: GraphState) -> dict:
        """성공한 질의를 기록 로그(wiki 테이블)에 저장한다.

        캐시가 아니라 '나중에 열람·평가'하기 위한 로그이므로 WIKI_ENABLED와 무관하게
        항상 기록한다. 단, 실행 실패/빈결과는 기록 가치가 낮아 제외한다.
        question_embedding은 유사도를 더 이상 쓰지 않으므로 저장하지 않는다(NULL).
        """
        if state.get("error") or state.get("row_count", 0) == 0:
            return {"notes": _note(state, "record: 실행 실패/빈결과는 기록 안 함")}
        model = deps.llm.model_for("sql") if deps.llm.available else None
        rid = deps.store.save_record(
            question=state["question"],
            raw_question=state.get("raw_question", state["question"]),
            sql=state["sql"],
            route=state.get("route", ""),
            model=model,
            data_version=state.get("data_version", ""),
            rows=state.get("rows", []),
        )
        return {"wiki_id": rid, "notes": _note(state, f"record: id={rid} model={model} 기록 저장")}

    # ---- 5.5 diagnose (execute 뒤): 결과가 이상하면 원인을 가린다 ----
    def diagnose_node(state: GraphState) -> dict:
        from .diagnose import extract_expected_count, run_diagnosis

        exp = extract_expected_count(state.get("question", ""), state.get("sql", ""))
        diag = run_diagnosis(deps, {**state, "expected_count": exp})
        cause = diag.get("cause")
        msg = (
            f"diagnose: status={diag.get('status')} cause={cause} "
            f"fixable={diag.get('fixable')} - {(diag.get('explanation') or '')[:60]}"
        )
        return {"expected_count": exp, "diagnosis": diag, "notes": _note(state, msg)}

    # ---- HITL: 자동으로 못 고치는 경우(데이터 문제/3회 초과) 사람 검토로 ----
    def human_review_node(state: GraphState) -> dict:
        d = state.get("diagnosis", {})
        return {
            "needs_human": True,
            "notes": _note(
                state,
                f"human_review: 사람 검토 필요 (cause={d.get('cause')}, "
                f"시도 {state.get('attempt_count', 0)}회) - {d.get('explanation', '')}",
            ),
        }

    # ---- 6. eval ----
    def eval_node(state: GraphState) -> dict:
        from src.eval.evaluator import evaluate_state

        ev = evaluate_state(state, deps)
        return {"evaluation": ev, "notes": _note(state, f"eval: {ev.get('summary','')}")}

    return {
        "refine_node": refine_node,
        "router_node": router_node,
        "sql_gen_node": sql_gen_node,
        "execute_node": execute_node,
        "diagnose_node": diagnose_node,
        "record_node": record_node,
        "human_review_node": human_review_node,
        "eval_node": eval_node,
    }
