"""FastAPI 웹 서버.

엔드포인트
  GET    /api/models       : 선택 가능한 LLM 목록 + 가용성
  POST   /api/query        : 자연어 질의 → 계층형 총괄 그래프 실행(종합결론 + 도메인별 원본결과).
                             uncertain=True 로 끝나도 legacy 로 자동 폴백하지 않고 그대로 반환한다.
  GET    /api/query/stream : 계층형 총괄 그래프 스트리밍(SSE, text/event-stream)
                             — 노드 진행 이벤트({"step","summary"})를 실시간 방출(HA-12)
  GET    /api/wiki        : 기록 목록 (?tag=)   ← 질의 기록 로그
  GET    /api/wiki/{id}   : 기록 상세
  PUT    /api/wiki/{id}   : SQL 수정 + 검증/평가 표시 (+태그)
  DELETE /api/wiki/{id}   : 삭제
  GET    /api/stats       : 기록 통계
  GET    /api/eval        : 정답셋 평가 (?limit=, legacy 파이프라인 기준)
  GET    /api/macro        : 거시지표(환율/해외·국내 지수/파마프렌치 5팩터)
  GET    /api/macro/signal : 거시지표 파생 신호
  GET    /api/macro/history: 거시지표 히스토리
  GET    /api/metric-defs  : 백테스트 UI용 지표 정의 목록
  GET    /api/sectors      : 백테스트 UI용 업종(KRX 업종분류) 목록
  POST   /api/backtest     : 백테스트 실행
  GET    /api/backtest-runs: 저장된 백테스트 이력
  GET    /                : 프론트(질의/기록 단일 페이지)
  GET    /macro            : 매크로 전용 페이지

참고: /api/wiki 경로는 하위호환을 위해 유지하지만, 의미는 '질의 기록 로그'다.
      신규 계층형 질의 경로(POST /api/query)는 이 기록에 자동 저장하지 않는다.

실행: uvicorn web.app:app --reload
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.graph import run_hierarchical, run_streaming
from src.config import CONFIG
from src.db import connect, connect_readonly
from src.ingest.exchange_rate import get_usdkrw_rate
from src.wiki.store import WikiStore

app = FastAPI(title="Quant Assistant")

STATIC_DIR = Path(__file__).parent / "static"

# sqlite conn은 스레드 귀속이므로 요청마다 새 읽기전용 연결을 연다(계층형 그래프가 소비).


# ---------------------------------------------------------------------------
# 스키마
# ---------------------------------------------------------------------------
class QueryReq(BaseModel):
    question: str
    model: Optional[str] = None


# 선택 가능한 LLM 후보 (OpenAI + 로컬 Ollama)
LLM_MODELS = [
    {"id": "gpt-5.4-mini", "label": "GPT-5.4-mini (OpenAI)", "provider": "openai"},
    {"id": "gpt-5.5", "label": "GPT-5.5 (OpenAI)", "provider": "openai"},
    {"id": "exaone:latest", "label": "EXAONE (로컬)", "provider": "ollama"},
    {"id": "qwen2.5-coder:7b-instruct-q4_K_M", "label": "Qwen2.5-coder (로컬)", "provider": "ollama"},
]


class WikiUpdate(BaseModel):
    sql: Optional[str] = None
    verified: Optional[bool] = None
    tags: Optional[str] = None


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
# /api/models 캐시 — 매 요청마다 각 모델의 가용성을 네트워크로 확인하면 짧은 간격의
# 반복 요청(프론트 폴링 등)에서 낭비가 크다. 60초 동안은 캐시된 값을 그대로 돌려준다.
_MODELS_CACHE: dict = {"data": None, "fetched_at": 0.0}
_MODELS_CACHE_TTL_SECONDS = 60.0


@app.get("/api/models")
def api_models():
    """선택 가능한 LLM 후보 + 각 가용성(60초 캐싱)."""
    now = time.time()
    if _MODELS_CACHE["data"] is not None and (now - _MODELS_CACHE["fetched_at"]) < _MODELS_CACHE_TTL_SECONDS:
        return _MODELS_CACHE["data"]

    from src.llm import LLMClient

    out = []
    for m in LLM_MODELS:
        try:
            available = LLMClient(model=m["id"]).available
        except Exception:
            available = False
        out.append({**m, "available": available})

    _MODELS_CACHE["data"] = out
    _MODELS_CACHE["fetched_at"] = now
    return out


@app.post("/api/query")
def api_query(req: QueryReq):
    """자연어 질의를 계층형 총괄 그래프(run_hierarchical)로 실행한다.

    총괄→도메인→검증→(최대 3회)재시도를 수행하고, 종합결론(conclusion)과 라우팅된
    도메인별 원본 결과(domain_results, 가공 없음)를 함께 담은 dict를 그대로 돌려준다
    (AC4). 3회 재시도 후에도 uncertain=True 로 끝나면 그 사실 그대로 반환한다(프론트가
    "확신 못함" 경고로 렌더링) — legacy 6단계 Pipeline 으로 자동 폴백하지 않는다.

    (과거 HA-16에서 legacy 자동 폴백 안전망을 붙였으나, 신규 구조가 스스로 정확히
    답하지 못하는 원인을 legacy 우회로 가리는 대신 근본 원인을 고치는 쪽으로 방향을
    바꿔 폴백 로직을 제거했다. legacy 파이프라인 자체는 src/legacy/ 에 참고용으로
    남아 있으나 이 라이브 경로에서는 더 이상 호출하지 않는다.)

    conn 은 요청마다 새 읽기전용 연결(connect_readonly) — 도메인/데이터 에이전트가 LLM 생성
    SQL 을 실행하므로 읽기전용 연결을 요구한다. llm_fn 은 HA-12 가 만든 _build_llm_fn 어댑터를
    재사용한다(가용하지 않으면 None → 결정론적 휴리스틱 폴백).
    """
    if not req.question.strip():
        raise HTTPException(400, "질문이 비어 있습니다.")
    conn = connect_readonly()
    try:
        llm_fn = _build_llm_fn(req.model)
        result = run_hierarchical(req.question, conn, llm_fn=llm_fn)
    except Exception as exc:  # noqa: BLE001 — 원인을 감추지 않고 그대로 500으로 노출
        raise HTTPException(500, f"계층형 에이전트 실행 실패: {type(exc).__name__}: {exc}") from exc
    finally:
        conn.close()
    return {**result, "answered_by": "hierarchical"}


# ---------------------------------------------------------------------------
# 계층형 총괄 그래프 스트리밍 (HA-12) — SSE(text/event-stream)
# ---------------------------------------------------------------------------
# src/agents/graph.py 의 run_streaming(question, conn, llm_fn=..., steps=...) 이
# 노드 완료마다 방출하는 진행 이벤트({"step","summary"})를 그대로 SSE 프레임으로 흘린다.
# 이벤트 스키마는 이미 상세(SQL 전문/원본 rows/결론 본문)를 제외하도록 설계돼 있으므로
# (AC15) 여기서 추가 가공 없이 순서대로 전달만 한다. 프론트(index.html)는 EventSource 로
# 구독해 들여쓰기 트리로 실시간 렌더한다.
#
# 동기 POST /api/query 는 HA-13 에서 이 run_hierarchical 기반으로 전환됐다(위 api_query 참고).
def _sse_message(event: dict) -> str:
    """진행 이벤트({"step","summary"}) 한 건을 SSE data 프레임으로 직렬화한다.

    ensure_ascii=False 로 한글이 \\uXXXX 로 깨지지 않게 한다. SSE 프레임 구분은 빈 줄(\\n\\n).
    """
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _build_llm_fn(model: Optional[str]):
    """LLMClient 를 도메인 에이전트 규약(Callable[[str], str])으로 감싼다.

    가용하지 않으면(키/데몬 없음) None 을 반환해 총괄·도메인 로직이 결정론적 휴리스틱으로
    폴백하게 한다(src/legacy/graph/nodes.py 의 llm_fn 어댑팅 관례와 동일). 총괄→도메인이 쓰는
    주 작업이 Text-to-SQL 이라 role="sql" 로 모델을 고른다.
    """
    from src.llm import LLMClient

    client = LLMClient(model=model)
    if not client.available:
        return None
    return lambda prompt: (client.complete(prompt, role="sql").text or "")


def _query_event_stream(question: str, model: Optional[str]):
    """run_streaming 의 진행 이벤트를 SSE 프레임으로 순서대로 방출하는 제너레이터.

    conn 은 요청마다 새 읽기전용 연결(connect_readonly) — 도메인/데이터 에이전트가
    LLM 생성 SQL 을 실행하므로 읽기전용 연결을 요구한다(src/agents/exec_runtime.py).
    스트림이 정상 종료되면 `event: done` 마커를 보내 프론트가 EventSource 를 닫게 하고
    (미종료 시 EventSource 가 자동 재연결→재실행하는 것을 막는다), 도중 실패는 `event: fail`
    로 알린다(api_macro 의 "실패 격리, 응답은 유지" 관례와 동일 철학).
    """
    conn = None
    try:
        conn = connect_readonly()
        llm_fn = _build_llm_fn(model)
        for event in run_streaming(question, conn, llm_fn=llm_fn):
            yield _sse_message(event)
        yield "event: done\ndata: {}\n\n"
    except Exception as exc:  # noqa: BLE001 — 스트림 도중 실패를 클라이언트에 명시적으로 알림
        yield f"event: fail\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"
    finally:
        if conn is not None:
            conn.close()


@app.get("/api/query/stream")
def api_query_stream(question: str, model: Optional[str] = None):
    """계층형 총괄 그래프를 스트리밍 실행해 노드 진행 이벤트를 SSE 로 방출한다.

    요청:  GET /api/query/stream?question=<질문>&model=<선택 LLM id>
    응답:  text/event-stream — 이벤트마다 `data: {"step","summary"}\\n\\n`,
           정상 종료 `event: done`, 실패 `event: fail`.
    """
    if not question.strip():
        raise HTTPException(400, "질문이 비어 있습니다.")
    return StreamingResponse(
        _query_event_stream(question, model),
        media_type="text/event-stream",
        # 프록시/브라우저 버퍼링으로 실시간성이 깨지지 않게 캐시·버퍼링을 끈다.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/wiki")
def api_wiki_list(tag: Optional[str] = None):
    conn = connect()
    try:
        return WikiStore(conn).list_pages(tag=tag)
    finally:
        conn.close()


@app.get("/api/wiki/{wiki_id}")
def api_wiki_get(wiki_id: int):
    conn = connect()
    try:
        page = WikiStore(conn).get_page(wiki_id)
        if not page:
            raise HTTPException(404, "없는 항목")
        return page
    finally:
        conn.close()


@app.put("/api/wiki/{wiki_id}")
def api_wiki_update(wiki_id: int, body: WikiUpdate):
    conn = connect()
    try:
        store = WikiStore(conn)
        if store.get_page(wiki_id) is None:
            raise HTTPException(404, "없는 항목")
        if body.sql is not None:
            store.update_sql(wiki_id, body.sql, verified=body.verified if body.verified is not None else True)
        elif body.verified is not None:
            store.set_verified(wiki_id, body.verified)
        if body.tags is not None:
            store.set_tags(wiki_id, body.tags)
        return store.get_page(wiki_id)
    finally:
        conn.close()


@app.delete("/api/wiki/{wiki_id}")
def api_wiki_delete(wiki_id: int):
    conn = connect()
    try:
        ok = WikiStore(conn).delete(wiki_id)
        if not ok:
            raise HTTPException(404, "없는 항목")
        return {"deleted": wiki_id}
    finally:
        conn.close()


@app.get("/api/stats")
def api_stats():
    conn = connect()
    try:
        return WikiStore(conn).stats()
    finally:
        conn.close()


@app.get("/api/eval")
def api_eval(limit: Optional[int] = 10):
    from src.eval.runner import run_evaluation

    rep = run_evaluation(limit=limit)
    rep.pop("rows", None)  # 요약만
    return rep


# ---------------------------------------------------------------------------
# 거시지표 티커 (환율 / 해외·국내 지수 / 파마프렌치 5팩터)
# ---------------------------------------------------------------------------
_MACRO_INDICES = [
    {"ticker": "^IXIC", "label": "나스닥종합"},
    {"ticker": "^DJI", "label": "다우존스"},
    {"ticker": "^GSPC", "label": "S&P500"},
    {"ticker": "^SOX", "label": "필라델피아반도체"},
    {"ticker": "^KS11", "label": "코스피"},
    {"ticker": "^KQ11", "label": "코스닥"},
]


def _fetch_index_quote(ticker: str) -> dict:
    import yfinance as yf  # 지연 import — us_prices.py의 기존 패턴과 동일

    hist = yf.Ticker(ticker).history(period="5d")
    closes = hist["Close"].dropna()
    if len(closes) < 2:
        raise ValueError("최근 종가 2개를 확보하지 못함")
    last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
    return {"close": last, "change_pct": round((last - prev) / prev * 100, 2)}


# 코스피/코스닥은 yfinance(^KS11/^KQ11)가 며칠씩 데이터가 안 붙는 지연·결측이 실서버에서
# 확인됐다(2026-07-13 이후 결측 → 등락률이 -8.95%로 왜곡, 실제 당일 등락은 +0.73%).
# 개별종목 가격에 이미 적용 중인 "네이버가 source of truth" 원칙(naver_prices.py)을
# 지수에도 동일하게 적용 — 네이버 실시간 지수 API(delayTime=0, 장마감 즉시 확정치 반영)로
# 대체한다. 나머지 4개(나스닥/다우/S&P500/필라델피아반도체) 미국 지수는 기존 yfinance
# 경로를 그대로 쓴다(같은 지연 문제가 보고된 바 없어 스코프를 좁힘).
_KR_INDEX_NAVER_CODES = {"^KS11": "KOSPI", "^KQ11": "KOSDAQ"}


def _fetch_krx_index_quote(naver_code: str) -> dict:
    """네이버 금융 실시간 지수 API로 코스피/코스닥 종가·등락률을 가져온다.

    응답 필드가 "6,856.83" 같은 쉼표 포함 문자열이라 그대로 float 변환하면 깨진다.
    """
    url = f"https://polling.finance.naver.com/api/realtime/domestic/index/{naver_code}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
    resp.raise_for_status()
    d = resp.json()["datas"][0]
    close = float(d["closePrice"].replace(",", ""))
    change_pct = float(d["fluctuationsRatio"].replace(",", ""))
    return {"close": close, "change_pct": change_pct}


@app.get("/api/macro")
def api_macro():
    """거시지표 티커 응답. 항목별로 실패를 격리해 하나가 죽어도 전체 응답은 유지한다
    (us_prices.py의 "종목별 실패 격리, 계속 진행" 관례와 동일).

    파마프렌치 팩터는 웹 티커에서 뺐다(사용자 요청) — 프롬프트로 직접 물어보면
    fama_french.py의 LLM 의도판단 경로로 여전히 조회 가능하다.
    """
    from datetime import datetime

    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    conn = connect()
    try:
        try:
            usdkrw: dict = {"rate": get_usdkrw_rate(conn)}
        except Exception as exc:  # noqa: BLE001 — 환율 실패해도 나머지는 계속 응답
            usdkrw = {"error": str(exc)}
    finally:
        conn.close()

    indices = []
    for item in _MACRO_INDICES:
        naver_code = _KR_INDEX_NAVER_CODES.get(item["ticker"])
        try:
            quote = _fetch_krx_index_quote(naver_code) if naver_code else _fetch_index_quote(item["ticker"])
            indices.append({**item, **quote})
        except Exception as exc:  # noqa: BLE001 — 지수별 실패 격리
            indices.append({**item, "error": str(exc)})

    return {
        "fetched_at": fetched_at,
        "usdkrw": usdkrw,
        "indices": indices,
        "note": {"quotes": "코스피/코스닥은 네이버 실시간 시세(장중 실시간·장마감 후 확정치), 해외지수는 약 15-20분 지연"},
    }


# ---------------------------------------------------------------------------
# 매크로 신호 (장단기금리차 레짐 기반 GREEN/YELLOW/RED) — /macro 전용 페이지 지원 API.
# 위 /api/macro(환율+지수 티커)와 이름이 겹치지 않도록 하위 경로(/signal, /history)로
# 분리한다. FastAPI는 리터럴 경로를 정확 매칭하므로 /api/macro가 이 둘을 가로채지 않는다.
# 종합신호·밴드는 macro_signal 테이블에 이미 정규화돼 저장되므로 그 한 행을 그대로 노출한다.
# ---------------------------------------------------------------------------
def _signal_payload(row) -> dict:
    """macro_signal 한 행(sqlite Row 또는 None)을 프론트용 구조(지표별 값+밴드)로 변환.
    이력이 없으면(row=None) available=False에 전 필드 None인 동일 구조를 반환한다."""
    if row is None:
        return {
            "available": False, "as_of": None, "overall": None,
            "spread": {"value": None, "regime": None},
            "cnn": {"value": None, "band": None},
            "vix": {"value": None, "band": None},
            "created_at": None,
        }
    return {
        "available": True,
        "as_of": row["as_of"],
        "overall": row["overall"],
        "spread": {"value": row["spread"], "regime": row["spread_regime"]},
        "cnn": {"value": row["cnn_value"], "band": row["cnn_band"]},
        "vix": {"value": row["vix_value"], "band": row["vix_band"]},
        "created_at": row["created_at"],
    }


@app.get("/api/macro/signal")
def api_macro_signal():
    """최신 매크로 신호(overall)와 각 지표 현재값+밴드. 이력이 없으면 available=False로 200."""
    conn = connect()
    try:
        row = conn.execute(
            "SELECT as_of, spread, spread_regime, cnn_value, cnn_band, "
            "vix_value, vix_band, overall, created_at "
            "FROM macro_signal ORDER BY as_of DESC, id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return _signal_payload(row)


@app.get("/api/macro/history")
def api_macro_history(days: int = 30):
    """최근 N일 신호 이력(스파크라인용). 조회 후 과거→최신 순으로 정렬해 반환한다."""
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT as_of, spread, cnn_value, vix_value, overall "
            "FROM macro_signal ORDER BY as_of DESC, id DESC LIMIT ?",
            (days,),
        ).fetchall()
    finally:
        conn.close()
    series = [
        {"as_of": r["as_of"], "overall": r["overall"], "spread": r["spread"],
         "cnn": r["cnn_value"], "vix": r["vix_value"]}
        for r in reversed(rows)  # 최신순 조회 결과를 뒤집어 과거→최신(스파크라인 그리기 순서)
    ]
    return {"days": days, "series": series}


# ---------------------------------------------------------------------------
# 백테스트 (모듈 B)
# ---------------------------------------------------------------------------
class BacktestReq(BaseModel):
    start_year: int = 2024
    end_year: int = 2026
    rebalance: str = "quarterly"
    n: int = 10
    criteria: list = []           # [{"key","direction","weight"}]
    combine: str = "zscore"
    sectors: Optional[list] = None
    markets: Optional[list] = None     # ['KOSPI','KOSDAQ'] 또는 None(전체)
    name: str = "전략"
    fee_rate: Optional[float] = None       # 수수료율(비율). None이면 서버 기본값
    tax_rate: Optional[float] = None       # 매도 거래세율(비율)
    slippage_rate: Optional[float] = None  # 슬리피지율(비율)


@app.get("/api/metric-defs")
def api_metric_defs():
    conn = connect()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT key,label,category,direction,description FROM metric_def")]
    finally:
        conn.close()


@app.get("/api/sectors")
def api_sectors():
    conn = connect()
    try:
        return [r["sector"] for r in conn.execute(
            "SELECT DISTINCT sector FROM company WHERE sector IS NOT NULL AND sector != '' ORDER BY sector")]
    finally:
        conn.close()


@app.post("/api/backtest")
def api_backtest(req: BacktestReq):
    from src.backtest.data_access import build_benchmark_fn, build_callbacks, rebalance_dates
    from src.backtest.engine import run_backtest, save_backtest_run

    if not req.criteria:
        raise HTTPException(400, "지표를 1개 이상 선택하세요.")
    conn = connect()
    try:
        maxd = conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        dates = [d for d in rebalance_dates(req.start_year, req.end_year, req.rebalance)
                 if maxd and d <= maxd]
        if len(dates) < 2:
            raise HTTPException(400, "선택 기간에 데이터가 부족합니다(주가 시계열 범위 확인).")
        mfn, pfn = build_callbacks(conn)
        bench_fn = build_benchmark_fn(dates, mfn, pfn)
        params = {
            "n": req.n, "criteria": req.criteria, "combine": req.combine,
            "sectors": req.sectors, "markets": req.markets, "rebalance": req.rebalance,
            "fee_rate": req.fee_rate if req.fee_rate is not None else CONFIG.fee_rate,
            "tax_rate": req.tax_rate if req.tax_rate is not None else CONFIG.tax_rate,
            "slippage_rate": req.slippage_rate if req.slippage_rate is not None else CONFIG.slippage_rate,
        }
        res = run_backtest(dates, mfn, pfn, params, benchmark_fn=bench_fn)
        save_backtest_run(conn, req.name, params, res["performance"], req.start_year, req.end_year)
        names = {r["stock_code"]: r["name"] for r in conn.execute("SELECT stock_code,name FROM company")}
        holdings = [{"date": h["date"], "names": [names.get(c, c) for c in h["codes"]]}
                    for h in res["holdings"]]
        return {"dates": res["dates"], "navs": res["navs"], "benchmark": res.get("benchmark"),
                "performance": res["performance"], "holdings": holdings}
    finally:
        conn.close()


@app.get("/api/backtest-runs")
def api_backtest_runs():
    import json

    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id,name,start_year,end_year,result_json,created_at "
            "FROM backtest_runs ORDER BY id DESC LIMIT 20").fetchall()
        return [{"id": r["id"], "name": r["name"], "period": f'{r["start_year"]}~{r["end_year"]}',
                 "result": json.loads(r["result_json"]), "created_at": r["created_at"]}
                for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 프론트
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/macro")
def macro_page():
    """매크로 신호 전용 페이지(index.html과 동일 패턴으로 정적 HTML 서빙)."""
    return FileResponse(STATIC_DIR / "macro.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
