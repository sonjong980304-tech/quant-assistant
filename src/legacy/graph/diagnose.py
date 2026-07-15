"""진단 서브에이전트 핵심 로직 (diagnose_node가 호출).

질의 결과가 '이상하면'(에러·빈결과·개수부족·이상치) 원인을 가린다.

설계 원칙: **증거 수집은 코드, 해석·분류는 LLM(gpt-5.5).**
LLM 혼자 "왜 결과가 적냐"를 물으면 데이터를 못 보니 환각한다. 그래서
코드가 먼저 기계적으로 증거(조건별 분해표·이상치 원본값)를 모으고,
LLM은 그 증거를 읽고 원인을 sql/refine/data/none 으로 분류만 한다.

단계
----
1) classify_status : 결과 상태 분류 (코드)        sql_error|empty|short|anomaly|ok
2) collect_evidence: 증거 수집 (코드)             조건별 분해 / 이상치 원본
3) llm_classify    : 원인 분류 (LLM, role=diagnose) {cause, fixable, explanation, fix_hint}
"""
from __future__ import annotations

import re

from src.llm import extract_json
from src.sql_exec import run_select

from . import prompts

# ---------------------------------------------------------------------------
# sanity 임계값: 결과 컬럼이 이 범위를 벗어나면 '이상치'로 본다.
#   - 지표는 보통 LLM이 per/roe/market_cap 같은 깔끔한 별칭으로 낸다.
#   - market_cap은 원 단위(3000조=3e15), 비율 지표는 %/배수.
# ---------------------------------------------------------------------------
SANITY: dict[str, tuple[float, float]] = {
    "per": (0, 500),
    "pbr": (0, 100),
    "psr": (0, 300),
    "pcr": (0, 500),
    "roe": (-200, 200),          # %
    "roa": (-200, 200),
    "operating_margin": (-1000, 100),
    "net_margin": (-1000, 100),
    "debt_ratio": (0, 5000),
    "market_cap": (0, 3e15),     # 3000조 초과 = 단위오류 의심
}


def _sanity_bound(col: str):
    """컬럼명에 맞는 (min,max) 임계. 정확 일치 우선, 없으면 토큰 단위 매칭.

    주의: 단순 부분문자열 매칭은 'oPERating_profit'(영업이익=금액)의 'per'에
    PER 임계(0~500)가 잘못 걸려 57조 같은 정상 금액을 이상치로 오판한다.
    그래서 단일 키(per/roe 등)는 컬럼을 '_'로 쪼갠 토큰과 '정확히' 일치할 때만,
    복합 키(operating_margin 등)는 전체 부분일치일 때만 적용한다.
    금액 지표(operating_profit/net_income/revenue 등)는 어디에도 안 걸려 sanity 제외된다."""
    c = (col or "").strip().lower()
    if c in SANITY:
        return SANITY[c]
    tokens = c.replace("-", "_").split("_")
    for key, bound in SANITY.items():
        if "_" in key:             # 복합 키: 전체가 부분문자열로 들어있을 때만
            if key in c:
                return bound
        elif key in tokens:        # 단일 키: 토큰과 정확히 일치할 때만 ('avg_per'의 'per'는 허용)
            return bound
    return None


def find_anomalies(rows: list[dict], columns: list[str]) -> list[dict]:
    """결과에서 sanity 범위를 벗어난 셀을 찾는다. 반환: [{row, column, value, bound}]."""
    out = []
    for r in rows:
        for col, val in r.items():
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            bound = _sanity_bound(col)
            if bound is None:
                continue
            lo, hi = bound
            if val < lo or val > hi:
                out.append({"row": r, "column": col, "value": val, "bound": [lo, hi]})
    return out


# ---------------------------------------------------------------------------
# 기대 개수 추출 (질문의 'N개/상위 N/top N' 또는 SQL의 LIMIT N)
# ---------------------------------------------------------------------------
_Q_COUNT = re.compile(r"(?:상위|top|최대|최고)?\s*(\d+)\s*(?:개|곳|종목|개사|位|위)\b", re.IGNORECASE)
_LIMIT = re.compile(r"\blimit\s+(\d+)", re.IGNORECASE)


def extract_expected_count(question: str, sql: str) -> int | None:
    """질문/SQL에서 사용자가 기대한 결과 개수. 못 찾으면 None.

    '가장 ~한 회사'(단수)는 1개를 기대하지만 정상 케이스가 많아 굳이 트리거하지 않는다.
    여기서는 명시적 숫자(N개)와 LIMIT N만 본다.
    """
    m = _Q_COUNT.search(question or "")
    if m:
        n = int(m.group(1))
        if 1 <= n <= 1000:
            return n
    m = _LIMIT.search(sql or "")
    if m:
        n = int(m.group(1))
        # LIMIT가 200(기본 폴백)이면 사용자가 명시한 개수로 보기 어렵다 → 무시
        if 1 <= n < 200:
            return n
    return None


# ---------------------------------------------------------------------------
# 결과 상태 분류 (코드)
# ---------------------------------------------------------------------------
def classify_status(state: dict) -> str:
    if state.get("error"):
        return "sql_error"
    rc = state.get("row_count", 0)
    if rc == 0:
        return "empty"
    exp = state.get("expected_count")
    if exp and rc < exp:
        return "short"
    if find_anomalies(state.get("rows", []), state.get("columns", [])):
        return "anomaly"
    return "ok"


# ---------------------------------------------------------------------------
# 조건별 분해 (WHERE의 top-level AND를 누적 적용하며 row_count 측정)
# ---------------------------------------------------------------------------
_CLAUSE_END = re.compile(r"\b(group\s+by|order\s+by|limit|having)\b", re.IGNORECASE)


def _split_where(sql: str):
    """sql을 (prefix, where_body, suffix)로 분해. 최상위(괄호깊이 0) WHERE만 대상.

    prefix = WHERE 앞(SELECT..FROM..), where_body = WHERE 본문,
    suffix = GROUP BY/ORDER BY/LIMIT.. 이후. 최상위 WHERE가 없으면 (None,None,None).
    """
    s = sql.rstrip().rstrip(";")
    low = s.lower()
    depth = 0
    where_pos = -1
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and low.startswith("where", i):
            before = s[i - 1] if i > 0 else " "
            after = s[i + 5] if i + 5 < n else " "
            if not before.isalnum() and not after.isalnum():
                where_pos = i
                break
        i += 1
    if where_pos == -1:
        return None, None, None
    body_start = where_pos + 5
    depth = 0
    end = n
    j = body_start
    while j < n:
        ch = s[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and _CLAUSE_END.match(low, j):
            end = j
            break
        j += 1
    return s[:where_pos], s[body_start:end], s[end:]


def _split_top_and(body: str) -> list[str]:
    """WHERE 본문을 괄호깊이 0의 ' AND '로 분리."""
    parts = []
    depth = 0
    cur = []
    low = body.lower()
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "(":
            depth += 1
            cur.append(ch)
            i += 1
        elif ch == ")":
            depth -= 1
            cur.append(ch)
            i += 1
        elif depth == 0 and low.startswith(" and ", i):
            parts.append("".join(cur))
            cur = []
            i += 5
        else:
            cur.append(ch)
            i += 1
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return [p.strip() for p in parts if p.strip()]


def probe_conditions(conn, sql: str, max_conditions: int = 8) -> list[dict]:
    """WHERE 조건을 하나씩 누적하며 row_count 측정 → 어느 조건에서 급감하는지.

    반환: [{"added": 조건문자열|"(조건없음)", "row_count": n}, ...]
    분해 불가(최상위 WHERE 없음/단일조건)면 빈 리스트.
    """
    prefix, body, suffix = _split_where(sql)
    if not body:
        return []
    conds = _split_top_and(body)
    if len(conds) < 1:
        return []
    if len(conds) > max_conditions:
        conds = conds[:max_conditions]
    steps = []
    # 조건 없음(WHERE 제거)
    base = f"{prefix} {suffix}".strip()
    steps.append({"added": "(조건없음)", "row_count": run_select(conn, base)["row_count"]})
    acc: list[str] = []
    for cond in conds:
        acc.append(cond)
        q = f"{prefix} WHERE {' AND '.join(acc)} {suffix}".strip()
        r = run_select(conn, q)
        steps.append({"added": cond, "row_count": r["row_count"] if r["ok"] else None})
    return steps


# ---------------------------------------------------------------------------
# 증거 수집 (상태별)
# ---------------------------------------------------------------------------
def collect_evidence(conn, state: dict, status: str) -> dict:
    if status == "sql_error":
        return {"type": "error", "message": state.get("error", "")}
    if status in ("empty", "short"):
        if state.get("sql_source") == "pipeline":
            # pipeline 경로는 state["sql"]에 SQL이 아니라 프리미티브 조립 JSON이 들어있어
            # probe_conditions(WHERE절 파싱)가 항상 실패해 증거가 빈 리스트가 된다(Fable5
            # 서브그래프 진단에서 발견). SQL 조건분해 대신 원본 파이프라인 JSON을 그대로
            # 증거로 준다 — LLM이 criteria/params를 보고 원인을 추론할 최소 재료가 된다.
            return {
                "type": "pipeline_probe",
                "expected": state.get("expected_count"),
                "got": state.get("row_count", 0),
                "pipeline": state.get("sql", ""),
            }
        return {
            "type": "probe",
            "expected": state.get("expected_count"),
            "got": state.get("row_count", 0),
            "steps": probe_conditions(conn, state.get("sql", "")),
        }
    if status == "anomaly":
        anomalies = find_anomalies(state.get("rows", []), state.get("columns", []))
        return {"type": "anomaly", "items": anomalies[:5]}
    return {"type": "none"}


# ---------------------------------------------------------------------------
# 원인 분류 (LLM, role=diagnose → gpt-5.5)
# ---------------------------------------------------------------------------
def llm_classify(deps, state: dict, status: str, evidence: dict) -> dict:
    """증거를 LLM에 주고 원인 분류. LLM 미가용 시 규칙 기반 보수적 판정."""
    if not deps.llm.available:
        # LLM 없으면: 에러는 sql, 그 외는 data로 보수적 폴백(사람 확인 권장)
        cause = "sql" if status == "sql_error" else ("none" if status == "ok" else "data")
        return {"cause": cause, "fixable": status == "sql_error",
                "explanation": "LLM 미사용 — 규칙 기반 보수 판정", "fix_hint": ""}

    res = deps.llm.complete(
        prompts.DIAGNOSE_USER.format(
            question=state.get("question", ""),
            sql=state.get("sql", ""),
            status=status,
            row_count=state.get("row_count", 0),
            expected=state.get("expected_count"),
            evidence=evidence,
        ),
        system=prompts.DIAGNOSE_SYSTEM,
        role="diagnose",
        max_tokens=400,
    )
    if not res.ok:
        return {"cause": "data", "fixable": False,
                "explanation": f"진단 LLM 실패: {res.error}", "fix_hint": ""}
    data = extract_json(res.text)
    cause = str(data.get("cause", "")).lower()
    if cause not in ("sql", "refine", "data", "none"):
        cause = "data"  # 모르면 사람에게(보수적)
    return {
        "cause": cause,
        "fixable": bool(data.get("fixable", cause in ("sql", "refine"))),
        "explanation": str(data.get("explanation", "")).strip(),
        "fix_hint": str(data.get("fix_hint", "")).strip(),
    }


def run_diagnosis(deps, state: dict) -> dict:
    """전체 오케스트레이션. diagnosis dict 반환.

    반환: {status, evidence, cause, fixable, explanation, fix_hint}
    status=='ok' 이고 이상 없으면 LLM 호출 없이 통과(cause='none').
    """
    status = classify_status(state)
    if status == "ok":
        return {"status": "ok", "evidence": {"type": "none"},
                "cause": "none", "fixable": False, "explanation": "", "fix_hint": ""}
    evidence = collect_evidence(deps.conn, state, status)
    verdict = llm_classify(deps, state, status, evidence)
    return {"status": status, "evidence": evidence, **verdict}
