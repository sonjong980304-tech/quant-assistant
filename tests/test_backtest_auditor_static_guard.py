"""정적 코드 감사(AUD-4): primitives.py가 원본 테이블을 직접 조회하지 않는지 검사.

생존편향/미래참조편향은 data_access.py의 metrics_at()/effective_quarter_at()/_is_alive()가
disclosed_date<=asof 가드와 delisting 가드로 구조적으로 막는다. 그런데 새 프리미티브를
추가하면서 실수로 financials/company/prices 원본을 metrics_at()/get_cross_section()을 거치지
않고 conn.execute()로 직접 조회하면 그 가드가 무력화된다(스펙 §3.2 ①②의 정적 감사).

이 테스트는 별도 CLII 명령이 아니라 `python3 -m pytest` 실행 시 자동으로 함께 돈다
(신규 명령 신설 금지 — 스펙 §5 제약). grep/ast 기반 결정론적 검사다(LLM 없음).

허용 예외: SELECT MAX(date) FROM prices 처럼 종목 단위 데이터를 뽑지 않는 순수 집계
메타데이터 조회(백테스트 기간 상한 산정용)는 편향과 무관하므로 위반이 아니다.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_PRIMITIVES = _REPO / "src" / "backtest" / "primitives.py"

_SENSITIVE = {"financials", "company", "prices"}
_AGG = re.compile(r"\b(MAX|MIN|COUNT|AVG|SUM)\s*\(", re.IGNORECASE)
_FROM_JOIN = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)


def _static_str(node: ast.AST) -> str | None:
    """.execute()의 첫 인자가 문자열 상수(또는 상수끼리의 + 연결)면 그 값을 반환."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _static_str(node.left), _static_str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _execute_sql_literals(source: str) -> list[str]:
    """소스에서 `<x>.execute("...")` 호출의 SQL 문자열 리터럴을 모두 수집한다."""
    tree = ast.parse(source)
    sqls: list[str] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute" and node.args):
            s = _static_str(node.args[0])
            if s:
                sqls.append(s)
    return sqls


def _select_clause_is_aggregate_only(sql: str) -> bool:
    """SQL 전체가 종목단위 필터 없는 순수 집계 메타데이터 조회인지.

    SELECT~FROM 구간이 집계함수로만 구성돼 있어도, WHERE/HAVING 등 나머지 구간에
    stock_code가 있으면 종목별 집계(예: SELECT MAX(amount) FROM financials
    WHERE stock_code=?)이므로 disclosed_date<=asof 가드 우회다 — 순수 집계가 아니다.
    """
    if "stock_code" in sql.lower():
        return False
    m = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return False
    clause = m.group(1)
    if "*" in clause:
        return False
    # 집계함수 호출을 모두 제거하고 남는 게(공백/콤마 제외) 없으면 순수 집계 메타데이터 조회다.
    residue = _AGG.sub("", clause)
    residue = re.sub(r"\)", "", residue)  # 집계 닫는 괄호 제거
    residue = re.sub(r"[a-zA-Z_][a-zA-Z0-9_.]*", "", residue)  # 집계 인자(컬럼명) 제거
    residue = re.sub(r"[\s,()]", "", residue)
    return residue == "" and bool(_AGG.search(clause))


def _sensitive_direct_reads(sql: str) -> list[str]:
    """SQL이 민감 원본 테이블을 종목단위로 직접 조회하면 그 테이블명 목록을 반환(위반)."""
    tables = {t.lower() for t in _FROM_JOIN.findall(sql)}
    hit = tables & _SENSITIVE
    if not hit:
        return []
    if _select_clause_is_aggregate_only(sql):
        return []  # 순수 집계 메타데이터(예: SELECT MAX(date) FROM prices)는 허용
    return sorted(hit)


# --------------------------------------------------------------------------
# 본 검사: 현재 primitives.py는 위반이 없어야 한다
# --------------------------------------------------------------------------
def test_primitives_does_not_directly_read_source_tables():
    source = _PRIMITIVES.read_text(encoding="utf-8")
    violations = []
    for sql in _execute_sql_literals(source):
        hit = _sensitive_direct_reads(sql)
        if hit:
            violations.append((hit, " ".join(sql.split())[:80]))
    assert violations == [], f"primitives.py가 원본 테이블을 직접 조회함(편향 가드 우회): {violations}"


# --------------------------------------------------------------------------
# 가드 자체가 실제로 위반을 잡는지(vacuous 아님) — 양성/음성 통제
# --------------------------------------------------------------------------
def test_guard_flags_direct_financials_read():
    bad = "conn.execute('SELECT amount FROM financials WHERE stock_code=?')"
    assert _sensitive_direct_reads(_execute_sql_literals(bad)[0]) == ["financials"]


def test_guard_flags_direct_price_row_read():
    bad = "conn.execute('SELECT close FROM prices WHERE stock_code=?')"
    assert _sensitive_direct_reads(_execute_sql_literals(bad)[0]) == ["prices"]


def test_guard_allows_aggregate_metadata_query():
    ok = "conn.execute('SELECT MAX(date) FROM prices')"
    assert _sensitive_direct_reads(_execute_sql_literals(ok)[0]) == []


def test_guard_flags_aggregate_with_stock_code_where_clause():
    """SELECT절만 순수집계여도 WHERE에 stock_code가 있으면 종목단위 우회이므로 위반이다.

    architect 검수 권고: _select_clause_is_aggregate_only가 SELECT~FROM 구간만 보고
    WHERE절을 안 봐서, `SELECT MAX(amount) FROM financials WHERE stock_code=?` 같은
    종목별 집계 조회가 "순수 집계"로 오인돼 disclosed_date<=asof 가드를 우회할 수 있었다.
    """
    bad = "conn.execute('SELECT MAX(amount) FROM financials WHERE stock_code=?')"
    assert _sensitive_direct_reads(_execute_sql_literals(bad)[0]) == ["financials"]
