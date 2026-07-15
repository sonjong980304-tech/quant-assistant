"""SQL 안전 실행 (읽기 전용). execute_node와 평가 Layer3에서 공용."""
from __future__ import annotations

import re
import sqlite3

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


def is_safe_select(sql: str) -> tuple[bool, str]:
    s = (sql or "").strip().rstrip(";")
    if not s:
        return False, "빈 SQL"
    if ";" in s:
        return False, "다중 문장 금지"
    if not re.match(r"^\s*(select|with)\b", s, re.IGNORECASE):
        return False, "SELECT/WITH 문만 허용"
    if _FORBIDDEN.search(s):
        return False, "쓰기/DDL 키워드 금지"
    return True, ""


def ensure_limit(sql: str, default: int = 200) -> str:
    s = sql.strip().rstrip(";")
    if re.search(r"\blimit\b", s, re.IGNORECASE):
        return s
    return f"{s} LIMIT {default}"


def run_select(conn: sqlite3.Connection, sql: str, max_rows: int = 1000) -> dict:
    """안전한 SELECT 실행. 반환: {ok, columns, rows, row_count, error}."""
    safe, reason = is_safe_select(sql)
    if not safe:
        return {"ok": False, "columns": [], "rows": [], "row_count": 0, "error": reason}
    try:
        cur = conn.execute(ensure_limit(sql))
        columns = [d[0] for d in cur.description] if cur.description else []
        fetched = cur.fetchmany(max_rows)
        rows = [dict(zip(columns, r)) for r in fetched]
        return {
            "ok": True,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "error": None,
        }
    except sqlite3.Error as e:
        return {
            "ok": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": f"{type(e).__name__}: {e}",
        }
