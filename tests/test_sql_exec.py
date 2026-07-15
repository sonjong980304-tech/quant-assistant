"""src/sql_exec.py 안전 검사 테스트.

_FORBIDDEN 정규식이 REPLACE 키워드를 무조건 금지했는데, sqlite REPLACE() 함수
(문자열 치환)를 쓰는 정당한 SELECT까지 거부했다. 함수 호출 형태("REPLACE(")는
허용하고, "REPLACE INTO ..." 같은 실제 쓰기 문장만 계속 차단해야 한다.
"""
from __future__ import annotations

from src.sql_exec import is_safe_select


def test_is_safe_select_allows_replace_function_call():
    safe, reason = is_safe_select("SELECT REPLACE(name, 'Inc.', '') FROM company")
    assert safe is True
    assert reason == ""


def test_is_safe_select_allows_replace_function_call_with_space_before_paren():
    safe, _ = is_safe_select("SELECT replace (name, 'A', 'B') FROM company")
    assert safe is True


def test_is_safe_select_still_blocks_replace_into_statement():
    safe, reason = is_safe_select("REPLACE INTO company(stock_code) VALUES('X')")
    assert safe is False
    assert reason


def test_is_safe_select_still_blocks_insert_or_replace():
    safe, reason = is_safe_select("INSERT OR REPLACE INTO company(stock_code) VALUES('X')")
    assert safe is False
    assert reason


def test_is_safe_select_still_blocks_other_forbidden_keywords():
    safe, reason = is_safe_select("SELECT * FROM company; DROP TABLE company")
    assert safe is False
    assert reason
