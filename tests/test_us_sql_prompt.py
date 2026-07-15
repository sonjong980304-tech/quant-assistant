"""SQL_USER 프롬프트에 US 테이블 구조/few-shot이 노출되는지 검증 (TDD, C-5 AC2/AC6).

.omc/specs/brainstorming-us-nl-sql-integration.md 참고.
- AC2: "애플 PER 알려줘"처럼 한글 종목명만으로 US 종목 질의가 SQL로 생성돼야 함
  → 프롬프트가 us_company/us_prices/us_financials 테이블 구조와 종목명 매핑 지침을 담아야 함.
- AC6: "삼성전자와 애플 중 PER 낮은 곳" 같은 비율지표 한미 비교가 UNION 단일 SQL로 생성돼야 함
  → 프롬프트에 UNION 예시가 있어야 함.
"""
from __future__ import annotations

from src.legacy.graph import prompts


def test_sql_user_prompt_describes_us_company_table():
    assert "us_company" in prompts.SQL_USER


def test_sql_user_prompt_describes_us_prices_table():
    assert "us_prices" in prompts.SQL_USER


def test_sql_user_prompt_describes_us_financials_table():
    assert "us_financials" in prompts.SQL_USER


def test_sql_user_prompt_has_union_cross_market_example():
    assert "UNION" in prompts.SQL_USER


def test_sql_user_prompt_still_formats_without_error_after_us_addition():
    formatted = prompts.SQL_USER.format(schema="(s)", question="q")
    assert "us_company" in formatted
    assert "UNION" in formatted
