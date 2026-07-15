"""cli.py 결과 출력의 통화(원/달러) 자동 표기 (TDD, C-5 AC4).

.omc/specs/brainstorming-us-nl-sql-integration.md 참고. SQL_USER 프롬프트가 US 종목
결과에 'currency'='USD' 컬럼을 붙이도록 지시했으므로(prompts.py), cli.py 출력 시
그 컬럼을 보고 금액성 컬럼에는 "$"를 붙이고 비율성 컬럼(per/roe 등)에는 붙이지 않는다.
KR 결과(currency 컬럼 없음)는 기존 동작(원화 억/조 변환) 그대로 유지한다.
"""
from __future__ import annotations

from cli import _fmt, print_table


def test_fmt_without_currency_keeps_existing_krw_behavior():
    assert _fmt(1_500_000_000_000.0) == "1.50조"
    assert _fmt(None) == "NULL"


def test_fmt_with_usd_currency_adds_dollar_sign():
    assert _fmt(1234.5, currency="USD") == "$1,234.50"


def test_fmt_with_usd_currency_and_none_value_still_null():
    assert _fmt(None, currency="USD") == "NULL"


def test_print_table_adds_dollar_sign_to_amount_column_for_usd_row(capsys):
    print_table(["name", "market_cap", "currency"], [{"name": "Apple", "market_cap": 3000000000.0, "currency": "USD"}])
    out = capsys.readouterr().out
    assert "$3,000,000,000.00" in out


def test_print_table_does_not_add_dollar_sign_to_ratio_column(capsys):
    print_table(["name", "per", "currency"], [{"name": "Apple", "per": 28.5, "currency": "USD"}])
    out = capsys.readouterr().out
    assert "$28.5" not in out
    assert "28.5" in out


def test_print_table_hides_currency_column_itself(capsys):
    print_table(["name", "market_cap", "currency"], [{"name": "Apple", "market_cap": 100.0, "currency": "USD"}])
    out = capsys.readouterr().out
    assert "currency" not in out


def test_print_table_krw_rows_unaffected_when_no_currency_column(capsys):
    print_table(["name", "market_cap"], [{"name": "삼성전자", "market_cap": 1_500_000_000_000.0}])
    out = capsys.readouterr().out
    assert "1.50조" in out
    assert "$" not in out
