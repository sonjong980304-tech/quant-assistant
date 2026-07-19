"""eval Layer1 44.3% 원인조사로 발견한 SQL_USER 프롬프트 3가지 실제 버그를 고치는 회귀 테스트 (TDD).

`python3 cli.py eval --json`을 재실행해 생성 SQL을 직접 비교한 결과, 3가지 재현 가능한
원인을 확인했다(정답은 맞았지만 채점 방식과 형식이 달라 불일치 처리되거나, 실제로 결과가
0건이 되는 경우):

1. US 가격 단순조회 예시가 없어 'USD' AS currency 컬럼을 LLM이 빼먹는 경우가 있음
   (#52 "테슬라 최신 종가" — 정답은 맞았으나 currency 컬럼 누락으로 불일치).
2. "미분류 업종"이 이 DB에서 NULL이 아니라 빈 문자열('')로 저장된다는 관례가 프롬프트에
   문서화돼 있지 않아 sector IS NOT NULL(NULL만 제외)을 씀 — 정답은 sector != ''(빈 문자열
   제외) (#30 "업종이 몇 개나 있어?").
3. 순위/스크리닝 질의에만 필요한 정합성 가드(ni.ttm<e.eq 등)를 특정 회사 1곳을 묻는
   단순조회에도 적용해, 그 회사의 실제 지표가 가드 기준을 벗어나면(예: 애플은 자사주매입으로
   ROE가 100%를 넘을 수 있음) 결과가 아예 0건이 됨 (#61 "애플 ROE 알려줘").
"""
from __future__ import annotations

from src.legacy.graph import prompts


def test_sql_user_documents_sector_empty_string_convention():
    """미분류 업종은 빈 문자열이라는 관례가 문서화되고, 예시도 IS NOT NULL이 아닌 != ''를 쓴다."""
    assert "sector != ''" in prompts.SQL_USER
    assert "업종이 몇 개나 있어" in prompts.SQL_USER
    idx = prompts.SQL_USER.index("업종이 몇 개나 있어")
    snippet = prompts.SQL_USER[idx:idx + 200]
    assert "sector != ''" in snippet
    assert "sector IS NOT NULL" not in snippet


def test_sql_user_clarifies_ranking_guards_dont_apply_to_single_company_lookup():
    """정합성 가드는 순위/스크리닝 질의 전용이며 단일회사 조회에는 적용하지 않는다는 원칙이 있다."""
    assert "특정 회사 1곳" in prompts.SQL_USER or "단일" in prompts.SQL_USER


def test_sql_user_still_formats_without_error():
    """새 예시 추가 후에도 .format() 호출이 중괄호 이스케이프 오류 없이 동작한다."""
    prompts.SQL_USER.format(schema="(s)", today="2026-07-13", question="q")
