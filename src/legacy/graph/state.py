"""LangGraph 파이프라인 공유 상태.

주의: LangGraph가 get_type_hints로 런타임 평가하므로 Python 3.9 호환을 위해
`X | None` 대신 Optional[...]을 사용한다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    # 입력
    raw_question: str          # 원본 질문
    question: str              # refine 후 정제 질문

    # 라우팅 / 버전
    route: str                 # 'financial' | 'price' | 'both' | 'pipeline'
    data_version: str          # 데이터 버전 키(기록 스냅샷용)

    # SQL
    sql: str
    sql_source: str            # 'generated' | 'fallback' | 'pipeline'

    # 파이프라인(route=='pipeline'): 프리미티브 조립 JSON (LLM 생성, 실행기가 소비)
    pipeline: List[Dict[str, Any]]

    # 기록(record)
    wiki_id: Optional[int]     # 저장된 기록 로그 행 id (wiki 테이블)
    result_hit: bool           # 결과 캐시 도출 폐기 → 항상 False(하위호환 유지)

    # 실행 결과
    rows: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    error: Optional[str]
    audit_warnings: List[Dict[str, Any]]  # 백테스트 감사 소프트경고(감지된 위험) — cli가 결과 아래 첨부

    # 평가
    gold_sql: str              # 정답 SQL (평가 모드에서만)
    do_eval: bool              # eval 노드 실행 여부
    evaluation: Dict[str, Any]

    # 진단 (diagnose_node)
    attempt_count: int         # sql_gen 생성 횟수 (그래프 루프 재시도 한도용)
    expected_count: Optional[int]  # 질문/SQL이 기대한 결과 개수
    diagnosis: Dict[str, Any]  # {status, cause, fixable, explanation, fix_hint, evidence}
    needs_human: bool          # 3회 초과/데이터문제 → 사람 검토(HITL) 필요

    # 로그
    notes: List[str]
