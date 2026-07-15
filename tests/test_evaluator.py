"""Layer1(Execution Accuracy) 비교로직 완화 회귀 테스트.

- 정답(gold) SQL 자체가 실행 실패하면 평가 하네스 결함이지 모델 오답이 아니므로
  EX 분모(applicable)에서 빠져야 한다.
- 대부분의 goldset 질문은 "Top-N" 랭킹 질의라 채점 목적은 "올바른 N개 행을
  뽑았는가"이며, 동점 타이브레이크까지 순서가 같아야 하는 건 아니므로 기본
  비교는 순서 무관(멀티셋)이어야 한다.
- 반올림 자릿수는 SQL 생성 프롬프트/goldset의 ROUND(x,2) 관례와 맞춰 2자리로
  통일해야 부동소수 오차로 인한 오탐 불일치가 없다.
"""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from src.eval.evaluator import eval_layer1


def _deps(conn: sqlite3.Connection):
    return SimpleNamespace(conn=conn)


def test_eval_layer1_gold_sql_failure_is_not_applicable():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(name TEXT)")
    conn.execute("INSERT INTO t VALUES ('a')")

    state = {
        "gold_sql": "SELECT nonexistent_col FROM t",
        "rows": [{"name": "a"}],
    }
    result = eval_layer1(state, _deps(conn))

    assert result["applicable"] is False, "gold SQL 실행실패는 EX 분모에서 제외돼야 한다"
    assert result["gold_error"] is True


def test_eval_layer1_match_ignores_row_order():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(name TEXT)")
    conn.executemany("INSERT INTO t VALUES (?)", [("B",), ("A",)])

    state = {
        "gold_sql": "SELECT name FROM t",  # 삽입 순서대로 B, A
        "rows": [{"name": "A"}, {"name": "B"}],  # 같은 집합, 순서만 반대
    }
    result = eval_layer1(state, _deps(conn))

    assert result["match"] is True, "Top-N 랭킹 질의는 행 집합만 맞으면 정답으로 봐야 한다"


def test_eval_layer1_match_tolerant_of_rounding_beyond_2_decimals():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(v REAL)")
    conn.execute("INSERT INTO t VALUES (12.341)")

    state = {
        "gold_sql": "SELECT v FROM t",
        "rows": [{"v": 12.344}],  # 4자리로는 다르지만 2자리로는 둘 다 12.34
    }
    result = eval_layer1(state, _deps(conn))

    assert result["match"] is True, "비교 반올림은 SQL의 ROUND(x,2) 관례에 맞춰 2자리여야 한다"
