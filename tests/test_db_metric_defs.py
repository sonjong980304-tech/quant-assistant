"""seed_metric_defs가 METRIC_DEFS에서 제거된 지표를 실제로 정리하는지 검증.

코드리뷰 발견: seed_metric_defs는 INSERT OR REPLACE(upsert)만 하고 DELETE가 없어서,
이미 시드된 DB에 init_db를 다시 돌려도 METRIC_DEFS에서 뺀 지표(예: dividend_yield,
momentum)가 metric_def 테이블에 그대로 남는다. UI 체크박스는 metric_def 테이블을
읽으므로, 크래시나던 필드를 코드에서 뺐어도 이미 시드된(다른 환경의) DB에서는
재발할 수 있었다. seed_metric_defs가 METRIC_DEFS에 없는 key를 DELETE하도록 고친다.
"""
from __future__ import annotations

import sqlite3

from src.db import METRIC_DEFS, init_db, seed_metric_defs


def test_seed_metric_defs_removes_stale_keys_not_in_metric_defs(tmp_path):
    db = str(tmp_path / "stale.db")
    init_db(db)
    conn = sqlite3.connect(db)
    try:
        # METRIC_DEFS에 없는 옛 지표가 이미 시드돼 있던 상황을 재현(다른 환경의 낡은 DB 흉내).
        conn.execute(
            "INSERT OR REPLACE INTO metric_def(key,label,category,direction,description) "
            "VALUES('dividend_yield','배당수익률','기타','high','주당배당금/주가(%)')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO metric_def(key,label,category,direction,description) "
            "VALUES('momentum','가격모멘텀','기타','high','최근 가격 상승률(%)')"
        )
        conn.commit()

        seed_metric_defs(conn)

        keys = {r[0] for r in conn.execute("SELECT key FROM metric_def")}
    finally:
        conn.close()

    assert keys == {k for k, *_ in METRIC_DEFS}
    assert "dividend_yield" not in keys
    assert "momentum" not in keys


def test_seed_metric_defs_keeps_all_current_keys(tmp_path):
    db = str(tmp_path / "clean.db")
    init_db(db)
    conn = sqlite3.connect(db)
    try:
        keys = {r[0] for r in conn.execute("SELECT key FROM metric_def")}
    finally:
        conn.close()
    assert keys == {k for k, *_ in METRIC_DEFS}
