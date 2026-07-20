"""질의 기록 로그 + 사람 편집/검증.

이전에는 '질문 임베딩 유사도로 이전 SQL/결과를 찾아 재사용'하는 2단계 캐시였으나,
그 도출(캐시) 기능을 완전히 폐기했다. 이제 wiki 테이블은 "질의 기록 로그"로만 쓰인다.
- 모든 성공한 질의의 질문/SQL/route/결과 스냅샷을 기록한다(항상 새로 생성·실행).
- 사람이 나중에 열람하고 verified(검증/평가)를 표시할 수 있다.
- 유사도 도출이 없으므로 question_embedding은 저장하지 않는다(NULL).

테이블명 'wiki'는 마이그레이션 위험을 피하기 위해 그대로 두되, 의미는 '기록 로그'다.
"""
from __future__ import annotations

import json
import sqlite3

from ..version import now_iso

# ---------------------------------------------------------------------------
# 자동 태그 규칙 (질문/SQL 키워드 → 태그)
# ---------------------------------------------------------------------------
_TAG_RULES: list[tuple[str, list[str]]] = [
    ("PER", ["per", "주가수익"]),
    ("PBR", ["pbr", "주가순자산"]),
    ("ROE", ["roe", "자기자본이익"]),
    ("부채", ["debt_ratio", "부채"]),
    ("영업이익률", ["operating_margin", "영업이익률", "영업이익"]),
    ("시가총액", ["market_cap", "시가총액", "시총"]),
    ("매출", ["revenue", "매출"]),
    ("순이익", ["net_income", "순이익"]),
]


def auto_tags(question: str, sql: str) -> str:
    text = f"{question} {sql}".lower()
    tags = [tag for tag, kws in _TAG_RULES if any(k in text for k in kws)]
    return ",".join(dict.fromkeys(tags))  # 순서 보존 dedup


class WikiStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ===================== 기록 저장 =====================
    def save_record(
        self,
        question: str,
        raw_question: str,
        sql: str,
        route: str,
        model: str | None = None,
        data_version: str = "",
        rows: list[dict] | None = None,
    ) -> int:
        """성공한 질의를 기록 로그(wiki 테이블)에 1행 저장한다.

        유사도 도출을 폐기했으므로 question_embedding은 저장하지 않는다(NULL).
        결과 스냅샷(result_json, 상위 20행)과 data_version도 함께 기록해 나중에 열람할 수 있게 한다.
        """
        ts = now_iso()
        tags = auto_tags(question, sql)
        snapshot = json.dumps((rows or [])[:20], ensure_ascii=False)
        cur = self.conn.execute(
            """INSERT INTO wiki
                 (question, raw_question, question_embedding, sql, route,
                  data_version, result_json, verified, tags, use_count, model,
                  created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,0,?,?,?)""",
            (question, raw_question, None, sql, route,
             data_version, snapshot, 0, tags, model, ts, ts),
        )
        self.conn.commit()
        return cur.lastrowid

    # ===================== 기록 편집 (사람) =====================
    def list_pages(self, tag: str | None = None) -> list[dict]:
        # 기록 로그이므로 최신순(id DESC)으로 보여준다.
        if tag:
            rows = self.conn.execute(
                "SELECT * FROM wiki WHERE tags LIKE ? ORDER BY id DESC",
                (f"%{tag}%",),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM wiki ORDER BY id DESC"
            ).fetchall()
        return [self._page_dict(r) for r in rows]

    def get_page(self, wiki_id: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM wiki WHERE id=?", (wiki_id,)).fetchone()
        return self._page_dict(r) if r else None

    def update_sql(self, wiki_id: int, new_sql: str, verified: bool = True) -> bool:
        """사람이 SQL을 수정하면 기본적으로 검증됨으로 표시."""
        cur = self.conn.execute(
            "UPDATE wiki SET sql=?, verified=?, updated_at=? WHERE id=?",
            (new_sql, int(verified), now_iso(), wiki_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def set_verified(self, wiki_id: int, verified: bool = True) -> bool:
        cur = self.conn.execute(
            "UPDATE wiki SET verified=?, updated_at=? WHERE id=?",
            (int(verified), now_iso(), wiki_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def set_tags(self, wiki_id: int, tags: str) -> bool:
        cur = self.conn.execute(
            "UPDATE wiki SET tags=?, updated_at=? WHERE id=?",
            (tags, now_iso(), wiki_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete(self, wiki_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM wiki WHERE id=?", (wiki_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # ===================== 통계 =====================
    def stats(self) -> dict:
        """기록 통계. 캐시 전용 지표(재사용·절감 LLM호출·결과캐시)는 의미가 사라져 제거.

        wiki_entries/verified_entries 키는 하위호환을 위해 유지하되 의미는 '기록 수/검증 수'다.
        """
        total = self.conn.execute("SELECT COUNT(*) c FROM wiki").fetchone()["c"]
        verified = self.conn.execute("SELECT COUNT(*) c FROM wiki WHERE verified=1").fetchone()["c"]
        tag_rows = self.conn.execute(
            "SELECT tags FROM wiki WHERE tags IS NOT NULL AND tags != ''"
        ).fetchall()
        tag_count: dict[str, int] = {}
        for r in tag_rows:
            for t in r["tags"].split(","):
                t = t.strip()
                if t:
                    tag_count[t] = tag_count.get(t, 0) + 1
        return {
            "record_entries": total,        # 기록 수
            "verified_entries": verified,   # 검증/평가 표시된 기록 수
            "tag_distribution": dict(sorted(tag_count.items(), key=lambda x: -x[1])),
        }

    @staticmethod
    def _page_dict(r: sqlite3.Row) -> dict:
        # 결과 스냅샷(result_json) 행수를 요약값으로 제공 (열람용).
        snap_rows = 0
        rj = r["result_json"] if "result_json" in r.keys() else None
        if rj:
            try:
                snap_rows = len(json.loads(rj))
            except (ValueError, TypeError):
                snap_rows = 0
        return {
            "id": r["id"],
            "question": r["question"],
            "raw_question": r["raw_question"],
            "sql": r["sql"],
            "route": r["route"],
            "verified": bool(r["verified"]),
            "tags": r["tags"] or "",
            "model": r["model"] if "model" in r.keys() else None,
            "data_version": r["data_version"],
            "result_rows": snap_rows,   # 결과 스냅샷 행수(요약)
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
