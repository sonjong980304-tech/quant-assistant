"""평가 배치 러너.

- run_evaluation : 정답셋 전체에 대해 3층 평가 → 리포트(EX%, Judge 평균, 통과율)

캐시 도출(유사도 재사용) 폐기로 '위키 효율 측정'(히트율/절감 LLM호출)은 의미가 사라져 제거했다.

평가는 항상 원본 DB의 격리된 사본에서 실행된다(_isolated_copy). 평가 경로도 record_node를
거쳐 성공한 질의를 wiki(질의 기록 로그)에 저장하는데, 이 저장이 원본 DB에 직접 일어나면
프로덕션 질의 기록이 평가용 합성 기록과 섞이거나(매 실행마다 오염) 리셋 로직이 실사용자
기록을 삭제하는 사고로 이어진다. 사본에서 실행하면 원본은 항상 그대로 보존된다.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import date

from .goldset import GOLDSET


def _isolated_copy(db_path: str | None) -> str:
    """평가용 임시 DB 사본을 만들어 그 경로를 반환한다.

    원본이 WAL 모드(src.db.connect)라 아직 메인 파일에 체크포인트되지 않은 최근
    커밋이 -wal 파일에만 있을 수 있다. 단순 파일 복사는 이를 놓칠 수 있으므로
    sqlite3의 온라인 백업 API(Connection.backup)로 복사해 WAL 상태에서도 항상
    최신 데이터를 담은 일관된 사본을 만든다.
    """
    from ..config import CONFIG

    src_path = db_path or CONFIG.db_path
    fd, tmp_path = tempfile.mkstemp(suffix=".db", prefix="dart_eval_")
    os.close(fd)
    src = sqlite3.connect(src_path)
    try:
        dst = sqlite3.connect(tmp_path)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return tmp_path


def _cleanup_copy(copy_path: str) -> None:
    for path in (copy_path, f"{copy_path}-wal", f"{copy_path}-shm"):
        if os.path.exists(path):
            os.remove(path)


def run_evaluation(
    db_path: str | None = None,
    today: date | None = None,
    limit: int | None = None,
    offline: bool = False,
) -> dict:
    from ..legacy.pipeline import Pipeline

    items = GOLDSET[:limit] if limit else GOLDSET
    copy_path = _isolated_copy(db_path)
    try:
        p = Pipeline(copy_path, today, offline=offline)
        try:
            rows = []
            ex_hits = ex_total = 0
            order_exact_hits = 0
            gold_errors = 0
            judge_scores: list[int] = []
            l3_pass = 0
            src_count: dict[str, int] = {}

            for item in items:
                s = p.run(item["question"], do_eval=True, gold_sql=item["sql"])
                ev = s.get("evaluation", {})
                l1, l2, l3 = ev.get("layer1", {}), ev.get("layer2", {}), ev.get("layer3", {})

                if l1.get("gold_error"):
                    gold_errors += 1
                if l1.get("applicable"):
                    ex_total += 1
                    if l1.get("match"):
                        ex_hits += 1
                    if l1.get("order_exact_match"):
                        order_exact_hits += 1
                if l2.get("applicable") and l2.get("score") is not None:
                    judge_scores.append(l2["score"])
                if l3.get("passed"):
                    l3_pass += 1
                src = s.get("sql_source", "?")
                src_count[src] = src_count.get(src, 0) + 1

                rows.append({
                    "id": item["id"],
                    "question": item["question"],
                    "sql_source": src,
                    "ex": l1.get("match"),
                    "judge": l2.get("score"),
                    "l3": l3.get("passed"),
                    "row_count": s.get("row_count"),
                    "error": s.get("error"),
                })

            n = len(items)
            report = {
                "n": n,
                "execution_accuracy": {
                    "applicable": ex_total,
                    "match": ex_hits,
                    "ex_pct": round(100 * ex_hits / ex_total, 1) if ex_total else None,
                    "order_exact_pct": round(100 * order_exact_hits / ex_total, 1) if ex_total else None,
                    "gold_errors": gold_errors,
                },
                "judge": {
                    "scored": len(judge_scores),
                    "avg": round(sum(judge_scores) / len(judge_scores), 2) if judge_scores else None,
                },
                "layer3_pass": {"pass": l3_pass, "pct": round(100 * l3_pass / n, 1) if n else None},
                "sql_source": src_count,
                "rows": rows,
            }
            return report
        finally:
            p.close()
    finally:
        _cleanup_copy(copy_path)


def format_eval_report(rep: dict) -> str:
    ex = rep["execution_accuracy"]
    jd = rep["judge"]
    l3 = rep["layer3_pass"]
    lines = [
        "================ 평가 리포트 ================",
        f"문항 수            : {rep['n']}",
        f"Layer1 EX (정확도) : {ex['ex_pct']}%  ({ex['match']}/{ex['applicable']})"
        + (f"   [순서까지 일치 {ex['order_exact_pct']}%]" if ex["order_exact_pct"] is not None else "")
        + (f"   [골드SQL 오류 {ex['gold_errors']}건]" if ex["gold_errors"] else ""),
        f"Layer2 Judge 평균  : {jd['avg']}  (채점 {jd['scored']}건)"
        if jd["avg"] is not None else "Layer2 Judge 평균  : (LLM 미사용)",
        f"Layer3 통과율      : {l3['pct']}%  ({l3['pass']}/{rep['n']})",
        f"SQL 출처 분포      : {rep['sql_source']}",
        "============================================",
    ]
    return "\n".join(lines)
