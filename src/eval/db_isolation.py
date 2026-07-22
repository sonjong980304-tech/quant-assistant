"""평가/factcheck용 원본 DB 격리 사본 생성·정리 유틸리티.

factcheck/backtest.py가 쓴다("원본 DB를 건드리지 않고 사본에서만 평가를 실행"해야
하는 요구). 원래 legacy 평가 러너(runner.py, 삭제됨)에 있던 것을 분리했다.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile


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
    try:
        src = sqlite3.connect(src_path)
        try:
            dst = sqlite3.connect(tmp_path)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    except Exception:
        # mkstemp가 만든 파일은 backup() 호출 전부터 이미 디스크에 존재한다 — 백업 도중
        # 실패(디스크 부족 등)해도 반쯤 만들어진 사본을 지우지 않으면 그대로 누적된다
        # (실측: disk I/O error로 28.9GB 임시 사본이 안 지워져 디스크가 꽉 찬 사고).
        _cleanup_copy(tmp_path)
        raise
    return tmp_path


def _cleanup_copy(copy_path: str) -> None:
    for path in (copy_path, f"{copy_path}-wal", f"{copy_path}-shm"):
        if os.path.exists(path):
            os.remove(path)
