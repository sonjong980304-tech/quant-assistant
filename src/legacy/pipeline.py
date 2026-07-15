"""고수준 파이프라인 러너 (그래프 빌드 + 1회 질의 실행)."""
from __future__ import annotations

from datetime import date

from src.db import connect, connect_readonly, schema_catalog
from src.llm import LLM
from src.wiki.store import WikiStore

from .graph.build import build_graph
from .graph.nodes import Deps


class Pipeline:
    def __init__(
        self,
        db_path: str | None = None,
        today: date | None = None,
        model: str | None = None,
        offline: bool = False,
    ):
        # init_db는 앱 시작 시 1회만 수행(app 모듈). 질의 경로에서는 write를 하지 않는다
        # (백필 등 동시 write와의 'database is locked' 방지). 단, 성공한 질의의 기록 저장은 수행한다.
        #
        # 보안(defense-in-depth): LLM이 생성한 신뢰불가 SQL은 읽기전용 연결(roconn)에서만
        # 실행한다 → is_safe_select 필터를 우회당해도 엔진이 쓰기를 거부한다.
        # 앱 내부의 정상 쓰기(WikiStore의 성공 질의 기록)는 쓰기 가능한 self.conn을 쓴다.
        self.conn = connect(db_path)          # 쓰기 가능 — WikiStore 전용
        self.roconn = connect_readonly(db_path)  # 읽기전용 — untrusted SQL 실행/조회
        self.store = WikiStore(self.conn)
        from src.llm import LLMClient

        # offline=True면 키 유무와 무관하게 휴리스틱 폴백만 사용(결정론).
        if offline:
            llm = LLMClient(offline=True)
        elif model:
            llm = LLMClient(model=model)
        else:
            llm = LLM
        self.deps = Deps(
            conn=self.roconn,  # 질의/조회는 읽기전용 연결로만
            store=self.store,
            schema=schema_catalog(),
            today=today,
            llm=llm,
        )
        self.graph = build_graph(self.deps)

    def run(self, question: str, do_eval: bool = False, gold_sql: str | None = None) -> dict:
        init: dict = {"raw_question": question, "do_eval": do_eval, "notes": []}
        if gold_sql:
            init["gold_sql"] = gold_sql
        return self.graph.invoke(init)

    def close(self) -> None:
        self.conn.close()
        self.roconn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
