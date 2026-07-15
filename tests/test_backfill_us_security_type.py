"""scripts/backfill_us_security_type.py 검증 (TDD).

배경: 미국 스크리닝(저PER 등)에서 Warrant/ADR 같은 파생·특수 증권이 시총이 비정상적으로
작아 PER이 비현실적으로 낮게 계산되고, 그 결과 상위권을 차지하는 문제가 실서버에서
재현됐다(HA15 후속). us_company.name에는 이미 "CEA Industries Inc. Warrant",
"Cresud S.A.C.I.F. y A. American Depositary Shares" 처럼 판별에 필요한 정보가 그대로
들어있으므로, 정규식/접미어 키워드 목록이 아니라 LLM이 회사명 전체 문자열의 의미를
읽고 판단하게 한다(사용자 명시 지시).

"판단은 AI, 실행은 캐싱" 원칙: 7,123개 US 종목 전체를 매 스크리닝 질문마다 LLM에
물어보면 비용/속도가 감당 안 되므로, 이 스크립트가 **1회 배치**로 전체 종목을 분류해
us_company.security_type 컬럼에 캐싱한다. 스크리닝 런타임(src/agents/domain_us.py)은
이 캐시를 읽기만 한다.

검증 대상:
- _security_type_prompt: 배치 프롬프트에 (티커, 회사명) 목록이 포함된다.
- _parse_security_types: LLM 응답("티커: 유형" 줄 목록)을 파싱해 dict로 만든다
  (대소문자/공백 관대, 매치 안 되는 줄은 무시).
- classify_batch: llm_fn을 호출해 파싱까지 완료한 dict를 반환한다(DI, 실제 LLM 미사용).
- backfill_us_security_type: DB에서 security_type이 NULL인 종목만 대상으로 배치 분류해
  UPDATE하고, 이미 분류된 종목은 재분류하지 않으며(idempotent), dry_run=True면 DB를
  변경하지 않고, 분류 결과 카운트를 리포트로 반환한다.
"""
from __future__ import annotations

from scripts.backfill_us_security_type import (
    _parse_security_types,
    _security_type_prompt,
    backfill_us_security_type,
    classify_batch,
)
from src.db import connect, init_db


# ── _security_type_prompt ────────────────────────────────────────────────────
def test_security_type_prompt_includes_ticker_and_name():
    prompt = _security_type_prompt([("AAPL", "Apple Inc."), ("BNCWW", "CEA Industries Inc. Warrant")])
    assert "AAPL" in prompt
    assert "Apple Inc." in prompt
    assert "BNCWW" in prompt
    assert "CEA Industries Inc. Warrant" in prompt


def test_security_type_prompt_lists_valid_type_labels():
    prompt = _security_type_prompt([("AAPL", "Apple Inc.")])
    for label in ("common", "warrant", "adr", "preferred", "unit", "right", "other"):
        assert label in prompt


# ── _parse_security_types ────────────────────────────────────────────────────
def test_parse_security_types_extracts_ticker_type_pairs():
    raw = "AAPL: common\nBNCWW: warrant\nCRESY: adr"
    result = _parse_security_types(raw, {"AAPL", "BNCWW", "CRESY"})
    assert result == {"AAPL": "common", "BNCWW": "warrant", "CRESY": "adr"}


def test_parse_security_types_is_case_and_whitespace_tolerant():
    raw = "  aapl :  Common  \n bncww:WARRANT"
    result = _parse_security_types(raw, {"AAPL", "BNCWW"})
    assert result == {"AAPL": "common", "BNCWW": "warrant"}


def test_parse_security_types_ignores_unknown_ticker_not_in_batch():
    raw = "AAPL: common\nZZZZ: warrant"  # ZZZZ는 이번 배치에 없던 티커(환각) → 무시
    result = _parse_security_types(raw, {"AAPL"})
    assert result == {"AAPL": "common"}


def test_parse_security_types_ignores_unparseable_lines():
    raw = "AAPL: common\n이건 그냥 설명입니다\nBNCWW: 이상한값"
    result = _parse_security_types(raw, {"AAPL", "BNCWW"})
    assert result == {"AAPL": "common"}  # BNCWW는 유효 라벨이 아니라 제외


# ── classify_batch ────────────────────────────────────────────────────────────
def test_classify_batch_calls_llm_fn_and_parses_result():
    seen_prompts = []

    def fake_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "AAPL: common\nBNCWW: warrant"

    result = classify_batch(fake_llm, [("AAPL", "Apple Inc."), ("BNCWW", "CEA Industries Inc. Warrant")])
    assert result == {"AAPL": "common", "BNCWW": "warrant"}
    assert len(seen_prompts) == 1


def test_classify_batch_returns_empty_dict_on_llm_exception():
    def boom(prompt: str) -> str:
        raise RuntimeError("LLM 다운")

    result = classify_batch(boom, [("AAPL", "Apple Inc.")])
    assert result == {}


# ── backfill_us_security_type: end-to-end (임시 DB) ──────────────────────────
def _seed_companies(db_path: str, companies: list[tuple[str, str, str | None]]) -> None:
    """companies: [(stock_code, name, initial_security_type_or_None), ...]."""
    conn = connect(db_path)
    try:
        for code, name, sec_type in companies:
            conn.execute(
                "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, "
                "security_type, updated_at) VALUES (?,?,?,?,?,?,?)",
                (code, name, "NASDAQ", "Technology", 1.0e9, sec_type, "2026-07-01"),
            )
        conn.commit()
    finally:
        conn.close()


def _fake_batch_llm(prompt: str) -> str:
    """이름에 뻔한 신호가 있는 소규모 fixture 데이터용 결정론 가짜 LLM(실제 LLM 미사용)."""
    lines = []
    for line in prompt.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        code, _, name = line.partition(":")
        code, name = code.strip(), name.strip().lower()
        if not code or not code.isupper() or len(code) > 6:
            continue
        if "warrant" in name:
            lines.append(f"{code}: warrant")
        elif "depositary" in name or "depository" in name:
            lines.append(f"{code}: adr")
        elif "preferred" in name:
            lines.append(f"{code}: preferred")
        else:
            lines.append(f"{code}: common")
    return "\n".join(lines)


def test_backfill_us_security_type_classifies_null_rows_and_reports_counts(tmp_path):
    db = tmp_path / "backfill_sec.db"
    init_db(str(db))
    _seed_companies(
        str(db),
        [
            ("AAPL", "Apple Inc.", None),
            ("BNCWW", "CEA Industries Inc. Warrant", None),
            ("CRESY", "Cresud S.A.C.I.F. y A. American Depositary Shares", None),
            ("PFDX", "Some Corp Preferred Stock", None),
        ],
    )
    report = backfill_us_security_type(str(db), llm_fn=_fake_batch_llm, batch_size=10)

    assert report["total_targets"] == 4
    assert report["classified"] == 4
    assert report["counts"]["common"] == 1
    assert report["counts"]["warrant"] == 1
    assert report["counts"]["adr"] == 1
    assert report["counts"]["preferred"] == 1

    conn = connect(str(db))
    try:
        rows = {
            r["stock_code"]: r["security_type"]
            for r in conn.execute("SELECT stock_code, security_type FROM us_company")
        }
    finally:
        conn.close()
    assert rows == {"AAPL": "common", "BNCWW": "warrant", "CRESY": "adr", "PFDX": "preferred"}


def test_backfill_us_security_type_skips_already_classified_rows(tmp_path):
    db = tmp_path / "backfill_sec2.db"
    init_db(str(db))
    _seed_companies(
        str(db),
        [
            ("AAPL", "Apple Inc.", "common"),  # 이미 분류됨 → 재분류 대상 아님
            ("BNCWW", "CEA Industries Inc. Warrant", None),
        ],
    )
    report = backfill_us_security_type(str(db), llm_fn=_fake_batch_llm, batch_size=10)

    assert report["total_targets"] == 1  # AAPL은 이미 분류돼 대상에서 제외
    assert report["classified"] == 1
    assert report["counts"]["warrant"] == 1
    assert "common" not in report["counts"] or report["counts"]["common"] == 0


def test_backfill_us_security_type_dry_run_does_not_write_to_db(tmp_path):
    db = tmp_path / "backfill_sec3.db"
    init_db(str(db))
    _seed_companies(str(db), [("AAPL", "Apple Inc.", None)])

    report = backfill_us_security_type(str(db), llm_fn=_fake_batch_llm, batch_size=10, dry_run=True)
    assert report["classified"] == 1  # 분류 자체는 계산됨(리포트용)

    conn = connect(str(db))
    try:
        row = conn.execute("SELECT security_type FROM us_company WHERE stock_code='AAPL'").fetchone()
    finally:
        conn.close()
    assert row["security_type"] is None  # dry-run이라 DB는 미변경


def test_backfill_us_security_type_batches_by_batch_size(tmp_path):
    """batch_size보다 종목 수가 많으면 여러 번(2회 이상) llm_fn을 호출한다."""
    db = tmp_path / "backfill_sec4.db"
    init_db(str(db))
    companies = [(f"T{i:03d}", f"Ticker {i} Inc.", None) for i in range(5)]
    _seed_companies(str(db), companies)

    calls = []

    def counting_llm(prompt: str) -> str:
        calls.append(prompt)
        return _fake_batch_llm(prompt)

    report = backfill_us_security_type(str(db), llm_fn=counting_llm, batch_size=2)
    assert len(calls) == 3  # 5개를 2개씩 묶으면 3배치(2+2+1)
    assert report["classified"] == 5
