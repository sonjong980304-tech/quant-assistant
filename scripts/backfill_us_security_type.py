"""us_company.security_type 배치 분류 — 회사명으로 일반 보통주/워런트/ADR/우선주 등을
LLM이 판단해 채운다(HA15 후속(B)).

배경: 미국 스크리닝(저PER 등)에서 Warrant/ADR 같은 파생·특수 증권이 시총이 비정상적으로
작아 PER이 비현실적으로 낮게 계산되고, 그 결과 상위권을 차지하는 문제가 실서버에서
재현됐다. us_company.name에는 이미 "CEA Industries Inc. Warrant", "Cresud S.A.C.I.F.
y A. American Depositary Shares" 처럼 판별에 필요한 정보가 그대로 들어있으므로, 정규식/
접미어 키워드 목록이 아니라 LLM이 회사명 전체 문자열의 의미를 읽고 판단하게 한다
(사용자 명시 지시 — 이번 세션 내내 반복된 "하드코딩 키워드 함정"을 다시 만들지 않는다).

"판단은 AI, 실행은 캐싱" 원칙: 7,123개 US 종목 전체를 매 스크리닝 질문마다 LLM에
물어보면 비용/속도가 감당 안 되므로, 이 스크립트가 **1회 배치**로(한 번에 batch_size개씩
묶어서) 전체 종목을 분류해 us_company.security_type 컬럼에 캐싱한다. 스크리닝 런타임
(src/agents/domain_us.py::_filter_common_stock)은 이 캐시를 읽기만 한다(매 요청마다
LLM을 부르지 않음).

분류값: 'common'(일반 보통주) | 'warrant'(워런트) | 'adr'(ADR/ADS) | 'preferred'(우선주) |
        'unit'(유닛/결합증권) | 'right'(신주인수권) | 'other'(그 외 특수증권/판단 불가)

idempotent: security_type이 이미 채워진 종목은 재분류 대상에서 제외한다(다시 돌려도 안전,
backfill_marketcap.py 등 기존 스크립트 관례와 동일).

사용법:
  python scripts/backfill_us_security_type.py --dry-run          # 분류 결과만 출력, DB 미반영
  python scripts/backfill_us_security_type.py                    # 실제 UPDATE
  python scripts/backfill_us_security_type.py --limit 50         # 샘플 소량만(정확도 확인용)
  python scripts/backfill_us_security_type.py --batch-size 30    # LLM 프롬프트당 종목 수 조정
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import connect, init_db

_VALID_TYPES: tuple[str, ...] = ("common", "warrant", "adr", "preferred", "unit", "right", "other")
_DEFAULT_BATCH_SIZE = 40


def _security_type_prompt(companies: list[tuple[str, str]]) -> str:
    """companies: [(stock_code, name), ...]. LLM에 배치로 증권종류 분류를 요청하는 프롬프트.

    회사명 전체 문자열의 의미를 LLM이 읽고 판단하게 한다(정규식/접미어 키워드 목록이 아니라
    AI 판단이 1차 — 사용자 명시 지시). 각 줄 "티커: 유형" 형식으로만 답하도록 강제해
    파싱을 단순하게 유지한다(route_question/classify_intent와 동일한 구조화 응답 관례).
    """
    lines = "\n".join(f"{code}: {name}" for code, name in companies)
    return (
        "다음은 미국 증권거래소에 상장된 종목들의 (티커: 회사명) 목록입니다.\n"
        "각 종목이 어떤 유형의 증권인지 회사명 전체 문자열의 의미를 읽고 판단하세요"
        "(고정 키워드 매칭이 아니라 의미 판단).\n"
        "- common: 일반 보통주(회사 자체를 소유하는 일반적인 주식)\n"
        "- warrant: 워런트(신주인수권부 파생상품)\n"
        "- adr: 미국예탁증권(ADR/ADS, 외국기업 주식을 미국에서 거래하도록 예탁한 증권)\n"
        "- preferred: 우선주\n"
        "- unit: 유닛(주식+워런트 등 결합증권)\n"
        "- right: 신주인수권(Rights)\n"
        "- other: 위에 해당하지 않는 특수증권 또는 판단 불가\n\n"
        "각 줄마다 '티커: 유형' 형식으로만 답하세요(설명/번호 매기기 금지).\n\n"
        f"{lines}\n답:"
    )


def _parse_security_types(raw: str, codes: set[str]) -> dict[str, str]:
    """LLM 응답("티커: 유형" 줄 목록)을 파싱한다.

    이번 배치에 없던 티커(환각)나 유효 라벨(_VALID_TYPES)이 아닌 줄은 조용히 무시한다
    (호출부가 미분류로 남겨 다음 실행 또는 런타임 이름 키워드 안전망이 처리).
    """
    out: dict[str, str] = {}
    for line in (raw or "").splitlines():
        line = line.strip().strip("-•* ")
        if not line or ":" not in line:
            continue
        code_part, _, type_part = line.partition(":")
        code = code_part.strip().upper()
        type_val = type_part.strip().lower()
        if code not in codes:
            continue
        matched = next((t for t in _VALID_TYPES if t == type_val or t in type_val), None)
        if matched:
            out[code] = matched
    return out


def classify_batch(llm_fn: Callable[[str], str], companies: list[tuple[str, str]]) -> dict[str, str]:
    """companies 한 배치를 llm_fn으로 분류한다. llm_fn 실패(예외)는 흡수하고 빈 dict를 반환한다
    (해당 배치는 미분류로 남아 다음 실행 또는 이름 키워드 안전망이 처리 — HA-6 _call_with_retry와
    동일하게 예외를 상위로 전파하지 않는다).
    """
    if not companies:
        return {}
    codes = {code for code, _ in companies}
    try:
        raw = llm_fn(_security_type_prompt(companies)) or ""
    except Exception:  # noqa: BLE001 — LLM 실패는 빈 결과로 흡수(해당 배치 미분류로 남음)
        return {}
    return _parse_security_types(raw, codes)


def _default_llm_fn(role: str = "sql"):
    """web/app.py._build_llm_fn / scripts/eval_hierarchical_goldset.py._build_llm_fn과 동일 관례.

    가용하지 않으면(키/데몬 없음) None을 반환한다.
    """
    from src.llm import LLMClient

    client = LLMClient()
    if not client.available:
        return None
    return lambda prompt: (client.complete(prompt, role=role).text or "")


def backfill_us_security_type(
    db_path: str | None = None,
    llm_fn: Callable[[str], str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """security_type이 NULL인 us_company 종목을 배치로 LLM 분류해 UPDATE한다.

    llm_fn 미지정 시 _default_llm_fn()으로 실제 LLM을 구성한다(테스트는 항상 llm_fn을 DI로
    주입 — fama_french.py/optimize_weights의 solve_fn과 동일한 DI 관례). 배치마다 커밋해
    중간에 중단돼도 진행 상황이 남는다(backfill_profiles.py의 체크포인트 관례와 동일 취지).
    dry_run=True면 분류는 계산하되 DB는 변경하지 않는다.

    반환: {"total_targets": int, "classified": int, "unmatched": int,
           "counts": {"common": N, "warrant": N, ...}}.
    """
    init_db(db_path)
    llm_fn = llm_fn or _default_llm_fn()
    conn = connect(db_path)
    try:
        sql = "SELECT stock_code, name FROM us_company WHERE security_type IS NULL"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()
        targets = [(r["stock_code"], r["name"]) for r in rows]

        counts: dict[str, int] = {}
        classified = 0
        unmatched = 0
        for i in range(0, len(targets), batch_size):
            batch = targets[i : i + batch_size]
            if llm_fn is None:
                unmatched += len(batch)
                continue
            result = classify_batch(llm_fn, batch)
            for code, _name in batch:
                sec_type = result.get(code)
                if sec_type is None:
                    unmatched += 1
                    continue
                classified += 1
                counts[sec_type] = counts.get(sec_type, 0) + 1
                if not dry_run:
                    conn.execute(
                        "UPDATE us_company SET security_type = ? WHERE stock_code = ?",
                        (sec_type, code),
                    )
            if not dry_run:
                conn.commit()

        report = {
            "total_targets": len(targets),
            "classified": classified,
            "unmatched": unmatched,
            "counts": counts,
        }
        return report
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="샘플 소량만 처리(정확도 확인용)")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    llm_fn = _default_llm_fn()
    if llm_fn is None:
        print("LLM 미가용(키/데몬 없음) — 중단", flush=True)
        return

    print(
        f"security_type 배치 분류 시작 (batch_size={args.batch_size}, "
        f"limit={args.limit}, dry_run={args.dry_run})",
        flush=True,
    )
    report = backfill_us_security_type(
        llm_fn=llm_fn, batch_size=args.batch_size, limit=args.limit, dry_run=args.dry_run,
    )
    print(f"\n분류 대상 {report['total_targets']}개")
    print(f"분류 완료 {report['classified']}개 / 미분류(파싱실패·환각) {report['unmatched']}개")
    print("유형별 카운트:")
    for t in _VALID_TYPES:
        if report["counts"].get(t):
            print(f"  {t}: {report['counts'][t]}")
    if args.dry_run:
        print("\n[dry-run] 실제 UPDATE는 수행하지 않았습니다.")
    else:
        print("\nUPDATE 및 commit 완료.")


if __name__ == "__main__":
    main()
