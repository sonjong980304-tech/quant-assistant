"""cli.py의 백테스트 결과 전용 출력 포맷 테스트.

파이프라인이 run_backtest 프리미티브 결과(performance+holdings 중첩 dict)를 돌려주면,
일반 print_table 대신 성과 요약 + 리밸런싱 시점별 종목 목록으로 사람이 읽기 좋게
포맷한다(Fable5 3라운드 조사에서 발견한 갭 3).
"""
from __future__ import annotations

from cli import _is_backtest_result, format_audit_warnings, format_backtest_result


# --------------------------------------------------------------------------
# _is_backtest_result
# --------------------------------------------------------------------------
def test_is_backtest_result_true_when_performance_and_holdings_present():
    row = {"performance": {"cagr": 5.0}, "holdings": [{"date": "2026-01-01", "names": ["삼성전자"]}]}
    assert _is_backtest_result(row) is True


def test_is_backtest_result_false_for_plain_sql_row():
    row = {"name": "삼성전자", "per": 12.3}
    assert _is_backtest_result(row) is False


# --------------------------------------------------------------------------
# format_backtest_result
# --------------------------------------------------------------------------
def test_format_backtest_result_includes_performance_labels_and_values():
    row = {
        "performance": {"total_return": 12.34, "cagr": 5.67, "mdd": -8.9, "sharpe": 0.85, "win_rate": 55.0},
        "holdings": [{"date": "2026-03-31", "names": ["삼성전자", "SK하이닉스"]}],
    }
    text = format_backtest_result(row)
    assert "12.34" in text
    assert "5.67" in text
    assert "-8.9" in text
    assert "0.85" in text


def test_format_backtest_result_lists_holdings_by_date():
    row = {
        "performance": {"cagr": 5.0},
        "holdings": [
            {"date": "2026-03-31", "names": ["삼성전자", "SK하이닉스"]},
            {"date": "2026-06-30", "names": ["카카오"]},
        ],
    }
    text = format_backtest_result(row)
    assert "2026-03-31" in text and "삼성전자" in text and "SK하이닉스" in text
    assert "2026-06-30" in text and "카카오" in text


def test_format_backtest_result_falls_back_to_codes_when_names_missing():
    row = {"performance": {"cagr": 5.0}, "holdings": [{"date": "2026-03-31", "codes": ["000001"]}]}
    text = format_backtest_result(row)
    assert "000001" in text


# --------------------------------------------------------------------------
# format_audit_warnings (AUD-8: 감지된 위험 섹션)
# --------------------------------------------------------------------------
def test_format_audit_warnings_lists_sin_labels_and_messages():
    warnings = [
        {"sin": "snooping", "triggered": True, "message": "사후정당화 의심"},
        {"sin": "signal_decay", "triggered": True, "message": "회전율 과도"},
    ]
    text = format_audit_warnings(warnings)
    assert "감지된 위험" in text
    assert "데이터마이닝·스누핑" in text and "사후정당화 의심" in text
    assert "신호감소·회전율" in text and "회전율 과도" in text


def test_format_audit_warnings_labels_survivorship_unverifiable():
    # US 백테스트의 생존편향 '검증불가' 경고도 사람이 읽을 수 있는 라벨로 렌더링된다.
    # (메시지엔 '생존편향'이라는 단어를 넣지 않아, '생존편향'이 라벨에서 나온 것임을 검증한다.)
    text = format_audit_warnings(
        [{"sin": "survivorship", "triggered": True, "message": "미국 종목은 상장폐지 데이터가 없어 검증 불가"}]
    )
    assert "survivorship" not in text  # 원시 sin 키가 아니라 한국어 라벨로 치환됨
    assert "생존편향" in text          # 라벨에서 나온 문구
    assert "미국 종목은 상장폐지 데이터가 없어 검증 불가" in text


def test_cmd_query_appends_audit_warnings_section(monkeypatch, capsys):
    """cmd_query() 출력 경로에 실제 배선됐는지: 소프트경고가 결과 아래에 첨부된다(AC10)."""
    import cli

    class FakePipeline:
        def run(self, question, do_eval=False):
            return {
                "raw_question": question, "question": question, "route": "pipeline",
                "sql_source": "pipeline", "sql": "{}", "wiki_id": None,
                "rows": [{"performance": {"cagr": 5.0},
                          "holdings": [{"date": "2020-03-31", "names": ["가나전자"]}]}],
                "columns": [], "row_count": 1, "error": None,
                "audit_warnings": [{"sin": "snooping", "triggered": True, "message": "사후정당화 의심"}],
            }

        def close(self):
            pass

    monkeypatch.setattr("src.legacy.pipeline.Pipeline", FakePipeline)
    monkeypatch.setattr("src.factors.fama_french.handle_query", lambda q: None)
    args = type("Args", (), {"question": "PER 낮은 종목 백테스트", "eval": False})()
    cli.cmd_query(args)
    out = capsys.readouterr().out
    assert "감지된 위험" in out
    assert "사후정당화 의심" in out
    assert "가나전자" in out  # 정상 결과도 그대로 유지
