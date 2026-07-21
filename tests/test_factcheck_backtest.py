"""run_backtest_pytest_check / run_backtest_index_check 단위 테스트 (US-7, TDD).

.omc/specs/brainstorming-factcheck-eval.md AC5 참고.
- (a) 기존 백테스트 하드차단 pytest(생존편향/미래참조/공매도 가드 + NAV·베타 검증)를
  subprocess로 재실행해 통과 여부를 기록한다.
- (b) 코스피 지수 최근 1년 실제 수익률 대비 동일가중 유니버스 백테스트 수익률을 신규
  시나리오로 대조한다(±2.5%p, tolerance.py의 within_pct_tolerance 재사용).

subprocess.run과 실제 백테스트 계산 함수(_compute_equal_weight_return/
_fetch_kospi_1y_return)는 전부 모킹한다 — 실제 pytest 서브프로세스 실행이나 실제
백테스트 계산은 이 단계에서 하지 않는다(느리고, 계산 자체 정확성은 이미
test_backtest_performance.py 등이 검증한다).
"""
from __future__ import annotations

from types import SimpleNamespace

from src.eval.factcheck import backtest as bt


# --------------------------------------------------------------------------
# run_backtest_pytest_check (AC5a)
# --------------------------------------------------------------------------
class TestRunBacktestPytestCheck:
    def test_all_pass_reports_pass_true(self, monkeypatch):
        calls = {}

        def fake_run(cmd, cwd=None, capture_output=None, text=None):
            calls["cmd"] = cmd
            calls["cwd"] = cwd
            return SimpleNamespace(returncode=0, stdout="...\n5 passed in 1.23s\n", stderr="")

        monkeypatch.setattr(bt.subprocess, "run", fake_run)

        result = bt.run_backtest_pytest_check()

        assert result["exit_code"] == 0
        assert result["pass"] is True
        assert result["summary"] == "5 passed in 1.23s"
        assert result["note"] == ""
        # 실제로 존재하는 대상 파일 3개가 모두 커맨드에 포함됐는지(사전에 find로 확인한
        # 결과 3개 모두 실존 — src/eval/factcheck/backtest.py의 _PYTEST_TARGETS와 일치해야 함)
        for f in bt._PYTEST_TARGETS:
            assert f in calls["cmd"]

    def test_some_fail_reports_pass_false(self, monkeypatch):
        def fake_run(cmd, cwd=None, capture_output=None, text=None):
            return SimpleNamespace(returncode=1, stdout="...\n1 failed, 4 passed in 1.10s\n", stderr="")

        monkeypatch.setattr(bt.subprocess, "run", fake_run)

        result = bt.run_backtest_pytest_check()

        assert result["exit_code"] == 1
        assert result["pass"] is False
        assert result["summary"] == "1 failed, 4 passed in 1.10s"

    def test_missing_target_file_is_skipped_and_noted(self, monkeypatch):
        fake_targets = bt._PYTEST_TARGETS + ["tests/test_does_not_exist_xyz.py"]
        monkeypatch.setattr(bt, "_PYTEST_TARGETS", fake_targets)

        calls = {}

        def fake_run(cmd, cwd=None, capture_output=None, text=None):
            calls["cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout="3 passed in 0.5s\n", stderr="")

        monkeypatch.setattr(bt.subprocess, "run", fake_run)

        result = bt.run_backtest_pytest_check()

        assert "tests/test_does_not_exist_xyz.py" not in calls["cmd"]
        assert "tests/test_does_not_exist_xyz.py" in result["note"]
        assert result["pass"] is True


# --------------------------------------------------------------------------
# run_backtest_index_check (AC5b)
# --------------------------------------------------------------------------
class TestRunBacktestIndexCheck:
    def test_within_tolerance_reports_pass_true(self, monkeypatch):
        copy_calls = []
        cleanup_calls = []

        def fake_isolated_copy(db_path):
            copy_calls.append(db_path)
            return f"/tmp/copy_{len(copy_calls)}.db"

        monkeypatch.setattr(bt, "_isolated_copy", fake_isolated_copy)
        monkeypatch.setattr(bt, "_cleanup_copy", lambda p: cleanup_calls.append(p))
        monkeypatch.setattr(bt, "_fetch_kospi_1y_return", lambda db_path: 0.10)
        monkeypatch.setattr(bt, "_compute_equal_weight_return", lambda db_path, scenario: 0.102)

        scenarios = [{"name": "quarterly", "start_year": 2025, "end_year": 2026, "rebalance": "quarterly"}]
        result = bt.run_backtest_index_check(scenarios, db_path="/data/market.db")

        assert result == [{
            "scenario": "quarterly",
            "expected": 0.10,
            "actual": 0.102,
            "pass": True,
            "note": "",
        }]
        assert copy_calls == ["/data/market.db"]
        assert cleanup_calls == ["/tmp/copy_1.db"]

    def test_missing_kospi_data_reports_unmeasurable(self, monkeypatch):
        cleanup_calls = []

        monkeypatch.setattr(bt, "_isolated_copy", lambda db_path: "/tmp/copy.db")
        monkeypatch.setattr(bt, "_cleanup_copy", lambda p: cleanup_calls.append(p))

        def raise_no_kospi(db_path):
            raise ValueError("prices 테이블에 코스피 지수 데이터가 없음")

        monkeypatch.setattr(bt, "_fetch_kospi_1y_return", raise_no_kospi)

        computed = []
        monkeypatch.setattr(
            bt, "_compute_equal_weight_return",
            lambda db_path, scenario: computed.append(1) or 0.5,
        )

        scenarios = [{"name": "semiannual", "start_year": 2025, "end_year": 2026}]
        result = bt.run_backtest_index_check(scenarios)

        assert len(result) == 1
        assert result[0]["scenario"] == "semiannual"
        assert result[0]["pass"] is None
        assert result[0]["expected"] is None
        assert result[0]["actual"] is None
        assert "측정불가" in result[0]["note"]
        assert computed == []  # 지수 데이터가 없으면 백테스트 계산 자체를 하지 않는다
        assert cleanup_calls == ["/tmp/copy.db"]
