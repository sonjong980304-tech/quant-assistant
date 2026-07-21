"""factcheck eval 공통 비교 유틸(허용오차/정확 일치) 단위 테스트 (TDD).

.omc/specs/brainstorming-factcheck-eval.md AC1 참고: "오차 경계값(정확히 1%)도
케이스에 포함"을 요구하므로, 경계값 포함/초과 케이스를 각각 명시한다.

대상: src/eval/factcheck/tolerance.py
- within_pct_tolerance(expected, actual, pct) -> bool
    |actual-expected|/|expected| <= pct 이면 True (경계값 포함, <=).
    expected == 0 인 경우 ZeroDivisionError 없이: actual도 0이면 True, 아니면 False로 정의.
- exact_match(expected, actual) -> bool
    숫자(int/float)면 epsilon(1e-9) 이내 근접 시 True, 그 외(문자열 등)는 == 비교.
"""
from __future__ import annotations

from src.eval.factcheck.tolerance import exact_match, within_pct_tolerance


# --------------------------------------------------------------------------
# within_pct_tolerance
# --------------------------------------------------------------------------
class TestWithinPctTolerance:
    def test_exact_match_is_within_tolerance(self):
        assert within_pct_tolerance(100.0, 100.0, 0.01) is True

    def test_error_below_pct_is_within_tolerance(self):
        # 100 -> 100.5 : 오차 0.5%, 허용치 1% 이내
        assert within_pct_tolerance(100.0, 100.5, 0.01) is True

    def test_error_exactly_at_pct_boundary_is_within_tolerance(self):
        # 100 -> 101 : 오차 정확히 1%, pct=0.01 경계값은 True(포함)여야 함
        assert within_pct_tolerance(100.0, 101.0, 0.01) is True

    def test_error_slightly_over_pct_boundary_is_not_within_tolerance(self):
        # 100 -> 101.5 : 오차 1.5%, 허용치 1%를 살짝 초과 -> False
        assert within_pct_tolerance(100.0, 101.5, 0.01) is False

    def test_error_negative_direction_exactly_at_boundary(self):
        # 100 -> 99 : 오차 정확히 1% (음의 방향)도 절대값 기준으로 경계 포함
        assert within_pct_tolerance(100.0, 99.0, 0.01) is True

    def test_expected_zero_and_actual_zero_is_within_tolerance(self):
        # expected==0 이면 나눗셈이 불가하므로 별도 정의: actual도 0이면 True
        assert within_pct_tolerance(0.0, 0.0, 0.01) is True

    def test_expected_zero_and_actual_nonzero_is_not_within_tolerance(self):
        # expected==0, actual!=0 이면 정의상 False (ZeroDivisionError 없이 처리)
        assert within_pct_tolerance(0.0, 5.0, 0.01) is False

    def test_expected_negative_uses_absolute_value_for_denominator(self):
        # expected=-100 -> actual=-101 : 오차 |−1|/|−100| = 1%, 경계값 포함 True
        assert within_pct_tolerance(-100.0, -101.0, 0.01) is True


# --------------------------------------------------------------------------
# exact_match
# --------------------------------------------------------------------------
class TestExactMatch:
    def test_identical_numbers_match(self):
        assert exact_match(100, 100) is True

    def test_identical_strings_match(self):
        assert exact_match("삼성전자", "삼성전자") is True

    def test_different_strings_do_not_match(self):
        assert exact_match("삼성전자", "SK하이닉스") is False

    def test_floats_within_epsilon_match(self):
        # 부동소수점 연산 오차(예: 0.1+0.2 != 0.3) 감안 근접값은 True
        assert exact_match(0.3, 0.1 + 0.2) is True

    def test_floats_outside_epsilon_do_not_match(self):
        assert exact_match(1.0, 1.000001) is False

    def test_int_and_float_equal_value_match(self):
        assert exact_match(100, 100.0) is True

    def test_different_numbers_do_not_match(self):
        assert exact_match(100, 200) is False
