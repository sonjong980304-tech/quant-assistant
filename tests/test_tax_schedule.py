"""매도 증권거래세+농특세 합산 실효세율 연도별 스케줄 (stt_rate_at).

references/securities_transaction_tax_rate_history.md 의 1차 출처 검증 표를 그대로
날짜→세율 순수 함수로 옮긴 것. 경계일 전후를 모두 확인해 시행일 오프바이원을 방지한다.
"""
from __future__ import annotations

from src.backtest.tax_schedule import stt_rate_at


def test_pre_2019_flat_rate():
    assert stt_rate_at("2014-01-02") == 0.0030
    assert stt_rate_at("2019-06-02") == 0.0030  # 인하 시행 전날


def test_2019_rate_cut_boundary():
    assert stt_rate_at("2019-06-03") == 0.0025  # 시행 첫날
    assert stt_rate_at("2020-12-31") == 0.0025


def test_2021_rate_cut_boundary():
    assert stt_rate_at("2021-01-01") == 0.0023
    assert stt_rate_at("2022-12-31") == 0.0023


def test_2023_rate_cut_boundary():
    assert stt_rate_at("2023-01-01") == 0.0020
    assert stt_rate_at("2023-12-31") == 0.0020


def test_2024_rate_boundary():
    assert stt_rate_at("2024-01-01") == 0.0018
    assert stt_rate_at("2024-12-31") == 0.0018


def test_2025_zero_transaction_tax_boundary():
    assert stt_rate_at("2025-01-01") == 0.0015
    assert stt_rate_at("2025-12-31") == 0.0015


def test_2026_rate_hike_boundary():
    assert stt_rate_at("2026-01-01") == 0.0020
    assert stt_rate_at("2026-07-26") == 0.0020
