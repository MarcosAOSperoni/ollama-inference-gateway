import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gateway"))

from metrics import calculate_tps


def test_calculate_tps_normal():
    result = calculate_tps(eval_count=100, eval_duration_ns=2_000_000_000)
    assert result == pytest.approx(50.0)


def test_calculate_tps_zero_duration_returns_zero():
    result = calculate_tps(eval_count=100, eval_duration_ns=0)
    assert result == 0.0


def test_calculate_tps_zero_tokens():
    result = calculate_tps(eval_count=0, eval_duration_ns=1_000_000_000)
    assert result == 0.0


def test_calculate_tps_small_batch():
    result = calculate_tps(eval_count=5, eval_duration_ns=500_000_000)
    assert result == pytest.approx(10.0)
