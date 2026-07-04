import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../flink"))

import pytest
from inference_stats import _compute_stats


def test_p95_of_100_values():
    latencies = list(range(1, 101))  # 1..100
    stats = _compute_stats(latencies, error_count=0)
    assert stats["p95_ms"] == 95


def test_mean_latency():
    stats = _compute_stats([100.0, 200.0], error_count=0)
    assert stats["mean_ms"] == pytest.approx(150.0)


def test_error_rate():
    stats = _compute_stats([100.0] * 10, error_count=2)
    assert stats["error_rate"] == pytest.approx(0.2)


def test_empty_list_returns_zeros():
    stats = _compute_stats([], error_count=0)
    assert stats["p95_ms"] == 0.0
    assert stats["mean_ms"] == 0.0
    assert stats["error_rate"] == 0.0


def test_single_event():
    stats = _compute_stats([500.0], error_count=0)
    assert stats["p95_ms"] == 500.0
    assert stats["mean_ms"] == pytest.approx(500.0)
    assert stats["error_rate"] == 0.0
