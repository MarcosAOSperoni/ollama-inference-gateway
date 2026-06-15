import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gateway"))

from router import select_model


def test_select_model_explicit_model_takes_priority():
    assert select_model("llama3:8b", "generate") == "llama3:8b"


def test_select_model_classify_routes_to_small():
    result = select_model(None, "classify")
    assert result == "llama3:8b"


def test_select_model_generate_routes_to_default():
    result = select_model(None, "generate")
    assert result == "llama3:70b"


def test_select_model_summarize_routes_to_default():
    result = select_model(None, "summarize")
    assert result == "llama3:70b"


def test_select_model_unknown_task_type_routes_to_default():
    result = select_model(None, "unknown_task")
    assert result == "llama3:70b"


def test_select_model_no_args_routes_to_default():
    result = select_model(None, None)
    assert result == "llama3:70b"
