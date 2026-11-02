"""
Smoke tests — run without a live Groq key or MLflow server.
They test only the utility functions that have no external dependencies.
"""
import pytest
from unittest.mock import MagicMock


# ── Helpers copied inline so tests don't need the full app imported ───────────

def extract_text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in msg.content
        )
    return ""


def is_final_ai(msg) -> bool:
    tool_calls = getattr(msg, "tool_calls", None) or []
    return msg.type == "ai" and not tool_calls


# ── extract_text ──────────────────────────────────────────────────────────────

def test_extract_text_plain_string():
    msg = MagicMock()
    msg.content = "Hello, Kyoto!"
    assert extract_text(msg) == "Hello, Kyoto!"


def test_extract_text_list_of_blocks():
    msg = MagicMock()
    msg.content = [{"type": "text", "text": "Day 1: "}, {"type": "text", "text": "Arrive"}]
    assert extract_text(msg) == "Day 1: Arrive"


def test_extract_text_empty_list():
    msg = MagicMock()
    msg.content = []
    assert extract_text(msg) == ""


def test_extract_text_non_dict_blocks():
    msg = MagicMock()
    msg.content = ["raw string block"]
    assert extract_text(msg) == "raw string block"


# ── is_final_ai ───────────────────────────────────────────────────────────────

def test_is_final_ai_true_when_no_tool_calls():
    msg = MagicMock()
    msg.type = "ai"
    msg.tool_calls = []
    assert is_final_ai(msg) is True


def test_is_final_ai_false_when_tool_calls_present():
    msg = MagicMock()
    msg.type = "ai"
    msg.tool_calls = [{"name": "get_weather_forecast"}]
    assert is_final_ai(msg) is False


def test_is_final_ai_handles_none_tool_calls():
    """Groq sometimes returns tool_calls=None instead of []."""
    msg = MagicMock()
    msg.type = "ai"
    msg.tool_calls = None
    assert is_final_ai(msg) is True


def test_is_final_ai_false_for_tool_message():
    msg = MagicMock()
    msg.type = "tool"
    msg.tool_calls = []
    assert is_final_ai(msg) is False
