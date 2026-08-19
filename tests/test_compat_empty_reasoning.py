from llama4codex.compat import apply_reasoning_compat


def test_empty_reasoning_object_is_removed() -> None:
    assert apply_reasoning_compat({"reasoning": {"effort": "high"}}) == {
        "reasoning": {"effort": "high"},
        "thinking_budget_tokens": 8192,
    }


def test_other_reasoning_fields_are_preserved() -> None:
    assert apply_reasoning_compat(
        {"reasoning": {"effort": "high", "summary": "auto"}}
    ) == {
        "reasoning": {"effort": "high", "summary": "auto"},
        "thinking_budget_tokens": 8192,
    }
