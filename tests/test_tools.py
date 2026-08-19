import pytest

from llama4codex import tools as tools_module
from llama4codex.tools import flat_name, map_namespace_tools
from llama4codex.compat import transform_request


def test_namespace_tools_are_flattened_and_reversible() -> None:
    flattened, reverse = map_namespace_tools(
        "mcp__codebase-memory-mcp",
        [{"name": "search_graph", "description": "search", "parameters": {"type": "object"}}],
    )

    assert flattened[0]["name"] == "mcp__codebase-memory-mcp__search_graph"
    assert reverse[flattened[0]["name"]].tool_name == "search_graph"


def test_map_namespace_tools_rejects_custom_children() -> None:
    with pytest.raises(ValueError, match="Unsupported namespace tool type: custom"):
        map_namespace_tools(
            "ns",
            [{"type": "custom", "name": "apply_patch"}],
        )


def test_map_namespace_tools_rejects_invalid_namespace_and_child_names() -> None:
    with pytest.raises(ValueError, match="Namespace requires a non-empty name"):
        map_namespace_tools(None, [])
    with pytest.raises(ValueError, match="Namespace tool requires a non-empty name"):
        map_namespace_tools("ns", [{"type": "function", "name": []}])


def test_same_inner_name_has_distinct_aliases() -> None:
    assert flat_name("one", "search") != flat_name("two", "search")


def test_unsafe_long_names_are_deterministic_and_bounded() -> None:
    namespace = "namespace with spaces/" + "x" * 100
    first = flat_name(namespace, "tool/name")

    assert first == flat_name(namespace, "tool/name")
    assert len(first) <= 64


def test_native_web_search_is_filtered_but_function_name_is_kept() -> None:
    transformed = transform_request(
        {
            "tools": [
                {"type": "web_search"},
                {"type": "function", "name": "web_search", "parameters": {}},
            ]
        }
    )

    assert transformed.payload["tools"] == [
        {"type": "function", "name": "web_search", "parameters": {}}
    ]


def test_forced_cross_namespace_collision_is_reserved_request_wide(monkeypatch) -> None:
    monkeypatch.setattr(tools_module, "flat_name", lambda namespace, tool: "forced_alias")
    reserved = set()

    first, first_reverse = map_namespace_tools("first", [{"name": "same"}], reserved)
    second, second_reverse = map_namespace_tools("second", [{"name": "same"}], reserved)

    first_alias = first[0]["name"]
    second_alias = second[0]["name"]
    assert first_alias != second_alias
    assert first_reverse[first_alias].namespace == "first"
    assert second_reverse[second_alias].namespace == "second"
    assert reserved == {first_alias, second_alias}
