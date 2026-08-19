import asyncio

import pytest

from llama4codex.compat import (
    apply_reasoning_compat,
    normalize_input_message_order,
    normalize_client_tool_search_history,
    ToolSearchIdentity,
    transform_sse_stream,
    transform_request,
    transform_response,
)


def message(role: str, marker: str) -> dict:
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": marker}],
        "id": f"id-{marker}",
        "status": "completed",
    }


def test_normalizes_codex_developer_user_pattern_stably() -> None:
    payload = {
        "instructions": "keep me unchanged",
        "input": [
            message("developer", "A"),
            message("user", "A"),
            message("developer", "B"),
            message("developer", "C"),
            message("user", "B"),
        ],
    }

    transformed = normalize_input_message_order(payload)

    assert [item["role"] for item in transformed["input"]] == [
        "developer",
        "developer",
        "developer",
        "user",
        "user",
    ]
    assert [item["content"][0]["text"] for item in transformed["input"]] == [
        "A",
        "B",
        "C",
        "A",
        "B",
    ]
    assert transformed["instructions"] == payload["instructions"]


def test_normalization_preserves_already_ordered_and_single_role_inputs() -> None:
    for roles in [
        ["developer", "developer", "user", "user"],
        ["user", "user"],
        ["developer", "developer"],
    ]:
        payload = {"input": [message(role, str(index)) for index, role in enumerate(roles)]}
        assert normalize_input_message_order(payload) == payload


def test_normalization_ignores_string_or_missing_input() -> None:
    assert normalize_input_message_order({"input": "hello"}) == {"input": "hello"}
    assert normalize_input_message_order({"instructions": "system"}) == {
        "instructions": "system"
    }


def test_normalization_keeps_nontarget_positions_and_content() -> None:
    function_call = {"type": "function_call", "call_id": "call-x", "name": "tool"}
    unknown = {"type": "custom_item", "payload": {"keep": True}}
    payload = {
        "input": [
            message("developer", "A"),
            message("user", "A"),
            function_call,
            message("developer", "B"),
            unknown,
            message("user", "B"),
        ]
    }

    transformed = normalize_input_message_order(payload)

    assert [item.get("role", item.get("type")) for item in transformed["input"]] == [
        "developer",
        "developer",
        "function_call",
        "user",
        "custom_item",
        "user",
    ]
    assert transformed["input"][2] == function_call
    assert transformed["input"][4] == unknown
    assert payload["input"][1]["role"] == "user"


def test_transform_request_applies_input_order_without_mutating_payload() -> None:
    payload = {"input": [message("user", "A"), message("developer", "B")]}

    transformed = transform_request(payload)

    assert [item["role"] for item in transformed.payload["input"]] == ["developer", "user"]
    assert [item["role"] for item in payload["input"]] == ["user", "developer"]


def client_tool_search(description="ORIGINAL", parameters=None) -> dict:
    return {
        "type": "tool_search",
        "execution": "client",
        "description": description,
        "parameters": parameters or {"type": "object", "properties": {"query": {}}},
    }


def test_client_tool_search_lowers_to_function_without_mutating_payload() -> None:
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}}
    payload = {"tools": [client_tool_search("ORIGINAL", parameters)]}

    transformed = transform_request(payload)
    lowered = transformed.payload["tools"][0]

    assert lowered == {
        "type": "function",
        "name": "tool_search",
        "description": "ORIGINAL",
        "parameters": parameters,
    }
    assert transformed.tool_reverse_map["tool_search"] == ToolSearchIdentity()
    assert payload["tools"][0]["type"] == "tool_search"
    assert payload["tools"][0]["execution"] == "client"


def test_client_tool_search_reserves_name_across_namespace_order() -> None:
    payload = {
        "tools": [
            {"type": "namespace", "name": "ns", "tools": [{"name": "tool_search"}]},
            client_tool_search(),
        ]
    }

    transformed = transform_request(payload)

    assert transformed.payload["tools"][0]["name"] != "tool_search"
    assert transformed.payload["tools"][1]["name"] == "tool_search"


def test_client_tool_search_collides_with_ordinary_function() -> None:
    with pytest.raises(ValueError, match="Ambiguous tool_search"):
        transform_request(
            {"tools": [{"type": "function", "name": "tool_search"}, client_tool_search()]}
        )


def test_non_client_tool_search_keeps_existing_passthrough_policy() -> None:
    tool = {"type": "tool_search", "description": "original", "parameters": {}}
    transformed = transform_request({"tools": [tool]})

    assert transformed.payload["tools"] == [tool]
    assert transformed.tool_reverse_map == {}


def test_tool_search_response_lifting_preserves_fields_and_parses_arguments() -> None:
    payload = {
        "output": [
            {
                "type": "function_call",
                "id": "item-1",
                "name": "tool_search",
                "arguments": '{"query":"memory","limit":4}',
                "call_id": "call-1",
                "status": "completed",
            }
        ]
    }

    from llama4codex.compat import transform_response

    lifted = transform_response(payload, {"tool_search": ToolSearchIdentity()})["output"][0]

    assert lifted == {
        "type": "tool_search_call",
        "id": "item-1",
        "arguments": {"query": "memory", "limit": 4},
        "call_id": "call-1",
        "status": "completed",
        "execution": "client",
    }
    assert payload["output"][0]["type"] == "function_call"


def test_ordinary_and_unknown_function_calls_are_not_tool_search_lifted() -> None:
    payload = {
        "output": [
            {"type": "function_call", "name": "tool_search", "arguments": "{}"},
            {"type": "function_call", "name": "unknown", "arguments": "{}"},
        ]
    }
    from llama4codex.compat import transform_response

    output = transform_response(payload, {"ordinary": ToolSearchIdentity()})["output"]

    assert output == payload["output"]


def test_sse_tool_search_lifting_handles_split_lf_frames_and_done() -> None:
    async def chunks():
        frame = (
            b'event: response.output_item.done\n'
            b'data: {"item":{"type":"function_call","name":"tool_search",'
            b'"arguments":"{\\"query\\":\\"memory\\"}","call_id":"call-1"}}\n\n'
            b'data: [DONE]\n\n'
        )
        for index in [7, 31, 62, len(frame)]:
            yield frame[:index]
            frame = frame[index:]
            if not frame:
                break

    async def collect():
        return b"".join(
            [chunk async for chunk in transform_sse_stream(chunks(), {"tool_search": ToolSearchIdentity()})]
        )

    output = asyncio.run(collect())

    assert b'"type":"tool_search_call"' in output
    assert b'"execution":"client"' in output
    assert b'"arguments":{"query":"memory"}' in output
    assert output.endswith(b"data: [DONE]\n\n")


def test_sse_tool_search_lifting_handles_crlf_and_non_target_events() -> None:
    async def chunks():
        yield (
            b"event: response.output_text.delta\r\n"
            b'data: {"delta":"before"}\r\n\r\n'
            b"event: response.output_item.done\r\n"
            b'data: {"item":{"type":"function_call","name":"ordinary","arguments":"{}"}}\r\n\r\n'
            b"data: [DONE]\r\n\r\n"
        )

    async def collect():
        return b"".join(
            [chunk async for chunk in transform_sse_stream(chunks(), {"tool_search": ToolSearchIdentity()})]
        )

    output = asyncio.run(collect())

    assert b'"delta":"before"' in output
    assert b'"type":"function_call"' in output
    assert b'data: [DONE]\r\n\r\n' in output


def tool_search_call(arguments=None, **extra) -> dict:
    item = {
        "type": "tool_search_call",
        "id": "fc-1",
        "call_id": "call-1",
        "status": "completed",
        "execution": "client",
        "arguments": arguments if arguments is not None else {"query": "memory", "limit": 4},
    }
    item.update(extra)
    return item


def tool_search_output(tools, **extra) -> dict:
    item = {
        "type": "tool_search_output",
        "id": "tso-1",
        "call_id": "call-1",
        "status": "completed",
        "execution": "client",
        "tools": tools,
    }
    item.update(extra)
    return item


def test_tool_search_history_lowers_call_and_empty_output() -> None:
    payload = {"input": [tool_search_call(), tool_search_output([])]}

    transformed, discovered = normalize_client_tool_search_history(payload)

    assert transformed["input"] == [
        {
            "type": "function_call",
            "id": "fc-1",
            "call_id": "call-1",
            "name": "tool_search",
            "arguments": '{"query":"memory","limit":4}',
        },
        {
            "type": "function_call_output",
            "id": "tso-1",
            "call_id": "call-1",
            "output": '{"tools":[]}',
        },
    ]
    assert discovered == []
    assert payload["input"][0]["type"] == "tool_search_call"


def test_tool_search_history_preserves_unrelated_item_positions() -> None:
    reasoning = {"type": "reasoning", "id": "r-1"}
    message_item = message("user", "hello")
    payload = {"input": [reasoning, tool_search_call(), tool_search_output([]), message_item]}

    transformed, _ = normalize_client_tool_search_history(payload)

    assert transformed["input"][0] == reasoning
    assert transformed["input"][1]["type"] == "function_call"
    assert transformed["input"][2]["type"] == "function_call_output"
    assert transformed["input"][3] == message_item


def test_tool_search_history_requires_valid_client_call_id_and_tools() -> None:
    with pytest.raises(ValueError, match="call_id"):
        normalize_client_tool_search_history({"input": [tool_search_call(call_id=None)]})
    with pytest.raises(ValueError, match="tools array"):
        normalize_client_tool_search_history({"input": [tool_search_output(None)]})


def test_non_client_tool_search_history_is_not_lowered() -> None:
    call = tool_search_call(execution="server")
    output = tool_search_output([], execution="server")
    payload = {"input": [call, output]}

    transformed, discovered = normalize_client_tool_search_history(payload)

    assert transformed == payload
    assert discovered == []


def test_discovered_direct_function_is_exposed_and_deduplicated() -> None:
    discovered = {
        "type": "function",
        "name": "search_graph",
        "description": "ORIGINAL",
        "parameters": {"type": "object"},
    }
    payload = {"tools": [client_tool_search()], "input": [tool_search_output([discovered]), tool_search_output([discovered])]}

    transformed = transform_request(payload)

    assert transformed.payload["tools"] == [
        {"type": "function", "name": "tool_search", "description": "ORIGINAL", "parameters": client_tool_search()["parameters"]},
        discovered,
    ]
    assert payload["input"][0]["type"] == "tool_search_output"


def test_discovered_namespace_uses_existing_flattening_and_reverse_map() -> None:
    namespace = {
        "type": "namespace",
        "name": "mcp__memory",
        "tools": [{"name": "search_graph", "description": "ORIGINAL", "parameters": {}}],
    }
    transformed = transform_request(
        {"tools": [client_tool_search()], "input": [tool_search_output([namespace])]}
    )

    flattened = [tool for tool in transformed.payload["tools"] if tool["name"] != "tool_search"][0]
    assert flattened["type"] == "function"
    assert transformed.tool_reverse_map[flattened["name"]].namespace == "mcp__memory"
    assert transformed.tool_reverse_map[flattened["name"]].tool_name == "search_graph"


def test_discovered_function_reserves_name_and_conflicts_are_explicit() -> None:
    discovered_function = {"type": "function", "name": "foo", "parameters": {}}
    namespace = {"type": "namespace", "name": "ns", "tools": [{"name": "foo"}]}
    transformed = transform_request(
        {"tools": [client_tool_search()], "input": [tool_search_output([discovered_function, namespace])]}
    )
    aliases = [tool["name"] for tool in transformed.payload["tools"]]
    assert aliases[1] == "foo"
    assert aliases[2] != "foo"

    with pytest.raises(ValueError, match="Ambiguous tool_search"):
        transform_request(
            {
                "tools": [client_tool_search()],
                "input": [tool_search_output([{"type": "function", "name": "tool_search"}])],
            }
        )


def test_conflicting_duplicate_discovered_definitions_fail() -> None:
    first = {"type": "function", "name": "foo", "description": "one"}
    second = {"type": "function", "name": "foo", "description": "two"}
    with pytest.raises(ValueError, match="Conflicting discovered function"):
        transform_request({"input": [tool_search_output([first, second])]})


@pytest.mark.parametrize(
    "function",
    [
        {"type": "function", "name": None},
        {"type": "function", "name": ""},
        {"type": "function", "name": []},
        {"type": "function"},
    ],
)
def test_invalid_function_names_raise_value_error(function) -> None:
    with pytest.raises(ValueError, match="Function requires a non-empty name"):
        transform_request({"tools": [function]})



def test_identical_top_level_and_discovered_functions_are_exposed_once() -> None:
    function = {
        "type": "function",
        "name": "foo",
        "description": "same",
        "parameters": {"type": "object"},
    }

    transformed = transform_request(
        {"tools": [function], "input": [tool_search_output([function])]}
    )

    assert transformed.payload["tools"] == [function]


def test_conflicting_top_level_and_discovered_functions_fail() -> None:
    top_level = {"type": "function", "name": "foo", "description": "one"}
    discovered = {"type": "function", "name": "foo", "description": "two"}

    with pytest.raises(ValueError, match="Conflicting .*function"):
        transform_request({"tools": [top_level], "input": [tool_search_output([discovered])]})


def test_discovered_tool_search_function_collides_with_client_declaration() -> None:
    discovered = {"type": "function", "name": "tool_search", "parameters": {}}

    with pytest.raises(ValueError, match="Ambiguous tool_search"):
        transform_request(
            {"tools": [client_tool_search()], "input": [tool_search_output([discovered])]}
        )


def test_namespaces_merge_across_top_level_and_discovered_sources() -> None:
    top_level = {
        "type": "namespace",
        "name": "codebase",
        "tools": [
            {
                "type": "function",
                "name": "list_projects",
                "description": "list",
                "parameters": {"type": "object"},
            }
        ],
    }
    discovered = {
        "type": "namespace",
        "name": "codebase",
        "tools": [
            {
                "type": "function",
                "name": "search_graph",
                "description": "search",
                "parameters": {"type": "object"},
            }
        ],
    }

    transformed = transform_request(
        {"tools": [top_level], "input": [tool_search_output([discovered])]}
    )

    emitted = transformed.payload["tools"]
    assert [tool["name"] for tool in emitted] == [
        "codebase__list_projects",
        "codebase__search_graph",
    ]
    assert all(tool["type"] == "function" for tool in emitted)
    assert transformed.tool_reverse_map["codebase__list_projects"].tool_name == "list_projects"
    assert transformed.tool_reverse_map["codebase__list_projects"].namespace == "codebase"
    assert transformed.tool_reverse_map["codebase__search_graph"].tool_name == "search_graph"
    assert transformed.tool_reverse_map["codebase__search_graph"].namespace == "codebase"


def test_identical_namespace_child_across_sources_is_exposed_once() -> None:
    child = {"type": "function", "name": "foo", "parameters": {}}
    namespace = {"type": "namespace", "name": "ns", "tools": [child]}

    transformed = transform_request(
        {"tools": [namespace], "input": [tool_search_output([namespace])]}
    )

    assert transformed.payload["tools"] == [
        {"type": "function", "name": "ns__foo", "parameters": {}}
    ]


def test_conflicting_namespace_child_across_sources_fails() -> None:
    top_level = {
        "type": "namespace",
        "name": "ns",
        "tools": [{"type": "function", "name": "foo", "description": "one"}],
    }
    discovered = {
        "type": "namespace",
        "name": "ns",
        "tools": [{"type": "function", "name": "foo", "description": "two"}],
    }

    with pytest.raises(ValueError, match="Conflicting namespace tool definition"):
        transform_request(
            {"tools": [top_level], "input": [tool_search_output([discovered])]}
        )


def test_repeated_namespace_child_across_all_sources_is_exposed_once() -> None:
    child = {"type": "function", "name": "foo", "parameters": {}}
    namespace = {"type": "namespace", "name": "ns", "tools": [child]}

    transformed = transform_request(
        {
            "tools": [namespace],
            "input": [tool_search_output([namespace]), tool_search_output([namespace])],
        }
    )

    assert [tool["name"] for tool in transformed.payload["tools"]] == ["ns__foo"]


@pytest.mark.parametrize("source", ["top-level", "discovered"])
def test_namespace_custom_children_are_rejected(source) -> None:
    namespace = {
        "type": "namespace",
        "name": "ns",
        "tools": [{"type": "custom", "name": "apply_patch"}],
    }
    payload = (
        {"tools": [namespace]}
        if source == "top-level"
        else {"input": [tool_search_output([namespace])]}
    )

    with pytest.raises(ValueError, match="Unsupported namespace tool type: custom"):
        transform_request(payload)


def test_namespace_function_child_preserves_compatible_fields() -> None:
    namespace = {
        "type": "namespace",
        "name": "ns",
        "tools": [
            {
                "type": "function",
                "name": "foo",
                "description": "desc",
                "parameters": {"type": "object"},
                "strict": True,
            }
        ],
    }

    transformed = transform_request({"tools": [namespace]})

    assert transformed.payload["tools"] == [
        {
            "type": "function",
            "name": "ns__foo",
            "description": "desc",
            "parameters": {"type": "object"},
            "strict": True,
        }
    ]


@pytest.mark.parametrize("namespace_name", [None, "", [], {}])
def test_invalid_namespace_names_raise_value_error(namespace_name) -> None:
    namespace = {"type": "namespace", "name": namespace_name, "tools": []}

    with pytest.raises(ValueError, match="Namespace requires a non-empty name"):
        transform_request({"tools": [namespace]})


@pytest.mark.parametrize("child_name", [None, "", [], {}])
def test_invalid_namespace_child_names_raise_value_error(child_name) -> None:
    namespace = {
        "type": "namespace",
        "name": "ns",
        "tools": [{"type": "function", "name": child_name}],
    }

    with pytest.raises(ValueError, match="Namespace tool requires a non-empty name"):
        transform_request({"tools": [namespace]})

@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        ("none", 0),
        ("minimal", 1024),
        ("low", 1024),
        ("medium", 4096),
        ("high", 8192),
        ("xhigh", 16384),
        ("max", 32768),
    ],
)
def test_nested_reasoning_effort_is_preserved_and_gets_safety_budget(effort, budget) -> None:
    assert apply_reasoning_compat({"reasoning": {"effort": effort}}) == {
        "reasoning": {"effort": effort},
        "thinking_budget_tokens": budget,
    }


def test_top_level_reasoning_effort_maps_to_budget() -> None:
    assert apply_reasoning_compat({"reasoning_effort": "low"}) == {
        "reasoning_effort": "low",
        "thinking_budget_tokens": 1024,
    }


def test_explicit_budget_wins() -> None:
    assert apply_reasoning_compat(
        {"reasoning": {"effort": "high"}, "thinking_budget_tokens": 77}
    ) == {"reasoning": {"effort": "high"}, "thinking_budget_tokens": 77}


def test_missing_effort_does_not_invent_budget() -> None:
    payload = {"input": "hello"}

    assert apply_reasoning_compat(payload) == payload


def test_unknown_effort_is_preserved_without_safety_budget() -> None:
    payload = {"reasoning": {"effort": "future_value"}}

    assert apply_reasoning_compat(payload) == payload


def test_reasoning_without_effort_is_unchanged() -> None:
    payload = {"reasoning": {"summary": "auto"}}

    assert apply_reasoning_compat(payload) == payload


def test_input_payload_is_not_mutated() -> None:
    payload = {"reasoning": {"effort": "medium"}, "metadata": {"source": "test"}}

    apply_reasoning_compat(payload)

    assert payload == {"reasoning": {"effort": "medium"}, "metadata": {"source": "test"}}


def test_unknown_and_ordinary_function_calls_are_passthrough() -> None:
    payload = {
        "output": [
            {"type": "function_call", "name": "ordinary", "arguments": "{}"},
            {"type": "function_call", "name": "unknown_alias", "arguments": "{\"x\":1}"},
        ]
    }

    assert transform_response(payload, {}) == payload
