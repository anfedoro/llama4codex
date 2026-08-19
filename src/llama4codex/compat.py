"""Pure request compatibility transforms for llama.cpp."""

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from .tools import ToolIdentity, map_namespace_tools

DEFAULT_REASONING_BUDGETS = {
    "none": 0,
    "minimal": 1024,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}


def apply_reasoning_compat(payload: dict[str, Any]) -> dict[str, Any]:
    """Preserve native reasoning and add an independent safety token budget."""
    transformed = deepcopy(payload)
    reasoning = transformed.get("reasoning")
    nested_effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    top_level_effort = transformed.get("reasoning_effort")
    effort = nested_effort if nested_effort is not None else top_level_effort

    if "thinking_budget_tokens" not in transformed and effort is not None:
        budget = DEFAULT_REASONING_BUDGETS.get(effort)
        if budget is not None:
            transformed["thinking_budget_tokens"] = DEFAULT_REASONING_BUDGETS[effort]

    return transformed


def normalize_input_message_order(payload: dict[str, Any]) -> dict[str, Any]:
    """Place developer messages before user messages without moving other items."""
    transformed = deepcopy(payload)
    input_items = transformed.get("input")
    if not isinstance(input_items, list):
        return transformed

    target_positions = [
        index
        for index, item in enumerate(input_items)
        if isinstance(item, dict)
        and item.get("type") == "message"
        and item.get("role") in {"developer", "user"}
    ]
    target_items = [input_items[index] for index in target_positions]
    ordered_items = [item for item in target_items if item.get("role") == "developer"]
    ordered_items.extend(item for item in target_items if item.get("role") == "user")
    for index, item in zip(target_positions, ordered_items):
        input_items[index] = item
    return transformed


class TransformedRequest:
    def __init__(self, payload: dict[str, Any], tool_reverse_map: dict[str, Any]):
        self.payload = payload
        self.tool_reverse_map = tool_reverse_map


@dataclass(frozen=True)
class ToolSearchIdentity:
    """Request-local identity for the lowered client-side tool_search function."""


def _lower_client_tool_search(tool: dict[str, Any]) -> dict[str, Any]:
    lowered = {
        key: value
        for key, value in tool.items()
        if key not in {"type", "execution", "name"}
    }
    lowered.update(type="function", name="tool_search")
    return lowered


def _require_call_id(item: dict[str, Any]) -> str:
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("Client tool_search history requires a non-empty call_id")
    return call_id


def _lower_tool_search_call(item: dict[str, Any]) -> dict[str, Any]:
    _require_call_id(item)
    if "arguments" not in item:
        raise ValueError("Client tool_search_call requires arguments")
    lowered = {
        key: value for key, value in item.items() if key not in {"type", "execution", "status", "arguments"}
    }
    lowered.update(
        type="function_call",
        name="tool_search",
        arguments=json.dumps(item["arguments"], separators=(",", ":"), ensure_ascii=False),
    )
    return lowered


def _lower_tool_search_output(item: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_call_id(item)
    tools = item.get("tools")
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        raise ValueError("Client tool_search_output requires a tools array")
    lowered = {
        key: value for key, value in item.items() if key not in {"type", "execution", "status", "tools"}
    }
    lowered.update(
        type="function_call_output",
        output=json.dumps({"tools": tools}, separators=(",", ":"), ensure_ascii=False),
    )
    return lowered, tools


def _require_non_empty_name(value: Any, message: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(message)
    return value


def _canonical_definition(tool: dict[str, Any]) -> str:
    return json.dumps(tool, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_namespace_child_definition(child: dict[str, Any]) -> str:
    comparable = dict(child)
    comparable.setdefault("type", "function")
    return _canonical_definition(comparable)


def _collect_discovered_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize top-level and discovered model-tool candidates together."""
    normalized: list[dict[str, Any]] = []
    direct: dict[str, tuple[str, dict[str, Any]]] = {}
    namespaces: dict[str, tuple[dict[str, Any], dict[str, tuple[str, dict[str, Any]]]]] = {}
    other_seen: set[str] = set()

    for spec in specs:
        tool_type = spec.get("type")
        if tool_type == "function":
            name = _require_non_empty_name(
                spec.get("name"),
                "Function requires a non-empty name",
            )
            serialized = _canonical_definition(spec)
            previous = direct.get(name)
            if previous is not None:
                if previous[0] != serialized:
                    raise ValueError(f"Conflicting discovered function definition: {name}")
                continue
            copied = deepcopy(spec)
            direct[name] = (serialized, copied)
            normalized.append(copied)
        elif tool_type == "namespace":
            namespace = _require_non_empty_name(
                spec.get("name"), "Namespace requires a non-empty name"
            )
            children = spec.get("tools")
            if not isinstance(children, list) or not all(isinstance(child, dict) for child in children):
                raise ValueError("Namespace requires a tools array")
            if namespace not in namespaces:
                merged = deepcopy(spec)
                merged["tools"] = []
                children_by_name: dict[str, tuple[str, dict[str, Any]]] = {}
                namespaces[namespace] = (merged, children_by_name)
                normalized.append(merged)
            merged, children_by_name = namespaces[namespace]
            for child in children:
                child_name = _require_non_empty_name(
                    child.get("name"), "Namespace tool requires a non-empty name"
                )
                child_type = child.get("type", "function")
                if child_type != "function":
                    raise ValueError(f"Unsupported namespace tool type: {child_type}")
                serialized = _canonical_namespace_child_definition(child)
                previous = children_by_name.get(child_name)
                if previous is not None:
                    if previous[0] != serialized:
                        raise ValueError(
                            f"Conflicting namespace tool definition: {namespace}/{child_name}"
                        )
                    continue
                copied = deepcopy(child)
                children_by_name[child_name] = (serialized, copied)
                merged["tools"].append(copied)
        else:
            serialized = _canonical_definition(spec)
            if serialized not in other_seen:
                other_seen.add(serialized)
                normalized.append(deepcopy(spec))

    return normalized

def normalize_client_tool_search_history(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Lower client tool-search history and collect its discovered tool specs."""
    transformed = deepcopy(payload)
    input_items = transformed.get("input")
    if not isinstance(input_items, list):
        return transformed, []

    discovered: list[dict[str, Any]] = []
    lowered_items = []
    for item in input_items:
        if not isinstance(item, dict) or item.get("execution") != "client":
            lowered_items.append(item)
            continue
        if item.get("type") == "tool_search_call":
            lowered_items.append(_lower_tool_search_call(item))
        elif item.get("type") == "tool_search_output":
            lowered, tools = _lower_tool_search_output(item)
            lowered_items.append(lowered)
            discovered.extend(tools)
        else:
            lowered_items.append(item)
    transformed["input"] = lowered_items
    return transformed, discovered


def transform_request(payload: dict[str, Any]) -> TransformedRequest:
    """Build the llama-facing request and its request-local reverse map."""
    transformed = normalize_input_message_order(apply_reasoning_compat(payload))
    transformed, discovered_tools = normalize_client_tool_search_history(transformed)
    outgoing_tools = []
    reverse: dict[str, Any] = {}
    candidate_tools = _collect_discovered_tools(
        list(transformed.get("tools", [])) + discovered_tools
    )
    has_client_tool_search = any(
        isinstance(tool, dict)
        and tool.get("type") == "tool_search"
        and tool.get("execution") == "client"
        for tool in candidate_tools
    )
    reserved_aliases: set[str] = {
        str(tool["name"])
        for tool in candidate_tools
        if tool.get("type") == "function" and "name" in tool
    }
    if has_client_tool_search:
        if "tool_search" in reserved_aliases:
            raise ValueError("Ambiguous tool_search function name")
        reserved_aliases.add("tool_search")
    for tool in candidate_tools:
        if tool.get("type") in {"web_search", "file_search", "computer_use"}:
            continue
        if tool.get("type") == "namespace":
            flattened, local_reverse = map_namespace_tools(
                tool["name"], tool.get("tools", []), reserved_aliases
            )
            outgoing_tools.extend(flattened)
            for alias, identity in local_reverse.items():
                if alias in reverse and reverse[alias] != identity:
                    raise ValueError(f"Alias reserved for multiple tools: {alias}")
                reverse[alias] = identity
        elif tool.get("type") == "tool_search" and tool.get("execution") == "client":
            if "tool_search" in reverse:
                raise ValueError("Ambiguous tool_search function name")
            outgoing_tools.append(_lower_client_tool_search(tool))
            reverse["tool_search"] = ToolSearchIdentity()
        else:
            outgoing_tools.append(tool)
    if "tools" in transformed or discovered_tools:
        transformed["tools"] = outgoing_tools
    return TransformedRequest(transformed, reverse)


def _lift_tool_search_item(item: dict[str, Any]) -> dict[str, Any]:
    lifted = deepcopy(item)
    lifted["type"] = "tool_search_call"
    lifted.pop("name", None)
    lifted["execution"] = "client"
    arguments = lifted.get("arguments")
    if isinstance(arguments, str):
        try:
            lifted["arguments"] = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return lifted


def _lift_response_item(item: dict[str, Any], reverse: dict[str, Any]) -> dict[str, Any]:
    if item.get("type") != "function_call":
        return item
    identity = reverse.get(item.get("name"))
    if isinstance(identity, ToolSearchIdentity):
        return _lift_tool_search_item(item)
    if isinstance(identity, ToolIdentity):
        lifted = deepcopy(item)
        lifted["namespace"] = identity.namespace
        lifted["name"] = identity.tool_name
        return lifted
    return item


def transform_response(payload: dict[str, Any], reverse: dict[str, Any]) -> dict[str, Any]:
    """Restore mapped function calls in a non-stream response."""
    transformed = deepcopy(payload)
    for index, item in enumerate(transformed.get("output", [])):
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        transformed["output"][index] = _lift_response_item(item, reverse)
    return transformed


def _transform_sse_frame(frame: bytes, reverse: dict[str, Any]) -> bytes:
    separator = b"\r\n\r\n" if b"\r\n\r\n" in frame else b"\n\n"
    content = frame[:-len(separator)]
    lines = content.replace(b"\r\n", b"\n").split(b"\n")
    event_type = next(
        (line[6:].decode("utf-8", "replace").strip() for line in lines if line.startswith(b"event:")),
        None,
    )
    if event_type != "response.output_item.done":
        return frame
    data_lines = [line[5:].lstrip() for line in lines if line.startswith(b"data:")]
    if not data_lines or data_lines == [b"[DONE]"]:
        return frame
    try:
        data = json.loads(b"\n".join(data_lines))
    except json.JSONDecodeError:
        return frame
    if not isinstance(data, dict) or not isinstance(data.get("item"), dict):
        return frame
    lifted_item = _lift_response_item(data["item"], reverse)
    if lifted_item == data["item"]:
        return frame
    data["item"] = lifted_item
    encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    output_lines = []
    data_written = False
    for line in lines:
        if line.startswith(b"data:") and not data_written:
            output_lines.append(b"data: " + encoded)
            data_written = True
        elif not line.startswith(b"data:"):
            output_lines.append(line)
    return b"\n".join(output_lines) + separator


async def transform_sse_stream(chunks, reverse: dict[str, Any]):
    """Incrementally lift only mapped tool_search output_item.done SSE events."""
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while True:
            lf_position = buffer.find(b"\n\n")
            crlf_position = buffer.find(b"\r\n\r\n")
            positions = [position for position in (lf_position, crlf_position) if position >= 0]
            if not positions:
                break
            position = min(positions)
            separator_length = 4 if buffer[position:position + 4] == b"\r\n\r\n" else 2
            frame = buffer[:position + separator_length]
            buffer = buffer[position + separator_length:]
            yield _transform_sse_frame(frame, reverse)
    if buffer:
        yield buffer
