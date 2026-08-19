"""Stateless MCP namespace-to-function name mapping."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import re

_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_MAX_NAME_LENGTH = 64


@dataclass(frozen=True)
class ToolIdentity:
    namespace: str
    tool_name: str


def _digest(namespace: str, tool_name: str) -> str:
    return hashlib.sha256(f"{namespace}\0{tool_name}".encode()).hexdigest()[:10]


@lru_cache(maxsize=1024)
def flat_name(namespace: str, tool_name: str) -> str:
    """Return a deterministic backend-safe name for an MCP tool."""
    readable = f"{namespace}__{tool_name}"
    if _SAFE_NAME.fullmatch(readable) and len(readable) <= _MAX_NAME_LENGTH:
        return readable
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", readable).strip("_") or "tool"
    suffix = _digest(namespace, tool_name)
    prefix = normalized[: _MAX_NAME_LENGTH - len(suffix) - 2].rstrip("_")
    return f"{prefix}__{suffix}"


def map_namespace_tools(
    namespace: str,
    tools: list[dict],
    reserved_aliases: set[str] | None = None,
) -> tuple[list[dict], dict[str, ToolIdentity]]:
    """Flatten tools from one namespace and return its request-local reverse map."""
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("Namespace requires a non-empty name")
    flattened = []
    reverse: dict[str, ToolIdentity] = {}
    reserved = reserved_aliases if reserved_aliases is not None else set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("Namespace tool must be an object")
        tool_type = tool.get("type", "function")
        if tool_type != "function":
            raise ValueError(f"Unsupported namespace tool type: {tool_type}")
        original = tool.get("name")
        if not isinstance(original, str) or not original:
            raise ValueError("Namespace tool requires a non-empty name")
        base_alias = flat_name(namespace, original)
        alias = base_alias
        attempt = 0
        while alias in reserved:
            attempt += 1
            suffix = _digest(namespace, f"{original}\0{attempt}")
            prefix = base_alias[: _MAX_NAME_LENGTH - len(suffix) - 2].rstrip("_")
            alias = f"{prefix}__{suffix}"
        reserved.add(alias)
        flattened_tool = {key: value for key, value in tool.items() if key != "name"}
        flattened_tool["type"] = "function"
        flattened_tool["name"] = alias
        flattened.append(flattened_tool)
        identity = ToolIdentity(namespace, original)
        if alias in reverse and reverse[alias] != identity:
            raise ValueError(f"Alias reserved for multiple tools: {alias}")
        reverse[alias] = identity
    return flattened, reverse
