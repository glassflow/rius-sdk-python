"""Tool definitions in the GenAI conventions' shape (``gen_ai.tool.definitions``).

The conventions define one tool type, ``function``:
``{"type": "function", "name", "description"?, "parameters"?}``, and a
definition whose ``type`` is ``function`` must match that shape: it can no
longer pass as the generic definition. Providers spell function tools their
own way, so each is rewritten:

* OpenAI Chat Completions nests the function, ``{"type": "function",
  "function": {"name", ...}}``: the nested object is lifted to the top level.
* Anthropic has no ``type`` (or ``"custom"``) and names the schema
  ``input_schema``; OpenAI's legacy ``functions`` entries have no ``type``
  either. A definition with a string ``name`` and no ``type`` is a function.
* The schema is ``parameters``; ``input_schema`` and the Vercel AI SDK's
  ``inputSchema`` are renamed to it, in place.

Any other ``type`` is a provider's built-in tool (Anthropic's
``web_search_20250305``, OpenAI's ``web_search``). Its fields are kept, and its
``type`` is prefixed with the provider name, as the conventions recommend, so
it cannot collide with a type they define later: ``anthropic.web_search_20250305``.
A type that already holds a dot is taken as namespaced and left alone, which
also keeps a second pass from stacking prefixes. Nothing is invented: a
built-in tool without a ``name`` stays without one.

Anything else (not an object, or no usable ``type`` or ``name``) is kept as
it is: it is the caller's data, and there is no shape to vouch for.

The TypeScript SDK's ``toolDefinitions.ts`` ports this rule for rule; both
must write the same JSON, key order included.
"""

from __future__ import annotations

from typing import Any

_FUNCTION = "function"
#: Schema spellings renamed to ``parameters``, first match wins.
_PARAMETER_SPELLINGS = ("input_schema", "inputSchema")
#: The Vercel AI SDK's provider tools, whose ``id`` is already ``<provider>.<tool>``.
_VERCEL_PROVIDER_TYPES = ("provider-defined", "provider")


def semconv_tool_definitions(tools: list[Any], provider: str | None) -> list[Any]:
    """The definitions in the conventions' shape; the caller's list is not mutated."""
    return [_semconv_tool_definition(tool, provider) for tool in tools]


def _semconv_tool_definition(tool: Any, provider: str | None) -> Any:
    if not isinstance(tool, dict):
        return tool
    kind = tool.get("type")
    if kind == _FUNCTION:
        nested = tool.get(_FUNCTION)
        if not isinstance(nested, dict):
            return _with_parameters(tool)
        flat: dict[str, Any] = {"type": _FUNCTION}
        flat.update((key, value) for key, value in nested.items() if key != "type")
        flat.update(
            (key, value)
            for key, value in tool.items()
            if key not in ("type", _FUNCTION) and key not in flat
        )
        return _with_parameters(flat)
    if isinstance(tool.get("name"), str) and (
        kind is None or (kind == "custom" and "input_schema" in tool)
    ):
        rest = {key: value for key, value in tool.items() if key != "type"}
        return _with_parameters({"type": _FUNCTION, **rest})
    if not isinstance(kind, str):
        return tool
    if kind in _VERCEL_PROVIDER_TYPES:
        identifier = tool.get("id")
        if isinstance(identifier, str) and "." in identifier:
            return {**tool, "type": identifier}
        return tool
    if provider and "." not in kind:
        return {**tool, "type": f"{provider}.{kind}"}
    return tool


def _with_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    """The tool with its schema under ``parameters``, renamed where it stood."""
    if "parameters" in tool:
        return tool
    for spelling in _PARAMETER_SPELLINGS:
        if spelling in tool:
            return {
                ("parameters" if key == spelling else key): value for key, value in tool.items()
            }
    return tool
