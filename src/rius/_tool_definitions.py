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
* Bedrock Converse wraps the function, ``{"toolSpec": {"name", "description",
  "inputSchema": {"json": schema}}}``: it is unwrapped, and ``json`` becomes
  ``parameters``.
* A Gemini tool holds a list of declarations, ``{"function_declarations":
  [...]}`` (``functionDeclarations`` in camel case). Each declaration with a
  string ``name`` becomes its own function tool, its schema taken from
  ``parameters`` or, failing that, ``parametersJsonSchema`` /
  ``parameters_json_schema``. A declaration without a name, and whatever else
  the Gemini tool holds (a built-in such as ``google_search``), is kept as
  its own entry.

Any other ``type`` is a provider's built-in tool (Anthropic's
``web_search_20250305``, OpenAI's ``web_search``). Its fields are kept, and its
``type`` is prefixed with the provider name, as the conventions recommend, so
it cannot collide with a type they define later: ``anthropic.web_search_20250305``.
A type that already holds a dot is taken as namespaced and left alone, which
also keeps a second pass from stacking prefixes. Nothing is invented: a
built-in tool without a ``name`` (OpenAI's ``web_search``, Gemini's
``google_search``) stays without one. Tool names are read as the tools the
model was offered, and an invented name would be a tool offered and never
called by that name.

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
#: Gemini's own schema spellings, renamed to ``parameters`` on a declaration.
_GEMINI_PARAMETER_SPELLINGS = ("parametersJsonSchema", "parameters_json_schema")
#: A Gemini tool's list of function declarations, snake and camel case.
_GEMINI_DECLARATIONS = ("function_declarations", "functionDeclarations")
#: The Vercel AI SDK's provider tools, whose ``id`` is already ``<provider>.<tool>``.
_VERCEL_PROVIDER_TYPES = ("provider-defined", "provider")


def semconv_tool_definitions(tools: list[Any], provider: str | None) -> list[Any]:
    """The definitions in the conventions' shape; the caller's list is not mutated.

    A Gemini tool expands to one entry per declaration, so the result can be
    longer than ``tools``.
    """
    definitions: list[Any] = []
    for tool in tools:
        expanded = _gemini_declarations(tool)
        if expanded is None:
            definitions.append(_semconv_tool_definition(tool, provider))
        else:
            definitions.extend(expanded)
    return definitions


def _gemini_declarations(tool: Any) -> list[Any] | None:
    """A Gemini tool as one entry per declaration, then the rest; None if not one."""
    if not isinstance(tool, dict) or tool.get("type") is not None:
        return None
    key = next((key for key in _GEMINI_DECLARATIONS if key in tool), None)
    if key is None or not isinstance(tool[key], list):
        return None
    entries = [_gemini_declaration(declaration) for declaration in tool[key]]
    rest = {name: value for name, value in tool.items() if name != key}
    if rest:
        entries.append(rest)
    return entries


def _gemini_declaration(declaration: Any) -> Any:
    if not isinstance(declaration, dict) or not isinstance(declaration.get("name"), str):
        return declaration
    flat = {"type": _FUNCTION, **{k: v for k, v in declaration.items() if k != "type"}}
    return _with_parameters(flat, _GEMINI_PARAMETER_SPELLINGS)


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
    spec = tool.get("toolSpec")
    if kind is None and isinstance(spec, dict) and isinstance(spec.get("name"), str):
        return _bedrock_tool(tool, spec)
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


def _bedrock_tool(tool: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Bedrock's ``toolSpec`` lifted to a function tool, ``inputSchema.json`` as parameters."""
    flat: dict[str, Any] = {"type": _FUNCTION}
    for key, value in spec.items():
        if key == "type":
            continue
        if key == "inputSchema" and isinstance(value, dict) and "json" in value:
            flat["parameters"] = value["json"]
        else:
            flat[key] = value
    flat.update(
        (key, value)
        for key, value in tool.items()
        if key not in ("type", "toolSpec") and key not in flat
    )
    return _with_parameters(flat)


def _with_parameters(
    tool: dict[str, Any], spellings: tuple[str, ...] = _PARAMETER_SPELLINGS
) -> dict[str, Any]:
    """The tool with its schema under ``parameters``, renamed where it stood."""
    if "parameters" in tool:
        return tool
    for spelling in spellings:
        if spelling in tool:
            return {
                ("parameters" if key == spelling else key): value for key, value in tool.items()
            }
    return tool
