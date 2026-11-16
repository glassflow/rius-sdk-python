"""Tool definitions in the GenAI conventions' shape (``gen_ai.tool.definitions``).

The conventions define one tool type, ``function``:
``{"type": "function", "name", "description"?, "parameters"?}``, and a
definition whose ``type`` is ``function`` must match that shape: it can no
longer pass as the generic definition. Providers spell function tools their
own way, so each is rewritten:

* OpenAI Chat Completions nests the function, ``{"type": "function",
  "function": {"name", ...}}``: the nested object is lifted to the top level,
  unless the top level already has a name (then the tool is already flat,
  and ``function`` is one of its fields).
* Chat Completions nests a custom tool the same way, ``{"type": "custom",
  "custom": {"name", "description", "format"}}``: the nested object is lifted
  by the same rule, and the tool stays a custom tool (a provider tool, below),
  unless the lifted tool is Anthropic's custom tool.
* Anthropic has no ``type`` (or ``"custom"``) and names the schema
  ``input_schema``; OpenAI's legacy ``functions`` entries have no ``type``
  either. A definition with a name and no ``type`` is a function, and so is
  a ``custom`` one with an ``input_schema``.
* A name is a non-empty string. An empty ``name`` is no name, as it is for a
  tool call: no function tool is made with an empty name.
* The schema is ``parameters``; ``input_schema`` and the Vercel AI SDK's
  ``inputSchema`` are renamed to it, in place.
* Bedrock Converse wraps the function, ``{"toolSpec": {"name", "description",
  "inputSchema": {"json": schema}}}``: it is unwrapped, and ``json`` becomes
  ``parameters``.
* A Gemini tool holds a list of declarations, ``{"function_declarations":
  [...]}`` (``functionDeclarations`` in camel case; every spelling holding a
  list expands, snake case first). Each declaration with a name
  becomes its own function tool, its schema taken from ``parameters`` or,
  failing that, ``parametersJsonSchema`` / ``parameters_json_schema`` /
  ``input_schema`` / ``inputSchema``. A declaration without a name, and
  whatever else the Gemini tool holds (a built-in such as ``google_search``),
  is its own entry, rewritten by the rules for a single tool.
* A ``null`` ``parameters`` counts as no schema, so a schema under another
  spelling still takes its place (a Python ``model_dump()`` of a Gemini
  declaration holds both).

Any other ``type`` is a provider's built-in tool (Anthropic's
``web_search_20250305``, OpenAI's ``web_search``). Its fields are kept, and its
``type`` is prefixed with the provider name, as the conventions recommend, so
it cannot collide with a type they define later: ``anthropic.web_search_20250305``.
An empty ``type`` is no type, and gets no prefix.
A type that already holds a dot is taken as namespaced and left alone, which
also keeps a second pass from stacking prefixes. Nothing is invented: a
built-in tool without a ``name`` (OpenAI's ``web_search``, Gemini's
``google_search``) stays without one. Tool names are read as the tools the
model was offered, and an invented name would be a tool offered and never
called by that name.

An MCP tool (OpenAI Responses' ``type: "mcp"``, or any ``<provider>.mcp``
type it is written as) loses its ``headers`` and its ``authorization``: the
headers are the requests' headers to the MCP server and can carry an
``Authorization`` credential, and ``authorization`` is an OAuth access token
for the server or connector. The Vercel AI SDK's ``openai.mcp`` provider tool
holds the same two members in its ``args``, and they go there too.
Credentials never belong in telemetry. This
applies to the definitions rewritten here, not to definitions another
instrumentation already wrote as ``gen_ai.tool.definitions``.

Anything else (not an object, or no usable ``type`` or ``name``) is kept as
it is: it is the caller's data, and there is no shape to vouch for.

Every rule's output is a fixed point: rewriting a rewritten list changes
nothing. The sink rewrites what an SDK exported, so this keeps one tool in one
shape.

The TypeScript SDK's ``toolDefinitions.ts`` and the sink port this rule for
rule; all three write the same canonical JSON, key order included, and
rius-semconv's ``sdk_cases/tool_definitions.json`` holds them to it. Number
formatting is each language's own (Python writes ``1.0``, TypeScript ``1``),
and so is the order of integer-like keys in TypeScript.
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
_CUSTOM = "custom"
#: OpenAI Responses' MCP tool type; ``<provider>.mcp`` once namespaced.
_MCP = "mcp"
#: An MCP tool's members that hold credentials, dropped.
_MCP_CREDENTIALS = ("headers", "authorization")


def _named(value: Any) -> bool:
    """A usable tool name: a non-empty string."""
    return isinstance(value, str) and bool(value)


def semconv_tool_definitions(tools: list[Any], provider: str | None) -> list[Any]:
    """The definitions in the conventions' shape; the caller's list is not mutated.

    A Gemini tool expands to one entry per declaration, so the result can be
    longer than ``tools``.
    """
    definitions: list[Any] = []
    for tool in tools:
        expanded = _gemini_declarations(tool, provider)
        if expanded is None:
            definitions.append(_semconv_tool_definition(tool, provider))
        else:
            definitions.extend(expanded)
    return definitions


def _gemini_declarations(tool: Any, provider: str | None) -> list[Any] | None:
    """A Gemini tool as one entry per declaration, then the rest; None if not one."""
    if not isinstance(tool, dict) or tool.get("type") is not None:
        return None
    keys = [key for key in _GEMINI_DECLARATIONS if isinstance(tool.get(key), list)]
    if not keys:
        return None
    entries: list[Any] = []
    for key in keys:
        for declaration in tool[key]:
            if isinstance(declaration, dict) and _named(declaration.get("name")):
                entries.append(_gemini_declaration(declaration))
            else:
                entries.extend(semconv_tool_definitions([declaration], provider))
    rest = {name: value for name, value in tool.items() if name not in keys}
    if rest:
        entries.extend(semconv_tool_definitions([rest], provider))
    return entries


def _gemini_declaration(declaration: dict[str, Any]) -> dict[str, Any]:
    flat = {"type": _FUNCTION, **{k: v for k, v in declaration.items() if k != "type"}}
    return _with_parameters(flat, _GEMINI_PARAMETER_SPELLINGS + _PARAMETER_SPELLINGS)


def _semconv_tool_definition(tool: Any, provider: str | None) -> Any:
    if not isinstance(tool, dict):
        return tool
    kind = tool.get("type")
    if kind == _FUNCTION:
        nested = tool.get(_FUNCTION)
        if not isinstance(nested, dict) or _named(tool.get("name")):
            return _with_parameters(tool)
        return _with_parameters(_lifted(tool, _FUNCTION, nested))
    if kind == _CUSTOM:
        nested = tool.get(_CUSTOM)
        if isinstance(nested, dict) and not _named(tool.get("name")):
            return _semconv_tool_definition(_lifted(tool, _CUSTOM, nested), provider)
    if _named(tool.get("name")) and (kind is None or (kind == _CUSTOM and "input_schema" in tool)):
        rest = {key: value for key, value in tool.items() if key != "type"}
        return _with_parameters({"type": _FUNCTION, **rest})
    spec = tool.get("toolSpec")
    if kind is None and isinstance(spec, dict) and _named(spec.get("name")):
        return _bedrock_tool(tool, spec)
    if not isinstance(kind, str) or not kind:
        return tool
    if kind in _VERCEL_PROVIDER_TYPES:
        identifier = tool.get("id")
        if isinstance(identifier, str) and "." in identifier:
            tool = {**tool, "type": identifier}
    elif provider and "." not in kind:
        tool = {**tool, "type": f"{provider}.{kind}"}
    return _without_credentials(tool)


def _lifted(tool: dict[str, Any], kind: str, nested: dict[str, Any]) -> dict[str, Any]:
    """The tool with its nested ``kind`` object's members at the top level.

    The nested members win; the top level's other members follow.
    """
    flat: dict[str, Any] = {"type": kind}
    flat.update((key, value) for key, value in nested.items() if key != "type")
    flat.update(
        (key, value) for key, value in tool.items() if key not in ("type", kind) and key not in flat
    )
    return flat


def _without_credentials(tool: dict[str, Any]) -> dict[str, Any]:
    """An MCP tool (``mcp`` or ``<provider>.mcp``) without its credential members.

    Dropped at the top level and in a Vercel provider tool's ``args``. Read
    from the type the tool is written with, so a second pass drops the same
    members.
    """
    kind = tool.get("type")
    if not isinstance(kind, str) or not (kind == _MCP or kind.endswith("." + _MCP)):
        return tool
    args = tool.get("args")
    if isinstance(args, dict) and any(key in args for key in _MCP_CREDENTIALS):
        tool = {**tool, "args": _without_members(args, _MCP_CREDENTIALS)}
    if not any(key in tool for key in _MCP_CREDENTIALS):
        return tool
    return _without_members(tool, _MCP_CREDENTIALS)


def _without_members(members: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: value for key, value in members.items() if key not in keys}


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
    """The tool with its schema under ``parameters``, renamed where it stood.

    A ``null`` ``parameters`` is no schema: when another spelling holds one,
    the null goes and that spelling is renamed.
    """
    if tool.get("parameters") is not None:
        return tool
    for spelling in spellings:
        if spelling in tool:
            return {
                ("parameters" if key == spelling else key): value
                for key, value in tool.items()
                if key != "parameters"
            }
    return tool
