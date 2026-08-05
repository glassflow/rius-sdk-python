"""Bundled auto-instrumentation via OpenInference/OpenLLMetry.

We reuse existing OTel instrumentors rather than rebuilding provider/framework
instrumentation. The registry below maps a short name to an instrumentor class;
packages are imported lazily, so nothing here adds a hard dependency. Install
via extras (``pip install glassflow-rius[openai]``) and ``init()`` enables what
it finds, passing our tracer provider so instrumentation spans nest under ours.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Instrumentors we enabled (name -> instance), so a legitimate re-init can
# re-bind them to the new provider without touching instrumentation someone
# else set up.
_ENABLED: dict[str, Any] = {}


@dataclass(frozen=True)
class InstrumentorSpec:
    """A third-party instrumentor we know how to enable."""

    name: str
    module: str
    class_name: str


# OpenInference instrumentors (Arize) — same conventions we emit natively.
# OpenLLMetry entries can be added alongside; the backend normalizer covers both.
REGISTRY: tuple[InstrumentorSpec, ...] = (
    InstrumentorSpec("openai", "openinference.instrumentation.openai", "OpenAIInstrumentor"),
    InstrumentorSpec(
        "anthropic", "openinference.instrumentation.anthropic", "AnthropicInstrumentor"
    ),
    InstrumentorSpec(
        "langchain", "openinference.instrumentation.langchain", "LangChainInstrumentor"
    ),
    InstrumentorSpec(
        "llama-index", "openinference.instrumentation.llama_index", "LlamaIndexInstrumentor"
    ),
    InstrumentorSpec("litellm", "openinference.instrumentation.litellm", "LiteLLMInstrumentor"),
    # ours — first-class MCP tool-call spans (see instrumentation_mcp.py)
    InstrumentorSpec("mcp", "rius.instrumentation_mcp", "MCPInstrumentor"),
)


def enable_instrumentations(
    tracer_provider: Any,
    instruments: Sequence[str] | None = None,
) -> list[str]:
    """Enable bundled instrumentors against ``tracer_provider``.

    ``instruments=None`` enables every registry entry whose package is
    installed; an explicit list restricts to those names (warning if one is
    unknown or not installed). Returns the names actually enabled.
    """
    known = {spec.name for spec in REGISTRY}
    if instruments is not None:
        for name in instruments:
            if name not in known:
                logger.warning(
                    "unknown instrument %r; known instruments: %s",
                    name,
                    ", ".join(sorted(known)),
                )

    enabled: list[str] = []
    for spec in REGISTRY:
        requested = instruments is None or spec.name in instruments
        if not requested:
            continue
        try:
            module = importlib.import_module(spec.module)
        except ImportError:
            if instruments is not None:
                logger.warning(
                    "instrument %r requested but %r is not installed; "
                    'install it via `pip install "glassflow-rius[%s]"`',
                    spec.name,
                    spec.module,
                    spec.name,
                )
            continue
        try:
            instrumentor = getattr(module, spec.class_name)()
            if getattr(instrumentor, "is_instrumented_by_opentelemetry", False):
                if spec.name in _ENABLED:
                    # We enabled it previously (e.g. before a shutdown/re-init):
                    # re-bind it to the new provider.
                    instrumentor.uninstrument()
                else:
                    continue  # someone else's instrumentation: leave it alone
            instrumentor.instrument(tracer_provider=tracer_provider)
        except Exception:
            # Instrumentation is best-effort: a broken instrumentor must not
            # take down init().
            logger.warning("failed to enable instrument %r", spec.name, exc_info=True)
            continue
        _ENABLED[spec.name] = instrumentor
        enabled.append(spec.name)
    return enabled
