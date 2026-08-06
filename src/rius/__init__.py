"""GlassFlow SDK: OpenTelemetry-native tracing for AI agents and LLM apps."""

__version__ = "0.8.1"  # x-release-please-version

from .client import GlassflowClient, build_span_exporter, get_tracer, init
from .config import GlassflowConfig, resolve_config
from .generation import Generation, start_as_current_generation, start_generation
from .observe import observe
from .semconv import SpanKind
from .spans import Observation, start_as_current_span, start_span

__all__ = [
    "Generation",
    "GlassflowClient",
    "GlassflowConfig",
    "Observation",
    "SpanKind",
    "__version__",
    "build_span_exporter",
    "get_tracer",
    "init",
    "observe",
    "resolve_config",
    "start_as_current_generation",
    "start_as_current_span",
    "start_generation",
    "start_span",
]
