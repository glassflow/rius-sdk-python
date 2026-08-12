"""Configuration resolution for the GlassFlow SDK.

Values are resolved with the precedence: explicit arguments > environment
variables > built-in defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://ingest.eu.console.rius-glassflow.com"
DEFAULT_SERVICE_NAME = "unknown_service"

# RIUS_* is the canonical prefix; GLASSFLOW_* is read as a deprecated
# fallback (product rename). Only the env names changed: wire contracts
# like the tracer scope and glassflow.span.pending keep their names.
ENV_PREFIX = "RIUS_"
DEPRECATED_ENV_PREFIX = "GLASSFLOW_"

ENV_ENDPOINT = "RIUS_ENDPOINT"
ENV_API_KEY = "RIUS_API_KEY"
ENV_SERVICE_NAME = "RIUS_SERVICE_NAME"
ENV_DISABLED = "RIUS_DISABLED"
ENV_SAMPLE_RATE = "RIUS_SAMPLE_RATE"
ENV_CAPTURE_CONTENT = "RIUS_CAPTURE_CONTENT"
ENV_HEARTBEAT = "RIUS_HEARTBEAT"
ENV_HEARTBEAT_INTERVAL = "RIUS_HEARTBEAT_INTERVAL"
ENV_AGENT_NAME = "RIUS_AGENT_NAME"
ENV_PARTIAL_SPANS = "RIUS_PARTIAL_SPANS"
ENV_PARTIAL_SPANS_DELAY = "RIUS_PARTIAL_SPANS_DELAY"

# The backend expresses staleness as multiples of the interval, so the clamp
# bounds are part of the heartbeat wire contract.
HEARTBEAT_INTERVAL_MIN = 5.0
HEARTBEAT_INTERVAL_MAX = 300.0
DEFAULT_HEARTBEAT_INTERVAL = 15.0

# Debounce for partial spans: 0 = emit immediately at span start;
# N>0 = emit only if the span is still open after N seconds. Beyond 60s a
# "live" view stops being live, so larger values are clamped.
PARTIAL_SPANS_DELAY_MIN = 0.0
PARTIAL_SPANS_DELAY_MAX = 60.0

_TRUENESS = frozenset({"1", "true", "yes", "on"})


def _getenv(name: str, deprecated_used: list[str]) -> str | None:
    """Read ``name``, falling back to its deprecated ``GLASSFLOW_*`` spelling.

    A fallback hit is recorded in ``deprecated_used`` so ``resolve_config``
    can emit one consolidated deprecation warning instead of one per
    variable. When both spellings are set the ``RIUS_*`` one wins and no
    deprecation is recorded; that caller has already migrated.
    """
    value = os.getenv(name)
    if value is not None:
        return value
    suffix = name.removeprefix(ENV_PREFIX)
    value = os.getenv(DEPRECATED_ENV_PREFIX + suffix)
    if value is not None:
        deprecated_used.append(suffix)
    return value


def _env_bool(name: str, deprecated_used: list[str], *, default: bool) -> bool:
    raw = _getenv(name, deprecated_used)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUENESS


def _env_float(name: str, deprecated_used: list[str], *, default: float) -> float:
    raw = _getenv(name, deprecated_used)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class GlassflowConfig:
    """Resolved, immutable SDK configuration.

    Produced by ``resolve_config`` (arguments over environment over
    defaults); consumed by ``init`` and ``build_span_exporter``.
    """

    endpoint: str
    api_key: str | None
    service_name: str
    headers: dict[str, str] = field(default_factory=dict)
    disabled: bool = False
    sample_rate: float = 1.0
    capture_content: bool = True
    heartbeat: bool = False
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL
    agent_name: str = DEFAULT_SERVICE_NAME
    partial_spans: bool = False
    partial_spans_delay: float = 0.0

    @property
    def traces_endpoint(self) -> str:
        """Full OTLP/HTTP traces URL (``<endpoint>/v1/traces``)."""
        return self.endpoint.rstrip("/") + "/v1/traces"

    @property
    def heartbeat_endpoint(self) -> str:
        """Heartbeat URL (``<endpoint>/v1/heartbeat``), same host as traces."""
        return self.endpoint.rstrip("/") + "/v1/heartbeat"


def _clamp_sample_rate(value: float) -> float:
    """Clamp to [0.0, 1.0]; an out-of-range value must degrade, not crash init()."""
    if 0.0 <= value <= 1.0:
        return value
    clamped = min(max(value, 0.0), 1.0)
    logger.warning("sample_rate %s is outside [0.0, 1.0]; clamped to %s", value, clamped)
    return clamped


def _clamp_partial_spans_delay(value: float) -> float:
    """Clamp to [0, 60]; out-of-range degrades, never crashes init()."""
    if PARTIAL_SPANS_DELAY_MIN <= value <= PARTIAL_SPANS_DELAY_MAX:
        return value
    clamped = min(max(value, PARTIAL_SPANS_DELAY_MIN), PARTIAL_SPANS_DELAY_MAX)
    logger.warning(
        "partial_spans_delay %s is outside [%s, %s]; clamped to %s",
        value,
        PARTIAL_SPANS_DELAY_MIN,
        PARTIAL_SPANS_DELAY_MAX,
        clamped,
    )
    return clamped


def _clamp_heartbeat_interval(value: float) -> float:
    """Clamp to the contract bounds; out-of-range degrades, never crashes init()."""
    if HEARTBEAT_INTERVAL_MIN <= value <= HEARTBEAT_INTERVAL_MAX:
        return value
    clamped = min(max(value, HEARTBEAT_INTERVAL_MIN), HEARTBEAT_INTERVAL_MAX)
    logger.warning(
        "heartbeat_interval %s is outside [%s, %s]; clamped to %s",
        value,
        HEARTBEAT_INTERVAL_MIN,
        HEARTBEAT_INTERVAL_MAX,
        clamped,
    )
    return clamped


def resolve_config(
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    service_name: str | None = None,
    headers: dict[str, str] | None = None,
    disabled: bool | None = None,
    sample_rate: float | None = None,
    capture_content: bool | None = None,
    heartbeat: bool | None = None,
    heartbeat_interval: float | None = None,
    agent_name: str | None = None,
    partial_spans: bool | None = None,
    partial_spans_delay: float | None = None,
) -> GlassflowConfig:
    """Resolve SDK configuration from arguments, environment, then defaults.

    Explicit arguments win over ``RIUS_*`` environment variables, which win
    over their deprecated ``GLASSFLOW_*`` spellings, which win over built-in
    defaults. Using a ``GLASSFLOW_*`` variable logs one deprecation warning
    naming the replacements. ``sample_rate`` is clamped to ``[0.0, 1.0]``
    with a warning; boolean environment variables accept ``1``/``true``/
    ``yes``/``on`` (case-insensitive).

    Args:
        endpoint: Base OTLP endpoint (``RIUS_ENDPOINT``).
        api_key: Bearer token for the managed platform (``RIUS_API_KEY``);
            ``None`` sends no Authorization header.
        service_name: ``service.name`` resource attribute
            (``RIUS_SERVICE_NAME``).
        headers: Extra exporter headers; an explicit ``Authorization`` entry
            wins over ``api_key``.
        disabled: Kill switch (``RIUS_DISABLED``); spans are dropped
            in-process.
        sample_rate: Head-sampling ratio for root traces
            (``RIUS_SAMPLE_RATE``).
        capture_content: When ``False``, content attributes are stripped at
            export (``RIUS_CAPTURE_CONTENT``).
        heartbeat: Enable the agent-lifetime heartbeat thread
            (``RIUS_HEARTBEAT``). Off by default this release.
        heartbeat_interval: Seconds between pings
            (``RIUS_HEARTBEAT_INTERVAL``), clamped to ``[5, 300]``;
            the backend derives staleness from this, so the bounds are part
            of the wire contract.
        agent_name: Identity heartbeats group under (``RIUS_AGENT_NAME``);
            defaults to ``service_name`` so the agents view and the traces
            view agree on what an "agent" is.
        partial_spans: Export a content-free pending snapshot of every
            sampled span at span START (``RIUS_PARTIAL_SPANS``), so
            in-flight work is visible and crashes leave a record. Off by
            default until the backend's unfinished-spans storage ships.
        partial_spans_delay: Debounce for pending snapshots
            (``RIUS_PARTIAL_SPANS_DELAY``), clamped to ``[0, 60]``
            seconds. ``0`` (default) emits at span start; ``N`` emits only if
            the span is still open after N seconds; spans that finish
            sooner cost no network at all.

    Returns:
        The resolved, immutable ``GlassflowConfig``.
    """
    deprecated_used: list[str] = []
    resolved_endpoint = endpoint or _getenv(ENV_ENDPOINT, deprecated_used) or DEFAULT_ENDPOINT
    resolved_api_key = api_key if api_key is not None else _getenv(ENV_API_KEY, deprecated_used)
    resolved_service_name = (
        service_name or _getenv(ENV_SERVICE_NAME, deprecated_used) or DEFAULT_SERVICE_NAME
    )
    resolved_disabled = (
        _env_bool(ENV_DISABLED, deprecated_used, default=False) if disabled is None else disabled
    )
    resolved_sample_rate = _clamp_sample_rate(
        _env_float(ENV_SAMPLE_RATE, deprecated_used, default=1.0)
        if sample_rate is None
        else sample_rate
    )
    resolved_capture_content = (
        _env_bool(ENV_CAPTURE_CONTENT, deprecated_used, default=True)
        if capture_content is None
        else capture_content
    )

    resolved_heartbeat = (
        _env_bool(ENV_HEARTBEAT, deprecated_used, default=False) if heartbeat is None else heartbeat
    )
    resolved_heartbeat_interval = _clamp_heartbeat_interval(
        _env_float(ENV_HEARTBEAT_INTERVAL, deprecated_used, default=DEFAULT_HEARTBEAT_INTERVAL)
        if heartbeat_interval is None
        else heartbeat_interval
    )
    resolved_agent_name = (
        agent_name or _getenv(ENV_AGENT_NAME, deprecated_used) or resolved_service_name
    )
    resolved_partial_spans = (
        _env_bool(ENV_PARTIAL_SPANS, deprecated_used, default=False)
        if partial_spans is None
        else partial_spans
    )
    resolved_partial_spans_delay = _clamp_partial_spans_delay(
        _env_float(ENV_PARTIAL_SPANS_DELAY, deprecated_used, default=0.0)
        if partial_spans_delay is None
        else partial_spans_delay
    )

    if deprecated_used:
        renames = ", ".join(
            f"{DEPRECATED_ENV_PREFIX}{s} -> {ENV_PREFIX}{s}" for s in deprecated_used
        )
        logger.warning(
            "deprecated GLASSFLOW_-prefixed environment variable(s) in use: %s. "
            "They keep working for now and will be removed in a future release; "
            "rename them to the RIUS_ prefix.",
            renames,
        )

    resolved_headers = dict(headers or {})
    has_auth = any(key.lower() == "authorization" for key in resolved_headers)
    if resolved_api_key and not has_auth:
        resolved_headers["Authorization"] = f"Bearer {resolved_api_key}"

    return GlassflowConfig(
        endpoint=resolved_endpoint,
        api_key=resolved_api_key,
        service_name=resolved_service_name,
        headers=resolved_headers,
        disabled=resolved_disabled,
        sample_rate=resolved_sample_rate,
        capture_content=resolved_capture_content,
        heartbeat=resolved_heartbeat,
        heartbeat_interval=resolved_heartbeat_interval,
        agent_name=resolved_agent_name,
        partial_spans=resolved_partial_spans,
        partial_spans_delay=resolved_partial_spans_delay,
    )
