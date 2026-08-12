"""SDK entrypoint: configure OpenTelemetry and export GenAI traces via OTLP."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from . import __version__
from .config import DEFAULT_ENDPOINT, GlassflowConfig, resolve_config
from .export_health import (
    ExportOutcomeExporter,
    ProbeTransport,
    _default_probe_send,
    check_connectivity,
)
from .heartbeat import HeartbeatSender, OpenRootSpanTracker
from .instrumentation import enable_instrumentations
from .masking import MaskingSpanExporter
from .pending import PendingSpanProcessor
from .semconv import TRACER_NAME

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_current_client: GlassflowClient | None = None


def build_span_exporter(config: GlassflowConfig) -> SpanExporter:
    """Build the default OTLP/HTTP span exporter for a resolved config.

    Args:
        config: A resolved configuration; the exporter posts to
            ``config.traces_endpoint`` with ``config.headers``.

    Returns:
        A ready-to-use OTLP/HTTP ``SpanExporter``.
    """
    return OTLPSpanExporter(
        endpoint=config.traces_endpoint,
        headers=config.headers or None,
    )


def _missing_managed_credentials(config: GlassflowConfig) -> bool:
    """True when the default exporter would hit the managed platform with no auth.

    Only the managed endpoint warrants a warning: a custom endpoint without
    credentials is a legitimate own-collector setup.
    """
    if config.endpoint != DEFAULT_ENDPOINT:
        return False
    return not any(key.lower() == "authorization" for key in (config.headers or {}))


class GlassflowClient:
    """Handle over a configured tracer provider.

    Returned by ``init``. Exposes the lifecycle operations (``flush``,
    ``shutdown``) and tracer access for the pipeline it owns; the resolved
    configuration is available as ``client.config``.
    """

    def __init__(
        self,
        provider: TracerProvider,
        config: GlassflowConfig,
        heartbeat: HeartbeatSender | None = None,
        export_health: ExportOutcomeExporter | None = None,
        connectivity_thread: threading.Thread | None = None,
    ) -> None:
        self._provider = provider
        self.config = config
        self._heartbeat = heartbeat
        self._export_health = export_health
        self._connectivity_thread = connectivity_thread
        self._is_shutdown = False

    def get_tracer(self, name: str = TRACER_NAME) -> trace.Tracer:
        """Return a tracer bound to this client's provider.

        Args:
            name: Instrumentation scope name; defaults to the SDK's own.
        """
        return self._provider.get_tracer(name, __version__)

    def flush(self, timeout_millis: int = 30_000) -> bool:
        """Force-flush pending spans and report delivery.

        Returns True only when the queue drained within ``timeout_millis``
        AND the most recent export attempt succeeded. Earlier releases
        reported queue drain alone, so it returned True even while every
        batch was being rejected (e.g. 401 on a bad API key). A False return
        therefore means either a flush timeout or that spans are currently
        not being delivered; the log carries the distinction.
        """
        drained = self._provider.force_flush(timeout_millis)
        if self._export_health is not None and self._export_health.last_export_failed:
            return False
        return drained

    def shutdown(self) -> None:
        """Drain pending spans and stop. Releases the global init() slot.

        Also stops the heartbeat thread and sends its final ``stopped`` ping,
        so the backend can tell a clean shutdown from a vanished agent.
        """
        global _current_client
        if self._heartbeat is not None:
            self._heartbeat.stop()
        self._provider.shutdown()
        self._is_shutdown = True
        with _lock:
            if _current_client is self:
                _current_client = None


def init(
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    service_name: str | None = None,
    headers: dict[str, str] | None = None,
    disabled: bool | None = None,
    sample_rate: float | None = None,
    capture_content: bool | None = None,
    mask: Callable[[Any], Any] | None = None,
    instruments: Sequence[str] | None = None,
    span_exporter: SpanExporter | None = None,
    heartbeat: bool | None = None,
    heartbeat_interval: float | None = None,
    agent_name: str | None = None,
    heartbeat_transport: Callable[[dict[str, Any]], None] | None = None,
    connectivity_transport: ProbeTransport | None = None,
    partial_spans: bool | None = None,
    partial_spans_delay: float | None = None,
    set_global: bool = True,
) -> GlassflowClient:
    """Initialize the SDK: build a tracer provider that exports OTLP traces.

    Calling ``init()`` again while a global client is active logs a warning and
    returns the existing client unchanged (the OpenTelemetry global tracer
    provider is write-once); call ``shutdown()`` on it first to reconfigure.

    Args:
        endpoint: Base OTLP endpoint. Traces are sent to ``<endpoint>/v1/traces``.
        api_key: API key; injected as an ``Authorization: Bearer`` header.
        service_name: Value for the ``service.name`` resource attribute.
        headers: Extra headers for the OTLP exporter.
        disabled: If True, no exporter is attached (spans are dropped).
        sample_rate: Head sampling ratio 0.0-1.0 (whole-trace). Default 1.0.
        capture_content: If False, strip prompt/response content at export. Default True.
        mask: Redact content attribute values at export (applies to all spans).
        instruments: Auto-instrumentation selection. ``None`` (default) enables
            every bundled instrumentor whose package is installed; a list
            restricts to those names; ``[]`` disables auto-instrumentation.
            Instrumentors are process-global, so with ``set_global=False`` they
            are only enabled when ``instruments`` is passed explicitly.
        span_exporter: Override the default OTLP exporter (useful for testing).
        heartbeat: Enable the agent-lifetime heartbeat thread
            (``GLASSFLOW_HEARTBEAT``; default off this release). Pings
            ``<endpoint>/v1/heartbeat`` from init until process exit so the
            platform can tell a live-but-idle agent from a vanished one.
        heartbeat_interval: Seconds between pings (default 15, clamped to
            ``[5, 300]``; the backend derives staleness from this).
        agent_name: Identity heartbeats group under; defaults to
            ``service_name``.
        heartbeat_transport: Override the heartbeat HTTP transport
            (useful for testing, like ``span_exporter``).
        connectivity_transport: Override the HTTP send used by the one-shot
            background connectivity check (useful for testing). The check
            POSTs an empty OTLP request at init and logs an actionable
            warning on 401/403, unreachable host, or other non-2xx, so a
            bad key or endpoint is visible immediately instead of surfacing
            as silently missing traces.
        set_global: Register the provider as the global OpenTelemetry provider.
    """
    global _current_client
    with _lock:
        if set_global and _current_client is not None and not _current_client._is_shutdown:
            logger.warning(
                "rius.init() was already called; keeping the existing configuration "
                "(the OpenTelemetry global tracer provider is write-once). Call .shutdown() "
                "on the existing client first if you need to reconfigure."
            )
            return _current_client
        return _do_init(
            endpoint=endpoint,
            api_key=api_key,
            service_name=service_name,
            headers=headers,
            disabled=disabled,
            sample_rate=sample_rate,
            capture_content=capture_content,
            mask=mask,
            instruments=instruments,
            span_exporter=span_exporter,
            heartbeat=heartbeat,
            heartbeat_interval=heartbeat_interval,
            agent_name=agent_name,
            heartbeat_transport=heartbeat_transport,
            connectivity_transport=connectivity_transport,
            partial_spans=partial_spans,
            partial_spans_delay=partial_spans_delay,
            set_global=set_global,
        )


def _do_init(
    *,
    endpoint: str | None,
    api_key: str | None,
    service_name: str | None,
    headers: dict[str, str] | None,
    disabled: bool | None,
    sample_rate: float | None,
    capture_content: bool | None,
    mask: Callable[[Any], Any] | None,
    instruments: Sequence[str] | None,
    span_exporter: SpanExporter | None,
    heartbeat: bool | None,
    heartbeat_interval: float | None,
    agent_name: str | None,
    heartbeat_transport: Callable[[dict[str, Any]], None] | None,
    connectivity_transport: ProbeTransport | None,
    partial_spans: bool | None,
    partial_spans_delay: float | None,
    set_global: bool,
) -> GlassflowClient:
    global _current_client
    config = resolve_config(
        endpoint=endpoint,
        api_key=api_key,
        service_name=service_name,
        headers=headers,
        disabled=disabled,
        sample_rate=sample_rate,
        capture_content=capture_content,
        heartbeat=heartbeat,
        heartbeat_interval=heartbeat_interval,
        agent_name=agent_name,
        partial_spans=partial_spans,
        partial_spans_delay=partial_spans_delay,
    )
    # telemetry.sdk.* is reserved for the OTel SDK itself (Resource.create fills
    # it); we identify as a distribution via telemetry.distro.*.
    resource = Resource.create(
        {
            "service.name": config.service_name,
            "telemetry.distro.name": "glassflow-rius",
            "telemetry.distro.version": __version__,
        }
    )
    sampler = ParentBased(root=TraceIdRatioBased(config.sample_rate))
    provider = TracerProvider(resource=resource, sampler=sampler)

    export_health: ExportOutcomeExporter | None = None
    connectivity_thread: threading.Thread | None = None
    if not config.disabled:
        if span_exporter is None:
            if _missing_managed_credentials(config):
                # The diagnosis is already certain; no probe needed (it
                # would only repeat this warning as a 401).
                logger.warning(
                    "no API key configured (GLASSFLOW_API_KEY unset and no Authorization "
                    "header): traces sent to %s will be rejected with 401. Set "
                    "GLASSFLOW_API_KEY or pass api_key= to rius.init().",
                    config.endpoint,
                )
            else:
                connectivity_thread = threading.Thread(
                    target=check_connectivity,
                    args=(
                        config.traces_endpoint,
                        config.headers,
                        connectivity_transport or _default_probe_send,
                    ),
                    name="rius-connectivity-check",
                    daemon=True,
                )
                connectivity_thread.start()
        exporter = span_exporter if span_exporter is not None else build_span_exporter(config)
        if not config.capture_content or mask is not None:
            exporter = MaskingSpanExporter(
                exporter, capture_content=config.capture_content, mask=mask
            )
        # Outermost wrapper so it observes the outcome of the whole chain
        # (masking included); client.flush() reads it for honest delivery
        # reporting.
        export_health = ExportOutcomeExporter(exporter, endpoint=config.endpoint)
        batch_processor = BatchSpanProcessor(export_health)
        if config.partial_spans:
            # Pending snapshots ride the SAME batch pipeline as final spans
            # (exporter, retries, masking); see pending.py for the contract.
            provider.add_span_processor(
                PendingSpanProcessor(batch_processor, delay=config.partial_spans_delay)
            )
        provider.add_span_processor(batch_processor)

    if set_global and not config.disabled:
        trace.set_tracer_provider(provider)
        if trace.get_tracer_provider() is not provider:
            logger.warning(
                "could not register the rius tracer provider as the OpenTelemetry "
                "global (another provider is already set); spans from @observe and "
                "rius.get_tracer() will keep using the pre-existing provider. Use "
                "the returned client's get_tracer() for scoped tracing."
            )

    # Instrumentors are process-global singletons: auto-enable only for a global
    # init; a scoped client must opt in explicitly via `instruments=[...]`.
    if not config.disabled and (set_global or instruments is not None):
        enable_instrumentations(provider, instruments)

    # Heartbeat: process-lifetime liveness, independent of trace
    # traffic. The tracker rides the provider as a span processor so payloads
    # can carry the currently-open root trace ids; disabled kills it too.
    sender: HeartbeatSender | None = None
    if config.heartbeat and not config.disabled:
        tracker = OpenRootSpanTracker()
        provider.add_span_processor(tracker)
        sender = HeartbeatSender(
            url=config.heartbeat_endpoint,
            headers=config.headers,
            interval=config.heartbeat_interval,
            agent_name=config.agent_name,
            tracker=tracker,
            transport=heartbeat_transport,
        )
        sender.start()

    client = GlassflowClient(
        provider,
        config,
        heartbeat=sender,
        export_health=export_health,
        connectivity_thread=connectivity_thread,
    )
    if set_global:
        _current_client = client
    return client


def get_tracer(name: str = TRACER_NAME) -> trace.Tracer:
    """Return a tracer from the globally configured provider."""
    return trace.get_tracer(name, __version__)
