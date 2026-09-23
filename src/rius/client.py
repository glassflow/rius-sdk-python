"""SDK entrypoint: configure OpenTelemetry and export GenAI traces via OTLP."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from . import __version__, _agent, _tracer
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
from .semconv import GEN_AI_AGENT_NAME, SERVICE_INSTANCE_ID, SERVICE_VERSION, TRACER_NAME
from .session import SessionSpanProcessor
from .user import UserSpanProcessor
from .workspace import ExporterFactory, RoutingSpanExporter, WorkspaceSpanProcessor

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
        routing: RoutingSpanExporter | None = None,
    ) -> None:
        self._provider = provider
        self.config = config
        self._heartbeat = heartbeat
        self._export_health = export_health
        self._connectivity_thread = connectivity_thread
        self._routing = routing
        self._is_shutdown = False

    def register_workspace(self, alias: str, api_key: str) -> None:
        """Add (or rotate the key of) a workspace destination at runtime.

        Requires routing to be enabled at ``init`` time via ``workspaces=``
        (an empty dict opts in with no static routes). Spans started inside
        ``rius.workspace(alias)`` are then exported with ``api_key``.
        """
        if self.config.disabled:
            # The kill switch builds no pipeline, so there is nothing to route
            # to; a call site that works when enabled must not throw here.
            return
        if self._routing is None:
            raise RuntimeError(
                "workspace routing is not enabled: pass workspaces={...} to init() "
                "(an empty dict is fine) to opt in before registering destinations"
            )
        self._routing.register(alias, api_key)

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
        if self._connectivity_thread is not None:
            # Daemon thread; give an in-flight probe a moment to log its
            # verdict before the pipeline it describes goes away.
            self._connectivity_thread.join(timeout=0.5)
        self._provider.shutdown()
        self._is_shutdown = True
        with _lock:
            if _current_client is self:
                _current_client = None
        _tracer.withdraw(self._provider)
        _agent.withdraw(self._provider)


def init(
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    service_name: str | None = None,
    service_version: str | None = None,
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
    session_id: str | None = None,
    workspaces: dict[str, str] | None = None,
    workspace_exporter_factory: ExporterFactory | None = None,
    set_global: bool = True,
) -> GlassflowClient:
    """Initialize the SDK: build a tracer provider that exports OTLP traces.

    Calling ``init()`` again while a global client is active logs a warning and
    returns the existing client unchanged; call ``shutdown()`` on it first to
    reconfigure. After that the SDK's own helpers (``start_span``, ``observe``,
    the generation helpers) follow the new client, as do the bundled
    instrumentors. Third-party code that took a tracer from the OpenTelemetry
    global keeps the first provider, because that global is write-once.

    Args:
        endpoint: Base OTLP endpoint. Traces are sent to ``<endpoint>/v1/traces``.
        api_key: API key; injected as an ``Authorization: Bearer`` header.
        service_name: Value for the ``service.name`` resource attribute.
        service_version: Value for the ``service.version`` resource
            attribute, e.g. the release or image tag this process is running.
            Resolution order is this argument, then ``RIUS_SERVICE_VERSION``,
            then whatever ``OTEL_RESOURCE_ATTRIBUTES`` supplies (the OTel SDK
            merges that variable into every resource), then unset. There is no
            placeholder default: a fake version would group every deployment
            into one bucket, which is worse than the attribute being absent.
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
            (``RIUS_HEARTBEAT``; on by default, set False to opt out). Pings
            ``<endpoint>/v1/heartbeat`` from init until process exit so the
            platform can tell a live-but-idle agent from a vanished one.
            Each ``init()`` mints one instance id, sent both in heartbeat
            payloads and on every span as the ``service.instance.id``
            resource attribute, so the platform can join the two and count
            replicas. Fork caveat: a child forked after ``init()`` heartbeats
            under a fresh id, but its spans keep the parent's (the OTel
            Resource is immutable), identifying the pre-fork process family;
            for exact per-worker span identity, call ``init()`` after the
            fork (e.g. in gunicorn's ``post_fork``).
        heartbeat_interval: Seconds between pings (default 15, clamped to
            ``[5, 300]``; the backend derives staleness from this).
        agent_name: Identity both heartbeats and spans group under, stamped
            on the resource as ``gen_ai.agent.name``; defaults to
            ``service_name``.
        heartbeat_transport: Override the heartbeat HTTP transport
            (useful for testing, like ``span_exporter``).
        connectivity_transport: Override the HTTP send used by the one-shot
            background connectivity check (useful for testing). The check
            POSTs an empty OTLP request at init and logs an actionable
            warning on 401/403, unreachable host, or other non-2xx, so a
            bad key or endpoint is visible immediately instead of surfacing
            as silently missing traces.
        session_id: Process-wide session id (``RIUS_SESSION_ID``), stamped as
            ``session.id`` on every span so the platform groups this
            process's traces into one session. For one-run-per-process
            agents; a server handling many sessions scopes each one with
            ``rius.session()`` instead, which overrides this default.
        workspaces: Enable multi-workspace routing: a mapping of alias to
            API key. Spans started inside ``rius.workspace(alias)`` are
            exported with that workspace's key; spans outside any scope use
            the default ``api_key``. Pass ``{}`` to opt in with no static
            routes and register destinations later via
            ``register_workspace()``. One trace must stay inside one
            workspace; see ``rius.workspace``.
        workspace_exporter_factory: Override how per-workspace exporters are
            built from an API key (useful for testing, like
            ``span_exporter``). Defaults to the standard OTLP exporter
            against the configured endpoint.
        set_global: Register the provider as the global OpenTelemetry provider.
    """
    global _current_client
    with _lock:
        if set_global and _current_client is not None and not _current_client._is_shutdown:
            logger.warning(
                "rius.init() was already called; keeping the existing configuration. "
                "Call .shutdown() on the existing client first if you need to reconfigure."
            )
            return _current_client
        return _do_init(
            endpoint=endpoint,
            api_key=api_key,
            service_name=service_name,
            service_version=service_version,
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
            session_id=session_id,
            workspaces=workspaces,
            workspace_exporter_factory=workspace_exporter_factory,
            set_global=set_global,
        )


def _do_init(
    *,
    endpoint: str | None,
    api_key: str | None,
    service_name: str | None,
    service_version: str | None,
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
    session_id: str | None,
    workspaces: dict[str, str] | None,
    workspace_exporter_factory: ExporterFactory | None,
    set_global: bool,
) -> GlassflowClient:
    global _current_client
    config = resolve_config(
        endpoint=endpoint,
        api_key=api_key,
        service_name=service_name,
        service_version=service_version,
        headers=headers,
        disabled=disabled,
        sample_rate=sample_rate,
        capture_content=capture_content,
        heartbeat=heartbeat,
        heartbeat_interval=heartbeat_interval,
        agent_name=agent_name,
        partial_spans=partial_spans,
        partial_spans_delay=partial_spans_delay,
        session_id=session_id,
    )
    # One identity per client lifetime, shared by spans (resource) and
    # heartbeats (payload instance_id) so the backend can join them. Minted
    # here — before the Resource — because the sender is constructed much
    # later. Fork caveat: the Resource is immutable, so a forked child's
    # spans keep this (ancestor) id while its heartbeat re-arms with a fresh
    # one; deployments needing exact per-worker span identity should init()
    # after fork.
    instance_id = str(uuid.uuid4())
    # telemetry.sdk.* is reserved for the OTel SDK itself (Resource.create fills
    # it); we identify as a distribution via telemetry.distro.*.
    resource = Resource.create(
        {
            "service.name": config.service_name,
            SERVICE_INSTANCE_ID: instance_id,
            # Only when known. Passing None would stamp the attribute with a
            # null OTel drops with a warning, and passing a placeholder would
            # be worse: an absent service.version reads as "unknown", a fake
            # one reads as a real (wrong) release. Leaving the key out also
            # lets OTEL_RESOURCE_ATTRIBUTES supply it, since Resource.create
            # merges these attributes OVER the detected ones.
            **({SERVICE_VERSION: config.service_version} if config.service_version else {}),
            # The same name the heartbeat sender reports. Config resolution
            # already defaults it to the service name, so a process that sets
            # only a service name is unchanged; one that sets both no longer
            # has its spans grouped under a different name than its heartbeats.
            GEN_AI_AGENT_NAME: config.agent_name,
            "telemetry.distro.name": "glassflow-rius",
            "telemetry.distro.version": __version__,
        }
    )
    sampler = ParentBased(root=TraceIdRatioBased(config.sample_rate))
    provider = TracerProvider(resource=resource, sampler=sampler)

    export_health: ExportOutcomeExporter | None = None
    connectivity_thread: threading.Thread | None = None
    routing: RoutingSpanExporter | None = None
    if not config.disabled:
        if span_exporter is None:
            if _missing_managed_credentials(config):
                # The diagnosis is already certain; no probe needed (it
                # would only repeat this warning as a 401).
                logger.warning(
                    "no API key configured (RIUS_API_KEY unset and no Authorization "
                    "header): traces sent to %s will be rejected with 401. Set "
                    "RIUS_API_KEY or pass api_key= to rius.init().",
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
        if workspaces is not None:
            # Innermost in the chain, so masking and export-health wrap the
            # whole fan-out and apply to every destination alike.
            factory = workspace_exporter_factory or _workspace_exporter_factory(config)
            routing = RoutingSpanExporter(exporter, factory, routes=workspaces)
            exporter = routing
        if not config.capture_content or mask is not None:
            exporter = MaskingSpanExporter(
                exporter, capture_content=config.capture_content, mask=mask
            )
        # Outermost wrapper so it observes the outcome of the whole chain
        # (masking included); client.flush() reads it for honest delivery
        # reporting.
        export_health = ExportOutcomeExporter(exporter, endpoint=config.endpoint)
        batch_processor = BatchSpanProcessor(export_health)
        # Registered BEFORE the pending processor: both act at on_start, and
        # the pending snapshot is built from the attributes already on the
        # span, so the session id (and the workspace route, which decides
        # which destination the snapshot itself goes to) must be stamped
        # first to ride it.
        provider.add_span_processor(SessionSpanProcessor(config.session_id))
        provider.add_span_processor(UserSpanProcessor())
        if routing is not None:
            provider.add_span_processor(WorkspaceSpanProcessor())
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
                "global (another provider is already set, or a previous init() "
                "claimed it). rius' own helpers (@observe, start_span, generations) "
                "follow this client regardless; third-party code using "
                "opentelemetry.trace.get_tracer() keeps the pre-existing provider."
            )
    if set_global:
        # The helpers follow the active client, not the write-once OTel global,
        # so a shutdown()+init() cycle moves them to the new pipeline too.
        _tracer.publish(provider)
        # AGENT spans fall back to this when the caller names no agent.
        _agent.publish(provider, config.agent_name)

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
            instance_id=instance_id,
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
        routing=routing,
    )
    if set_global:
        _current_client = client
    return client


def get_tracer(name: str = TRACER_NAME) -> trace.Tracer:
    """Return a tracer bound to the active global client's provider.

    Falls back to the OpenTelemetry global provider when no global ``init()``
    is active, so it is safe to call before or without ``init()``.
    """
    if name == TRACER_NAME:
        return _tracer.sdk_tracer()
    with _lock:
        client = _current_client
    if client is not None and not client._is_shutdown:
        return client._provider.get_tracer(name, __version__)
    return trace.get_tracer(name, __version__)


def _workspace_exporter_factory(config: GlassflowConfig) -> Callable[[str], SpanExporter]:
    """Per-workspace OTLP exporters: same endpoint, that workspace's key."""

    def build(api_key: str) -> SpanExporter:
        headers = {
            **{k: v for k, v in (config.headers or {}).items() if k.lower() != "authorization"},
            "Authorization": f"Bearer {api_key}",
        }
        return OTLPSpanExporter(endpoint=config.traces_endpoint, headers=headers)

    return build


def register_workspace(alias: str, api_key: str) -> None:
    """Add (or rotate the key of) a workspace destination on the global client.

    The module-level twin of ``client.register_workspace()``. Requires a
    global ``init(workspaces=...)`` to have opted into routing.
    """
    with _lock:
        client = _current_client
    if client is None:
        raise RuntimeError("rius.init() has not been called (no global client)")
    client.register_workspace(alias, api_key)
