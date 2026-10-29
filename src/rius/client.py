"""SDK entrypoint: configure OpenTelemetry and export GenAI traces via OTLP."""

from __future__ import annotations

import logging
import os
import threading
import uuid
import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import unquote

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, SynchronousMultiSpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, Sampler, TraceIdRatioBased

from . import __version__, _agent, _tracer
from ._credential_headers import CredentialHeaderDroppingExporter
from ._native_spans import NativeChatSpanDemoter
from .aggregated_usage import WrappedCallTracker
from .compaction import CompactedContextProcessor
from .config import DEFAULT_ENDPOINT, GlassflowConfig, resolve_config
from .export_health import (
    ExportOutcomeExporter,
    ProbeTransport,
    _default_probe_send,
    check_connectivity,
)
from .foreign import (
    Bridge,
    BridgeAwareSampler,
    ForeignParentDetector,
    OwnPipeline,
    ResourceAdoptingExporter,
    bridge,
)
from .heartbeat import HeartbeatSender, OpenRootSpanTracker, mint_instance_identity
from .hints import Hint, HintInbox, current_trace_id
from .instrumentation import enable_instrumentations, instruments_for
from .instrumentation import release as release_instrumentations
from .masking import MaskingSpanExporter
from .normalization import NormalizingSpanExporter, NormalizingSpanProcessor
from .pending import PendingSpanProcessor
from .semconv import (
    GEN_AI_AGENT_NAME,
    GEN_AI_MAIN_AGENT_DESCRIPTION,
    GEN_AI_MAIN_AGENT_ID,
    GEN_AI_MAIN_AGENT_NAME,
    RIUS_MAIN_AGENT_DESCRIPTION,
    RIUS_MAIN_AGENT_ID,
    RIUS_MAIN_AGENT_NAME,
    RIUS_MAIN_AGENT_VERSION,
    RIUS_SDK_GLOBAL_PROVIDER,
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_VERSION,
    TELEMETRY_DISTRO_NAME,
    TELEMETRY_DISTRO_VERSION,
    TRACER_NAME,
)
from .session import SessionSpanProcessor
from .user import UserSpanProcessor
from .workspace import ExporterFactory, RoutingSpanExporter, WorkspaceSpanProcessor

logger = logging.getLogger(__name__)

#: The per-span attribute count ``init()`` gives its provider, in place of
#: OTel's default of 128. The OpenInference instrumentors write one attribute
#: per message field and per tool field, so an agent loop with ten tools
#: passes 128 within about six (anthropic) to nine (openai) turns, and OTel
#: then evicts the OLDEST keys: the request bag (model and parameters), the
#: tool definitions and the system prompt go first. 4096 holds hundreds of
#: turns; the value length is not raised, since our JSON attributes carry
#: their own cap.
DEFAULT_SPAN_ATTRIBUTE_COUNT_LIMIT = 4096

#: The standard OTel env vars that set the span attribute count. When the user
#: set either, their value wins and ``init()`` leaves resolution to OTel.
_GLOBAL_COUNT_ENV_VAR = "OTEL_ATTRIBUTE_COUNT_LIMIT"
_ATTRIBUTE_COUNT_ENV_VARS = ("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", _GLOBAL_COUNT_ENV_VAR)


#: OTel's default for event and link attribute counts (the spec's 128).
#: Passed explicitly whenever the SDK sets the span count, for the reason in
#: :func:`_span_limits`.
_OTEL_DEFAULT_ATTRIBUTE_COUNT_LIMIT = 128


def _honoured_count(name: str) -> bool:
    """True when ``name`` holds a count OTel's own env parsing accepts.

    Mirrors ``SpanLimits._from_env_if_absent`` (opentelemetry-sdk): the value
    must parse with ``int()`` to a non-negative number. ``int()`` itself
    ignores surrounding whitespace and rejects a blank string, which OTel
    reads as "unset", so a blank value is not a count either.
    """
    try:
        return int(os.environ.get(name, "")) >= 0
    except ValueError:
        return False


def _unusable_count(name: str) -> bool:
    """True when ``name`` is set but holds no count OTel honours.

    ``SpanLimits()`` raises on such a value, or, for a blank one, reads it as
    "no limit"; either way it is not the user's choice of a limit.
    """
    return name in os.environ and not _honoured_count(name)


def _warn_unusable(name: str, fallback: str) -> None:
    # A blank entry is routine in a container env; it is not worth a warning.
    if os.environ[name].strip():
        logger.warning(
            "%s=%r is not a non-negative integer, so OpenTelemetry cannot use it; %s",
            name,
            os.environ[name],
            fallback,
        )


def _span_limits() -> SpanLimits:
    """The provider's limits: ours for the attribute count, unless the user chose.

    An explicit ``max_span_attributes`` beats the env var inside ``SpanLimits``,
    so passing ours unconditionally would silently override a user's
    ``OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT``. When either count variable holds a
    count OTel honours, OTel resolves everything itself, its own precedence
    included (the span count, then the global count). That is the TypeScript
    SDK's rule too.

    Anything else is not a choice of limit, and leaving it to OTel would be
    wrong: ``SpanLimits()`` parses both variables whatever it is passed, and
    raises on a set one that is no count (out of ``init()``), or reads a blank
    one as "no limit". So each unusable variable is replaced by an explicit
    value: the global count by OTel's default, the span count by the user's
    global count when there is one and by ours when there is not.
    """
    span_var, global_var = _ATTRIBUTE_COUNT_ENV_VARS
    if not any(_honoured_count(name) for name in _ATTRIBUTE_COUNT_ENV_VARS):
        for name in _ATTRIBUTE_COUNT_ENV_VARS:
            if name in os.environ:
                _warn_unusable(
                    name,
                    f"rius applies its span attribute count limit of "
                    f"{DEFAULT_SPAN_ATTRIBUTE_COUNT_LIMIT} instead.",
                )
        return SpanLimits(
            max_attributes=_OTEL_DEFAULT_ATTRIBUTE_COUNT_LIMIT,
            max_span_attributes=DEFAULT_SPAN_ATTRIBUTE_COUNT_LIMIT,
        )
    # One count is the user's. Keep it, and replace only the unusable one.
    limits: dict[str, int] = {}
    if _unusable_count(global_var):
        _warn_unusable(
            global_var,
            f"rius uses OpenTelemetry's default of {_OTEL_DEFAULT_ATTRIBUTE_COUNT_LIMIT} "
            "for it instead.",
        )
        limits["max_attributes"] = _OTEL_DEFAULT_ATTRIBUTE_COUNT_LIMIT
    if _unusable_count(span_var):
        # The span count is not honoured, so the global count is: OTel's own
        # fallback for an unset span count.
        global_count = int(os.environ[global_var])
        _warn_unusable(span_var, f"rius uses {global_var}={global_count} for spans instead.")
        limits["max_span_attributes"] = global_count
    return SpanLimits(**limits)


_lock = threading.Lock()
_current_client: GlassflowClient | None = None
# Every provider init() built, so a global left over from an earlier
# shutdown()+init() cycle is recognised as ours rather than another SDK's.
_own_providers: weakref.WeakSet[TracerProvider] = weakref.WeakSet()


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
        bridge: Bridge | None = None,
        hints: HintInbox | None = None,
    ) -> None:
        self._provider = provider
        self._hints = hints
        self._bridge = bridge
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

    def drain_hints(self, trace_id: str | None = None) -> list[Hint]:
        """Remove and return the hints received for a run, oldest first.

        Returns the hints for ``trace_id`` (default: the trace of the current
        span context) plus the hints about the instance as a whole. Outside
        any span, only the instance-wide ones. Always empty when hints are
        off (``init(hints=True)`` / ``RIUS_HINTS`` turns them on).
        """
        if self._hints is None:
            return []
        return self._hints.drain(trace_id if trace_id is not None else current_trace_id())

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
        if self._bridge is not None:
            # First, so the other provider's spans stop entering a pipeline
            # that is about to shut down. Its own processors are never touched.
            self._bridge.release()
        self._provider.shutdown()
        self._is_shutdown = True
        with _lock:
            if _current_client is self:
                _current_client = None
        _tracer.withdraw(self._provider)
        _agent.withdraw(self._provider)
        release_instrumentations(self._provider)


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
    hints: bool | None = None,
    hints_per_trace: int | None = None,
    on_hint: Callable[[Hint], object] | None = None,
    agent_name: str | None = None,
    main_agent_id: str | None = None,
    main_agent_description: str | None = None,
    main_agent_version: str | None = None,
    heartbeat_transport: Callable[[dict[str, Any]], None] | None = None,
    connectivity_transport: ProbeTransport | None = None,
    partial_spans: bool | None = None,
    partial_spans_delay: float | None = None,
    session_id: str | None = None,
    workspaces: dict[str, str] | None = None,
    workspace_exporter_factory: ExporterFactory | None = None,
    set_global: bool = True,
    bridge_foreign_provider: bool | None = None,
) -> GlassflowClient:
    """Initialize the SDK: build a tracer provider that exports OTLP traces.

    Calling ``init()`` again while a global client is active logs a warning and
    returns the existing client unchanged; call ``shutdown()`` on it first to
    reconfigure. After that the SDK's own helpers (``start_span``, ``observe``,
    the generation helpers) follow the new client, as do the bundled
    instrumentors. Third-party code that took a tracer from the OpenTelemetry
    global keeps the first provider, because that global is write-once; its
    spans still reach the new client, which attaches its pipeline to that
    provider whatever ``bridge_foreign_provider`` says (the provider is the
    SDK's own, not another SDK's). Such a span is sampled by the first
    provider's sampler as well as the new client's.

    The provider allows up to 4096 attributes per span rather than OTel's
    default of 128, which a long OpenInference agent span passes within a few
    turns and then silently evicts its oldest keys (model, tools, system
    prompt). ``OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT`` or
    ``OTEL_ATTRIBUTE_COUNT_LIMIT``, when set, wins instead. A span that still
    hits the limit reports its drop count on the wire.

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
            Needs credentials (``api_key`` or an ``Authorization`` header):
            without them it does not start, since the endpoint would reject
            every ping. A ``heartbeat_transport`` runs regardless.
            Each ``init()`` mints one instance id, sent both in heartbeat
            payloads and on every span as the ``service.instance.id``
            resource attribute, so the platform can join the two and count
            replicas. Fork caveat: a child forked after ``init()`` heartbeats
            under a fresh id, but its spans keep the parent's (the OTel
            Resource is immutable), identifying the pre-fork process family;
            for exact per-worker span identity, call ``init()`` after the
            fork (e.g. in gunicorn's ``post_fork``).
        heartbeat_interval: Seconds between pings (default 15, clamped to
            ``[5, 300]``). The server fixes stale at 30s and gone at 60s
            since the last ping, so keep this well under 30s.
        hints: Ask Rius for hints on the heartbeat response (``RIUS_HINTS``;
            off by default). A hint is a short note about a problem Rius
            detected in one of this process's open runs; read them with
            ``drain_hints()``. Needs the heartbeat and its default transport:
            with a ``heartbeat_transport`` hints stay off.
        hints_per_trace: The most hints accepted for one run
            (``RIUS_HINTS_PER_TRACE``, default 3); later ones are dropped.
        on_hint: Called once with each newly accepted ``Hint``, on a
            dedicated daemon thread, so it can never delay a heartbeat. If it
            falls behind, the oldest pending calls are skipped. An exception
            it raises is logged, and the hint stays queued for
            ``drain_hints()`` either way.
        agent_name: Identity both heartbeats and spans group under, stamped
            on the resource as ``gen_ai.main_agent.name`` AND, unchanged, as
            ``gen_ai.agent.name``; defaults to ``service_name``. The
            ``gen_ai.main_agent.*`` family is where the process-level question
            "which agent is this?" lives: ``gen_ai.agent.name`` has no
            resource-level meaning in the conventions and on a span means the
            agent being INVOKED. Both keys are stamped — a swap would break
            agent identity for every deployment running an SDK newer than the
            sink, and resource attributes ride once per OTLP batch, so the
            duplication costs essentially nothing. The deprecated
            ``rius.main_agent.name`` carries the same value through its
            deprecation window. A process whose name resolves to the
            ``unknown_service`` placeholder emits no agent name at all on the
            resource (neither key), the same suppression the span helpers
            apply. It is ALSO the
            ``gen_ai.main_agent.id`` when no ``main_agent_id`` is set, so
            renaming the agent changes that id.
        main_agent_id: Stable identifier of the agent this process IS
            (``RIUS_MAIN_AGENT_ID``), stamped as ``gen_ai.main_agent.id`` and
            the deprecated ``rius.main_agent.id``. It must be STABLE across
            restarts and deployments; the conventions say it MUST NOT be
            generated from an in-memory or process-local id, because a value
            that changes every restart mints a new "agent" every restart.
            When it is not set, ``gen_ai.main_agent.id`` is a
            ``gen_ai.main_agent.id`` already in ``OTEL_RESOURCE_ATTRIBUTES``,
            else the agent name, so RENAMING THE AGENT CHANGES THE ID unless
            an id is set (a one-time DEBUG log says so); ``rius.main_agent.id`` is
            stamped only when this is set. With neither an id nor a name
            (the ``unknown_service`` placeholder), no ``gen_ai.main_agent.*``
            key is emitted at all: the conventions do not allow the entity
            without an identifier.
        main_agent_description: What this agent does
            (``RIUS_MAIN_AGENT_DESCRIPTION``), stamped as
            ``rius.main_agent.description`` when set, and as
            ``gen_ai.main_agent.description`` when set and
            ``gen_ai.main_agent.id`` is emitted.
        main_agent_version: Version of the agent DEFINITION — its prompt,
            tools and policy (``RIUS_MAIN_AGENT_VERSION``), stamped as
            ``rius.main_agent.version``. Deliberately NOT ``service.version``
            and never derived from it in either direction: a service can sit
            at 2.3.1 while its agent definition is at 7, and the two move
            independently. Stamped only when set.
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
        bridge_foreign_provider: What to do when another SDK already holds the
            OpenTelemetry global provider (``RIUS_BRIDGE_FOREIGN_PROVIDER``;
            off by default). Off, rius keeps to its own provider: LLM spans
            started inside the other SDK's spans arrive without their parent,
            flagged ``rius.parent.foreign``, counted in the heartbeat, and the
            first one logs a warning. On, rius also exports the other SDK's
            spans (the parents), under rius's resource identity and subject
            to rius's sampling and content settings, and it never modifies
            what the other SDK exports. Either way the resource records the
            conflict as ``rius.sdk.global_provider``.
            Applies to a global ``init()`` only.
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
            hints=hints,
            hints_per_trace=hints_per_trace,
            on_hint=on_hint,
            agent_name=agent_name,
            main_agent_id=main_agent_id,
            main_agent_description=main_agent_description,
            main_agent_version=main_agent_version,
            heartbeat_transport=heartbeat_transport,
            connectivity_transport=connectivity_transport,
            partial_spans=partial_spans,
            partial_spans_delay=partial_spans_delay,
            session_id=session_id,
            workspaces=workspaces,
            workspace_exporter_factory=workspace_exporter_factory,
            set_global=set_global,
            bridge_foreign_provider=bridge_foreign_provider,
        )


def _main_agent_attributes(config: GlassflowConfig) -> dict[str, str]:
    """The main-agent resource attributes, each only when known.

    ``rius.main_agent.*`` (deprecated, kept through their window so a sink
    that reads only the old keys keeps the agent's identity), plus the
    conventions' ``gen_ai.main_agent.*`` entity, whose rule lives in
    ``_upstream_main_agent_attributes``. ``rius.main_agent.version`` has no
    upstream twin.

    Every rius key is omitted rather than defaulted, for the reason
    ``service.version`` is: an absent attribute reads as "not told", a
    fabricated one reads as a real (wrong) answer. The name additionally
    drops the ``unknown_service`` placeholder, which is not an identity.
    """
    name = _agent.named_agent(config.agent_name)
    attributes: dict[str, str] = {}
    if name is not None:
        attributes[RIUS_MAIN_AGENT_NAME] = name
    if config.main_agent_id:
        attributes[RIUS_MAIN_AGENT_ID] = config.main_agent_id
    if config.main_agent_description:
        attributes[RIUS_MAIN_AGENT_DESCRIPTION] = config.main_agent_description
    attributes.update(
        _upstream_main_agent_attributes(
            name=config.agent_name,
            configured_id=config.main_agent_id,
            description=config.main_agent_description,
            operator=_operator_main_agent_attributes(os.getenv("OTEL_RESOURCE_ATTRIBUTES")),
        )
    )
    # Never from config.service_version: different fact, independent lifecycle.
    if config.main_agent_version:
        attributes[RIUS_MAIN_AGENT_VERSION] = config.main_agent_version
    return attributes


# Once per process: the hint is advice about configuration, not news.
_derived_main_agent_id_hinted = False


_UPSTREAM_MAIN_AGENT_KEYS = (
    GEN_AI_MAIN_AGENT_ID,
    GEN_AI_MAIN_AGENT_NAME,
    GEN_AI_MAIN_AGENT_DESCRIPTION,
)


def _operator_main_agent_attributes(raw: str | None) -> dict[str, str]:
    """The ``gen_ai.main_agent.*`` keys an operator put in
    ``OTEL_RESOURCE_ATTRIBUTES``, as that variable carries them.

    The format is the spec's: comma-separated ``key=value`` pairs, values
    percent-encoded. A malformed pair is skipped: ``Resource.create`` already
    warns about it, and a broken variable must not stop a process starting.
    """
    found: dict[str, str] = {}
    for pair in (raw or "").split(","):
        key, separator, value = pair.partition("=")
        if separator and key.strip() in _UPSTREAM_MAIN_AGENT_KEYS:
            found[key.strip()] = unquote(value.strip())
    return found


def _usable(value: str | None) -> str | None:
    """``value``, unless it is unset, empty or only whitespace."""
    return value if value and value.strip() else None


def _upstream_main_agent_attributes(
    *,
    name: str | None,
    configured_id: str | None,
    description: str | None,
    operator: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The conventions' ``gen_ai.main_agent`` entity: id, name, description.

    The ONE place the rule for emitting it lives. The conventions make
    ``gen_ai.main_agent.id`` the entity's identity and say the entity MUST NOT
    be emitted without a stable one. So the id is, in order: the configured
    ``main_agent_id`` (the argument, then ``RIUS_MAIN_AGENT_ID``), then a
    ``gen_ai.main_agent.id`` the operator already put on the resource
    (``operator``, from ``OTEL_RESOURCE_ATTRIBUTES``), then the resolved agent
    name, as is: it is stable across restarts and readable, but renaming the
    agent changes it. An empty or whitespace-only value counts as unset. With
    none of them (and the name empty or the ``unknown_service`` placeholder)
    no ``gen_ai.main_agent.*`` key is emitted from the SDK's settings. The
    SDK's name and description ride only with the id; an operator-set one is
    the operator's own and is passed through as is.

    An operator-set id counts as configured, so it logs no hint. An
    operator-set name or description is never overwritten.

    ``rius.main_agent.id`` is NOT derived, nor taken from the operator's
    value: it stays the configured id only.
    """
    global _derived_main_agent_id_hinted
    operator = operator or {}
    named = _usable(_agent.named_agent(name))
    configured = _usable(configured_id) or _usable(operator.get(GEN_AI_MAIN_AGENT_ID))
    entity_id = configured or named
    attributes: dict[str, str] = {}
    if entity_id:
        if not configured and not _derived_main_agent_id_hinted:
            _derived_main_agent_id_hinted = True
            logger.debug(
                "rius: gen_ai.main_agent.id is the agent name %r because no main_agent_id "
                "is set; set main_agent_id (RIUS_MAIN_AGENT_ID) to a stable identifier so "
                "renaming the agent does not change it.",
                named,
            )
        attributes[GEN_AI_MAIN_AGENT_ID] = entity_id
        if named is not None:
            attributes[GEN_AI_MAIN_AGENT_NAME] = named
        if description:
            attributes[GEN_AI_MAIN_AGENT_DESCRIPTION] = description
    # The operator's own name and description stand, as they would on any
    # resource the OTel SDK merges them into.
    for key in (GEN_AI_MAIN_AGENT_NAME, GEN_AI_MAIN_AGENT_DESCRIPTION):
        if _usable(operator.get(key)):
            attributes[key] = operator[key]
    return attributes


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
    hints: bool | None,
    hints_per_trace: int | None,
    on_hint: Callable[[Hint], object] | None,
    agent_name: str | None,
    main_agent_id: str | None,
    main_agent_description: str | None,
    main_agent_version: str | None,
    heartbeat_transport: Callable[[dict[str, Any]], None] | None,
    connectivity_transport: ProbeTransport | None,
    partial_spans: bool | None,
    partial_spans_delay: float | None,
    session_id: str | None,
    workspaces: dict[str, str] | None,
    workspace_exporter_factory: ExporterFactory | None,
    set_global: bool,
    bridge_foreign_provider: bool | None,
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
        hints=hints,
        hints_per_trace=hints_per_trace,
        agent_name=agent_name,
        main_agent_id=main_agent_id,
        main_agent_description=main_agent_description,
        main_agent_version=main_agent_version,
        partial_spans=partial_spans,
        partial_spans_delay=partial_spans_delay,
        session_id=session_id,
        bridge_foreign_provider=bridge_foreign_provider,
    )
    # Read before the provider exists: the resource is immutable once built.
    foreign_global = _foreign_global_name() if set_global else None
    # One identity per client lifetime, shared by spans (resource) and
    # heartbeats (payload instance_id) so the backend can join them. Minted
    # here — before the Resource — because the sender is constructed much
    # later. Fork caveat: the Resource is immutable, so a forked child's
    # spans keep this (ancestor) id while its heartbeat re-arms with a fresh
    # one; deployments needing exact per-worker span identity should init()
    # after fork.
    # Derived from a per-instance secret the heartbeat presents, so another
    # holder of the workspace key cannot ping as this instance. A custom
    # transport is handed the payload only and could never send the secret,
    # so it pings as a legacy instance: liveness as before, never hints.
    instance_id: str
    instance_secret: str | None
    if heartbeat_transport is None:
        instance_id, instance_secret = mint_instance_identity()
    else:
        instance_id, instance_secret = str(uuid.uuid4()), None
    agent_name = _agent.named_agent(config.agent_name)
    # telemetry.sdk.* is reserved for the OTel SDK itself (Resource.create fills
    # it); we identify as a distribution via telemetry.distro.*.
    resource = Resource.create(
        {
            SERVICE_NAME: config.service_name,
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
            # Omitted on the unknown_service placeholder, by the rule the main
            # agent keys below follow: a process that named nothing claims no
            # agent identity.
            **({GEN_AI_AGENT_NAME: agent_name} if agent_name is not None else {}),
            # The SAME resolved name as gen_ai.main_agent.name (and, through
            # its deprecation window, rius.main_agent.name), the keys where the
            # process-level question belongs: gen_ai.agent.name has no
            # resource-level meaning in the conventions and means the INVOKED
            # agent on a span. Additive on purpose, so a deployment on a newer
            # SDK than the sink keeps its identity. Suppressed on the
            # placeholder by the same rule the span helpers use: a process
            # that named nothing claims no agent identity.
            **_main_agent_attributes(config),
            **({RIUS_SDK_GLOBAL_PROVIDER: f"foreign:{foreign_global}"} if foreign_global else {}),
            TELEMETRY_DISTRO_NAME: "glassflow-rius",
            TELEMETRY_DISTRO_VERSION: __version__,
        }
    )
    sampler = BridgeAwareSampler(ParentBased(root=TraceIdRatioBased(config.sample_rate)))
    limits = _span_limits()
    provider = TracerProvider(resource=resource, sampler=sampler, span_limits=limits)
    _own_providers.add(provider)
    # Every processor rides this one chain, so a bridged provider gets exactly
    # the treatment rius's own spans do.
    pipeline = SynchronousMultiSpanProcessor()  # type: ignore[no-untyped-call]
    # Behind a switch shutdown() turns off: this provider may stay the OTel
    # global after shutdown() and keep starting third-party spans.
    provider.add_span_processor(OwnPipeline(pipeline))
    detector: ForeignParentDetector | None = None

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
        # Credential headers are dropped whatever capture_content and mask say,
        # so unlike masking this wrapper is always in the chain.
        exporter = CredentialHeaderDroppingExporter(exporter)
        if not config.capture_content or mask is not None:
            exporter = MaskingSpanExporter(
                exporter, capture_content=config.capture_content, mask=mask
            )
        # OUTSIDE masking, so normalization runs FIRST and masking only has to
        # recognise canonical content keys. Reversed, masking would strip
        # llm.input_messages before it could be mapped and gen_ai.input.messages
        # would arrive empty. Always on: the wire format is not optional.
        # Shared with the exporter: it learns, from end order, which wrapper
        # spans repeat a child's usage, and the exporter acts on it.
        wrapped_calls = WrappedCallTracker()
        compacted_context = CompactedContextProcessor()
        exporter = NormalizingSpanExporter(
            exporter, wrapped_calls=wrapped_calls, compacted_context=compacted_context
        )
        exporter = ResourceAdoptingExporter(exporter, resource)
        # Outermost wrapper so it observes the outcome of the whole chain
        # (masking included); client.flush() reads it for honest delivery
        # reporting.
        # A caller's own exporter or factory has no RIUS_ENDPOINT behind it to
        # blame; the router reports which destination failed, so the warning
        # names that one and hints at the key and endpoint only if it is ours.
        export_health = ExportOutcomeExporter(
            exporter,
            endpoint=config.endpoint if span_exporter is None else None,
            routing=routing,
            workspace_endpoint=config.endpoint if workspace_exporter_factory is None else None,
        )
        batch_processor = BatchSpanProcessor(export_health)
        detector = ForeignParentDetector(foreign_global)
        pipeline.add_span_processor(detector)
        # Ahead of everything that reads a span at start or end: the anthropic
        # SDK's own chat span must reach normalization, the compaction flag
        # and the pending snapshot already demoted, or it counts as a second
        # LLM span next to OpenInference's.
        pipeline.add_span_processor(
            NativeChatSpanDemoter(lambda: instruments_for("anthropic", provider))
        )
        # Everything that writes at on_start (normalization, the compaction
        # flag, the session, user and workspace stamps) is registered BEFORE
        # the pending processor: the pending snapshot is built from the
        # attributes already on the span, so canonical keys, the session id
        # and the workspace route (which decides which destination the
        # snapshot itself goes to) must be there first to ride it.
        pipeline.add_span_processor(NormalizingSpanProcessor())
        pipeline.add_span_processor(compacted_context)
        pipeline.add_span_processor(wrapped_calls)
        pipeline.add_span_processor(SessionSpanProcessor(config.session_id))
        pipeline.add_span_processor(UserSpanProcessor())
        if routing is not None:
            pipeline.add_span_processor(WorkspaceSpanProcessor())
        if config.partial_spans:
            # Pending snapshots ride the SAME batch pipeline as final spans
            # (exporter, retries, masking); see pending.py for the contract.
            pipeline.add_span_processor(
                PendingSpanProcessor(batch_processor, delay=config.partial_spans_delay)
            )
        pipeline.add_span_processor(batch_processor)

    bridged: Bridge | None = None
    if set_global and not config.disabled:
        bridged = _register_global(
            provider, pipeline, sampler, limits, bridge_foreign=config.bridge_foreign_provider
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
        enable_instrumentations(provider, instruments, for_global=set_global)

    # Heartbeat: process-lifetime liveness, independent of trace
    # traffic. The tracker rides the provider as a span processor so payloads
    # can carry the currently-open root trace ids; disabled kills it too.
    sender: HeartbeatSender | None = None
    # Without credentials the managed endpoint rejects every ping, so the
    # default transport would only fail each interval. An injected transport
    # knows its own destination and runs regardless.
    can_authenticate = heartbeat_transport is not None or any(
        key.lower() == "authorization" for key in config.headers
    )
    if config.heartbeat and not config.disabled and not can_authenticate:
        # Silent otherwise: auth supplied only through OTEL_EXPORTER_OTLP_HEADERS
        # reaches the exporter but not this check.
        logger.debug(
            "rius heartbeat not started: no credentials (set api_key or an "
            "Authorization header in headers)"
        )
    inbox: HintInbox | None = None
    hints_off_because = _why_hints_are_off(config, heartbeat_transport, can_authenticate)
    if config.hints and hints_off_because is not None:
        logger.warning("rius hints were requested but are off: %s", hints_off_because)
    if config.heartbeat and not config.disabled and can_authenticate:
        tracker = OpenRootSpanTracker()
        pipeline.add_span_processor(tracker)
        if config.hints and heartbeat_transport is None:
            inbox = HintInbox(
                tracker.open_trace_ids, per_trace=config.hints_per_trace, on_hint=on_hint
            )
        sender = HeartbeatSender(
            url=config.heartbeat_endpoint,
            headers=config.headers,
            interval=config.heartbeat_interval,
            agent_name=config.agent_name,
            instance_id=instance_id,
            instance_secret=instance_secret,
            tracker=tracker,
            transport=heartbeat_transport,
            foreign_parent_spans=None if detector is None else (lambda: detector.count),
            hints=inbox,
        )
        sender.start()

    client = GlassflowClient(
        provider,
        config,
        heartbeat=sender,
        export_health=export_health,
        connectivity_thread=connectivity_thread,
        routing=routing,
        bridge=bridged,
        hints=inbox,
    )
    if set_global:
        _current_client = client
    return client


def _why_hints_are_off(
    config: GlassflowConfig,
    heartbeat_transport: Callable[[dict[str, Any]], None] | None,
    can_authenticate: bool,
) -> str | None:
    """Why hints cannot run with this configuration, or None when they can."""
    if config.disabled:
        return "rius is disabled (the kill switch is on)"
    if not config.heartbeat:
        return "the heartbeat is off, and hints ride on its response"
    if heartbeat_transport is not None:
        return (
            "a custom heartbeat_transport is handed the payload only, so it cannot "
            "authenticate the instance, and hints are delivered only to authenticated pings"
        )
    if not can_authenticate:
        return "no credentials (set api_key or an Authorization header), so no heartbeat runs"
    return None


def _foreign_global_name() -> str | None:
    """``module.Class`` of the global provider when another SDK already set it."""
    existing = trace.get_tracer_provider()
    if isinstance(existing, trace.ProxyTracerProvider) or existing in _own_providers:
        return None
    kind = type(existing)
    return f"{kind.__module__}.{kind.__qualname__}"


def _register_global(
    provider: TracerProvider,
    pipeline: SynchronousMultiSpanProcessor,
    sampler: Sampler,
    limits: SpanLimits,
    *,
    bridge_foreign: bool,
) -> Bridge | None:
    """Claim the OpenTelemetry global, or bridge to whichever provider holds it."""
    trace.set_tracer_provider(provider)
    existing = trace.get_tracer_provider()
    if existing is provider:
        return None
    if existing in _own_providers and isinstance(existing, TracerProvider):
        # A previous init() claimed the write-once global and has since shut
        # down. That provider is ours, not another SDK's: bridging it is what
        # keeps third-party get_tracer() spans flowing across a re-init, so it
        # needs no opt-in and earns no "another SDK" warning.
        logger.debug(
            "the OpenTelemetry global provider belongs to a previous init(); rius "
            "attached this client's span pipeline to it."
        )
        return bridge(existing, pipeline, sampler, limits)
    if bridge_foreign and isinstance(existing, TracerProvider):
        logger.info(
            "the OpenTelemetry global provider was already set (%s); rius attached its "
            "span pipeline to it, so spans started through it are exported to rius too.",
            type(existing).__qualname__,
        )
        return bridge(existing, pipeline, sampler, limits)
    logger.warning(
        "could not register the rius tracer provider as the OpenTelemetry "
        "global (%s is already set, or a previous init() claimed it), and rius is "
        "not attached to it (%s). rius' own helpers (@observe, start_span, "
        "generations) follow this client regardless; third-party code using "
        "opentelemetry.trace.get_tracer() keeps the pre-existing provider, and "
        "LLM spans started inside its spans arrive without their parent. Set "
        "RIUS_BRIDGE_FOREIGN_PROVIDER=true or init(bridge_foreign_provider=True) "
        "to send that provider's spans to Rius as well.",
        type(existing).__qualname__,
        "bridge_foreign_provider is off"
        if not bridge_foreign
        else "it is not an OpenTelemetry SDK TracerProvider",
    )
    return None


def drain_hints(trace_id: str | None = None) -> list[Hint]:
    """Remove and return the global client's hints for a run, oldest first.

    See :meth:`GlassflowClient.drain_hints`. Empty when no global ``init()`` is
    active or hints are off.
    """
    with _lock:
        client = _current_client
    if client is None or client._is_shutdown:
        return []
    return client.drain_hints(trace_id)


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
