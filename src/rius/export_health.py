"""Export outcome tracking.

The OTel ``BatchSpanProcessor`` discards export results by design, so a
misconfigured key or endpoint is invisible unless something else records it.
``ExportOutcomeExporter`` wraps the real exporter to (a) warn once, loudly and
actionably, on the first failed export, and (b) expose ``last_export_failed``
so ``client.flush()`` can report delivery honestly instead of only "queue
drained".

Same reliability contract as the rest of the SDK: never raises into the
export pipeline, never blocks beyond the wrapped exporter itself.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from .workspace import RoutingSpanExporter

logger = logging.getLogger(__name__)

ProbeTransport = Callable[[str, "dict[str, str] | None"], int]

_PROBE_TIMEOUT_SECONDS = 5.0


def _default_probe_send(url: str, headers: dict[str, str] | None) -> int:
    """POST an empty OTLP request; the status code is the diagnosis.

    An empty body is a valid (zero-span) ``ExportTraceServiceRequest``, so a
    healthy backend answers 2xx and a bad key answers 401/403, without a
    single fake span landing anywhere.
    """
    request = urllib.request.Request(
        url,
        data=b"",
        headers={**(headers or {}), "Content-Type": "application/x-protobuf"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_PROBE_TIMEOUT_SECONDS) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def check_connectivity(url: str, headers: dict[str, str] | None, send: ProbeTransport) -> None:
    """Probe the traces endpoint once and log an actionable verdict.

    Never raises: an unusable endpoint is worth exactly one warning, not a
    crashed init. Runs on a background thread (see ``client.init``), so it
    must not touch shared state.
    """
    try:
        status = send(url, headers)
    except Exception as exc:  # noqa: BLE001 - reliability contract: never raise
        logger.warning(
            "cannot reach %s (%s); traces will not be delivered. "
            "Check RIUS_ENDPOINT and network egress.",
            url,
            exc,
        )
        return
    if 200 <= status < 300:
        logger.debug("connectivity check to %s ok (HTTP %s)", url, status)
    elif status in (401, 403):
        logger.warning(
            "%s rejected the SDK's credentials (HTTP %s); traces will not be "
            "delivered. Check RIUS_API_KEY (revoked or mistyped key?).",
            url,
            status,
        )
    else:
        logger.warning(
            "connectivity check to %s returned HTTP %s; traces may not be delivered.",
            url,
            status,
        )


class ExportOutcomeExporter(SpanExporter):
    """Delegate to ``inner`` and remember whether the last export succeeded.

    The first failure logs one WARNING naming its cause; repeats log at DEBUG
    (the heartbeat sender's warn-once discipline). A subsequent success logs
    one INFO and re-arms the warning, so a *new* outage later in the process
    lifetime warns again.

    The cause is what the chain told us. An exception raised inside it (the
    SDK's own wrappers or a user exporter) is logged with its traceback, and
    says nothing about keys or endpoints. A FAILURE result with ``endpoint``
    set comes from the OTLP exporter, so the warning points at the key and
    endpoint; ``endpoint=None`` means the caller passed its own
    ``span_exporter``, and the warning blames that instead.

    With workspace ``routing``, one failing destination fails the whole
    export, so the warning names the destinations the router reports as
    failed: the default one (managed at ``endpoint``, or the caller's
    ``span_exporter`` when it is None) and each failed workspace (managed at
    ``workspace_endpoint``, or the caller's ``workspace_exporter_factory``
    when it is None). Only a managed destination gets the key and endpoint
    hint. A destination that starts failing after the warning warns again,
    naming only itself, so a workspace that never recovers cannot hide it.
    """

    def __init__(
        self,
        inner: SpanExporter,
        *,
        endpoint: str | None,
        routing: RoutingSpanExporter | None = None,
        workspace_endpoint: str | None = None,
    ) -> None:
        self._inner = inner
        self._endpoint = endpoint
        self._routing = routing
        self._workspace_endpoint = workspace_endpoint
        self._warned = False
        # Routed destinations a warning has named since the last success: a
        # destination that starts failing later warns again, since a revoked
        # workspace key may keep the export from ever succeeding.
        self._warned_destinations: frozenset[str | None] = frozenset()
        self.last_export_failed = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        error: Exception | None = None
        try:
            result = self._inner.export(spans)
        except Exception as exc:  # noqa: BLE001 - reliability contract: never raise
            error = exc
            result = SpanExportResult.FAILURE
        if result is SpanExportResult.SUCCESS:
            if self.last_export_failed:
                logger.info("span export%s recovered", self._destination())
            self.last_export_failed = False
            self._warned = False
            self._warned_destinations = frozenset()
            return result
        self.last_export_failed = True
        newly_failed: frozenset[str | None] = frozenset()
        if self._routing is not None and error is None:
            newly_failed = self._routing.last_failed - self._warned_destinations
        if self._warned and not newly_failed:
            logger.debug(
                "span export%s failed again", self._destination(), exc_info=error is not None
            )
            return result
        self._warned = True
        repeats = "Further failures log at DEBUG until an export succeeds."
        if error is not None:
            logger.warning(
                "span export%s failed: the export chain raised %s: %s; traces are not "
                "being delivered. %s",
                self._destination(),
                type(error).__name__,
                error,
                repeats,
                exc_info=error,
            )
        elif self._routing is not None:
            failed = newly_failed
            self._warned_destinations |= failed
            logger.warning(
                "span export failed: %s; their spans are not being delivered. %s%s",
                "; ".join(self._failed_destinations(failed)) or "a destination failed",
                self._routed_hint(failed),
                repeats,
            )
        elif self._endpoint is None:
            logger.warning(
                "span export failed: the configured span_exporter returned FAILURE; "
                "traces are not being delivered. %s",
                repeats,
            )
        else:
            logger.warning(
                "span export to %s failed; traces are not being delivered. "
                "Check RIUS_API_KEY and RIUS_ENDPOINT (the exporter's "
                "own log line above has the HTTP status). %s",
                self._endpoint,
                repeats,
            )
        return result

    def _failed_destinations(self, failed: frozenset[str | None]) -> list[str]:
        named = []
        if None in failed:
            if self._endpoint is None:
                named.append("the configured span_exporter failed")
            else:
                named.append(f"the default destination at {self._endpoint} failed")
        aliases = sorted(alias for alias in failed if alias is not None)
        if aliases:
            plural = "s" if len(aliases) > 1 else ""
            noun = f"workspace{plural}"
            listed = ", ".join(repr(alias) for alias in aliases)
            if self._workspace_endpoint is None:
                named.append(
                    f"the workspace_exporter_factory exporter{plural} for {noun} {listed} failed"
                )
            else:
                named.append(f"{noun} {listed} at {self._workspace_endpoint} failed")
        return named

    def _routed_hint(self, failed: frozenset[str | None]) -> str:
        checks = []
        if None in failed and self._endpoint is not None:
            checks.append("RIUS_API_KEY")
        if any(alias is not None for alias in failed) and self._workspace_endpoint is not None:
            checks.append("each failed workspace's API key")
        if not checks:
            return ""
        return (
            f"Check {', '.join(checks)} and RIUS_ENDPOINT (the OTLP exporter's own log "
            "line above has the HTTP status). "
        )

    def _destination(self) -> str:
        # With a caller's exporter or workspace routing in the chain the
        # endpoint is not the whole destination, so naming it alone would
        # mislead.
        if self._endpoint is None or self._routing is not None:
            return ""
        return f" to {self._endpoint}"

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)
