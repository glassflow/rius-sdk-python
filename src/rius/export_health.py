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

    The first failure logs one WARNING with the endpoint and the fix hints;
    repeats log at DEBUG (the heartbeat sender's warn-once discipline). A
    subsequent success logs one INFO and re-arms the warning, so a *new*
    outage later in the process lifetime warns again.
    """

    def __init__(self, inner: SpanExporter, *, endpoint: str) -> None:
        self._inner = inner
        self._endpoint = endpoint
        self._warned = False
        self.last_export_failed = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            result = self._inner.export(spans)
        except Exception:  # noqa: BLE001 - reliability contract: never raise
            logger.debug("span exporter raised", exc_info=True)
            result = SpanExportResult.FAILURE
        if result is SpanExportResult.SUCCESS:
            if self.last_export_failed:
                logger.info("span export to %s recovered", self._endpoint)
            self.last_export_failed = False
            self._warned = False
            return result
        self.last_export_failed = True
        if not self._warned:
            self._warned = True
            logger.warning(
                "span export to %s failed; traces are not being delivered. "
                "Check RIUS_API_KEY and RIUS_ENDPOINT (the exporter's "
                "own log line above has the HTTP status). Further failures log "
                "at DEBUG until an export succeeds.",
                self._endpoint,
            )
        else:
            logger.debug("span export to %s failed again", self._endpoint)
        return result

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)
