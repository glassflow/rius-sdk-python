"""Surfacing export misconfiguration and failures.

A missing/wrong key or endpoint must produce a visible, actionable signal
without ever raising into app code or blocking init; the failure mode this
guards against is a user who believes they are instrumented and has no data.
"""

from __future__ import annotations

import logging

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.config import DEFAULT_ENDPOINT


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLASSFLOW_API_KEY", raising=False)
    monkeypatch.delenv("GLASSFLOW_ENDPOINT", raising=False)


def _init_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "set_global": False,
        "service_name": "test-svc",
        # Stubbed by default so no test ever probes a real endpoint; the
        # probe-wiring tests override it with their own recording stubs.
        "connectivity_transport": lambda url, headers: 200,
    }
    kwargs.update(overrides)
    return kwargs


class _FlakyExporter:
    """Scripted exporter: pop the next result per export() call."""

    def __init__(self, script: list[object]) -> None:
        self.script = script
        self.exported: list[object] = []

    def export(self, spans):  # noqa: ANN001, ANN201
        from opentelemetry.sdk.trace.export import SpanExportResult

        self.exported.append(spans)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        assert isinstance(step, SpanExportResult)
        return step

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class TestExportOutcomeExporter:
    def _make(self, script: list[object]):  # noqa: ANN202
        from rius.export_health import ExportOutcomeExporter

        inner = _FlakyExporter(script)
        return ExportOutcomeExporter(inner, endpoint=DEFAULT_ENDPOINT), inner

    def test_first_failure_warns_once_then_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        from opentelemetry.sdk.trace.export import SpanExportResult

        wrapper, _ = self._make([SpanExportResult.FAILURE, SpanExportResult.FAILURE])
        with caplog.at_level(logging.DEBUG, logger="rius.export_health"):
            wrapper.export([])
            wrapper.export([])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "not being delivered" in msg
        assert DEFAULT_ENDPOINT in msg
        assert "GLASSFLOW_API_KEY" in msg

    def test_success_records_no_failure_and_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from opentelemetry.sdk.trace.export import SpanExportResult

        wrapper, _ = self._make([SpanExportResult.SUCCESS])
        with caplog.at_level(logging.DEBUG, logger="rius.export_health"):
            assert wrapper.export([]) is SpanExportResult.SUCCESS
        assert wrapper.last_export_failed is False
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_inner_exception_never_raises_and_counts_as_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from opentelemetry.sdk.trace.export import SpanExportResult

        wrapper, _ = self._make([RuntimeError("boom")])
        with caplog.at_level(logging.DEBUG, logger="rius.export_health"):
            result = wrapper.export([])
        assert result is SpanExportResult.FAILURE
        assert wrapper.last_export_failed is True

    def test_recovery_resets_state_and_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        from opentelemetry.sdk.trace.export import SpanExportResult

        wrapper, _ = self._make([SpanExportResult.FAILURE, SpanExportResult.SUCCESS])
        with caplog.at_level(logging.DEBUG, logger="rius.export_health"):
            wrapper.export([])
            wrapper.export([])
        assert wrapper.last_export_failed is False
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("recovered" in m for m in infos)


class TestHonestFlush:
    def test_flush_returns_false_when_exports_fail(self) -> None:
        from opentelemetry.sdk.trace.export import SpanExportResult

        failing = _FlakyExporter([SpanExportResult.FAILURE] * 10)
        client = init(**_init_kwargs(span_exporter=failing))  # type: ignore[arg-type]
        with client.get_tracer().start_as_current_span("doomed"):
            pass
        assert client.flush() is False
        client.shutdown()

    def test_flush_returns_true_when_exports_succeed(self) -> None:
        exporter = InMemorySpanExporter()
        client = init(**_init_kwargs(span_exporter=exporter))
        with client.get_tracer().start_as_current_span("fine"):
            pass
        assert client.flush() is True
        client.shutdown()

    def test_flush_true_after_recovery(self) -> None:
        from opentelemetry.sdk.trace.export import SpanExportResult

        flaky = _FlakyExporter([SpanExportResult.FAILURE, SpanExportResult.SUCCESS])
        client = init(**_init_kwargs(span_exporter=flaky))  # type: ignore[arg-type]
        with client.get_tracer().start_as_current_span("first"):
            pass
        assert client.flush() is False
        with client.get_tracer().start_as_current_span("second"):
            pass
        assert client.flush() is True
        client.shutdown()


class TestConnectivityCheck:
    def _check(self, send) -> None:  # noqa: ANN001
        from rius.export_health import check_connectivity

        check_connectivity(f"{DEFAULT_ENDPOINT}/v1/traces", {"Authorization": "Bearer x"}, send)

    def test_auth_rejection_warns_with_status_and_key_hint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rius.export_health"):
            self._check(lambda url, headers: 401)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "401" in warnings[0]
        assert "GLASSFLOW_API_KEY" in warnings[0]

    def test_success_stays_quiet(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="rius.export_health"):
            self._check(lambda url, headers: 200)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_unreachable_endpoint_warns_with_endpoint_hint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def send(url: str, headers: object) -> int:
            raise OSError("nodename nor servname provided")

        with caplog.at_level(logging.WARNING, logger="rius.export_health"):
            self._check(send)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "GLASSFLOW_ENDPOINT" in warnings[0]

    def test_other_status_warns_generically(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="rius.export_health"):
            self._check(lambda url, headers: 500)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "500" in warnings[0]

    def test_never_raises(self) -> None:
        def send(url: str, headers: object) -> int:
            raise RuntimeError("total transport meltdown")

        self._check(send)  # must not raise


class TestConnectivityProbeWiring:
    def test_init_probes_managed_endpoint_in_background(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        seen: list[str] = []

        def send(url: str, headers: object) -> int:
            seen.append(url)
            return 401

        with caplog.at_level(logging.WARNING, logger="rius.export_health"):
            client = init(
                **_init_kwargs(api_key="gf_revoked", connectivity_transport=send)  # type: ignore[arg-type]
            )
            thread = client._connectivity_thread
            assert thread is not None
            thread.join(timeout=5)
        client.shutdown()
        assert seen == [f"{DEFAULT_ENDPOINT}/v1/traces"]
        assert any("401" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)

    def test_probe_skipped_for_custom_exporter(self) -> None:
        seen: list[str] = []

        def send(url: str, headers: object) -> int:
            seen.append(url)
            return 200

        client = init(
            **_init_kwargs(  # type: ignore[arg-type]
                span_exporter=InMemorySpanExporter(), connectivity_transport=send
            )
        )
        assert client._connectivity_thread is None
        client.shutdown()
        assert seen == []

    def test_probe_skipped_when_disabled(self) -> None:
        client = init(
            **_init_kwargs(disabled=True, connectivity_transport=lambda u, h: 200)  # type: ignore[arg-type]
        )
        assert client._connectivity_thread is None
        client.shutdown()

    def test_probe_skipped_when_key_already_known_missing(self) -> None:
        # The missing-key warning already made the diagnosis; probing would
        # just produce a second warning for the same misconfiguration.
        seen: list[str] = []
        client = init(
            **_init_kwargs(connectivity_transport=lambda u, h: seen.append(u) or 401)  # type: ignore[arg-type]
        )
        assert client._connectivity_thread is None
        client.shutdown()
        assert seen == []


class TestMissingKeyInitWarning:
    def test_managed_endpoint_without_key_warns_actionably(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rius.client"):
            client = init(**_init_kwargs())
        client.shutdown()
        messages = [r.getMessage() for r in caplog.records]
        assert any("GLASSFLOW_API_KEY" in m for m in messages), messages
        assert any(DEFAULT_ENDPOINT in m for m in messages), messages

    def test_custom_endpoint_without_key_stays_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Pointing at your own collector without auth is a legitimate setup.
        with caplog.at_level(logging.WARNING, logger="rius.client"):
            client = init(**_init_kwargs(endpoint="http://localhost:4318"))
        client.shutdown()
        assert not any("GLASSFLOW_API_KEY" in r.getMessage() for r in caplog.records)

    def test_managed_endpoint_with_key_stays_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="rius.client"):
            client = init(**_init_kwargs(api_key="gf_test_key"))
        client.shutdown()
        assert not any("GLASSFLOW_API_KEY" in r.getMessage() for r in caplog.records)

    def test_custom_auth_header_counts_as_credentials(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="rius.client"):
            client = init(**_init_kwargs(headers={"authorization": "Bearer x"}))
        client.shutdown()
        assert not any("GLASSFLOW_API_KEY" in r.getMessage() for r in caplog.records)

    def test_disabled_stays_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="rius.client"):
            client = init(**_init_kwargs(disabled=True))
        client.shutdown()
        assert not any("GLASSFLOW_API_KEY" in r.getMessage() for r in caplog.records)

    def test_custom_span_exporter_stays_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        # An injected exporter does not post to the managed endpoint at all.
        with caplog.at_level(logging.WARNING, logger="rius.client"):
            client = init(**_init_kwargs(span_exporter=InMemorySpanExporter()))
        client.shutdown()
        assert not any("GLASSFLOW_API_KEY" in r.getMessage() for r in caplog.records)
