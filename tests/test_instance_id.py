"""Instance identity: one id per client, on spans AND heartbeats (RIUS-436).

The heartbeat identifies a process lifetime with an ``instance_id``; spans
must carry the same identity as the standard ``service.instance.id``
resource attribute, so the backend can join heartbeats to traces and
distinguish replicas sharing one ``agent_name``. The id belongs to the
client, not the heartbeat feature: it is stamped even with heartbeat off.
"""

from __future__ import annotations

import uuid
from typing import Any

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.semconv import SERVICE_INSTANCE_ID


def _resource_instance_id(client: Any) -> object:
    return client._provider.resource.attributes.get(SERVICE_INSTANCE_ID)  # noqa: SLF001


def test_resource_contains_service_instance_id() -> None:
    client = init(set_global=False, service_name="svc", span_exporter=InMemorySpanExporter())
    try:
        instance_id = _resource_instance_id(client)
        assert isinstance(instance_id, str)
        uuid.UUID(instance_id)  # a valid UUID
    finally:
        client.shutdown()


def test_exported_spans_carry_the_instance_id() -> None:
    exporter = InMemorySpanExporter()
    client = init(set_global=False, service_name="svc", span_exporter=exporter)
    try:
        client.get_tracer().start_span("root").end()
        assert client.flush()
        (span,) = exporter.get_finished_spans()
        assert span.resource.attributes[SERVICE_INSTANCE_ID] == _resource_instance_id(client)
    finally:
        client.shutdown()


def test_heartbeat_instance_id_matches_resource() -> None:
    sent: list[dict[str, Any]] = []
    client = init(
        set_global=False,
        service_name="svc",
        heartbeat=True,
        heartbeat_transport=sent.append,
        span_exporter=InMemorySpanExporter(),
    )
    try:
        client._heartbeat._send_ping()  # noqa: SLF001 — deterministic ping
        assert sent[-1]["instance_id"] == _resource_instance_id(client)
    finally:
        client.shutdown()


def test_instance_id_present_with_heartbeat_disabled() -> None:
    client = init(
        set_global=False,
        service_name="svc",
        heartbeat=False,
        span_exporter=InMemorySpanExporter(),
    )
    try:
        assert isinstance(_resource_instance_id(client), str)
    finally:
        client.shutdown()


def test_instance_id_present_when_disabled() -> None:
    client = init(set_global=False, service_name="svc", disabled=True)
    try:
        assert isinstance(_resource_instance_id(client), str)
    finally:
        client.shutdown()


def test_distinct_clients_get_distinct_ids() -> None:
    a = init(set_global=False, service_name="svc", span_exporter=InMemorySpanExporter())
    b = init(set_global=False, service_name="svc", span_exporter=InMemorySpanExporter())
    try:
        assert _resource_instance_id(a) != _resource_instance_id(b)
    finally:
        a.shutdown()
        b.shutdown()
