#!/usr/bin/env python3
"""Reproduce RIUS-1070: LLM spans exported with a parent rius never receives.

When another SDK registers the OpenTelemetry global tracer provider before
``rius.init()``, rius keeps a private provider (the global is write-once) and
binds its instrumentors to it. Span CONTEXT, however, is process-wide: an LLM
call made inside a span of the foreign provider takes that span as parent. The
LLM span reaches rius, its parent goes to the foreign provider's exporter, and
the trace rius receives is flat and rootless with exactly one missing parent.

Each case runs in its own subprocess, because the global provider can be set
only once per process:

- ``control``  rius alone; the customer's job spans use the global = rius.
- ``foreign``  a plain ``TracerProvider`` claims the global first.
- ``langfuse`` Langfuse v3 (OTel-based) is constructed first; it claims the
  global itself when none is set. Skipped when ``langfuse`` is not importable.

OpenAI and Anthropic are called through their real client libraries over an
``httpx.MockTransport``, so the bundled instrumentors produce real LLM spans and
nothing touches the network (Langfuse points at a closed local port).

Run (the Langfuse case needs the extra package; it is not a dev dependency):

    uv run --with 'langfuse>=3,<4' python scripts/repro_rius_1070.py

Exits 0 when every case is OK, 1 otherwise (a single ``--case`` run exits 3
when ORPHANED).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

JOBS = 5
LLM_CALLS_PER_JOB = 3
CASES = ("control", "foreign", "langfuse")
NO_PARENT = 0
EXIT_ORPHANED = 3  # distinct from 1, which an uncaught exception also yields


# --------------------------------------------------------------------------- analysis


@dataclass
class TraceReport:
    trace_id: str
    spans: int
    has_root: bool
    missing_parent_ids: list[str]
    orphan_parent_is_remote: list[bool]


@dataclass
class ExportReport:
    total_spans: int
    spans_with_parent: int
    parents_resolved: int
    traces: list[TraceReport] = field(default_factory=list)

    @property
    def orphaned(self) -> bool:
        return any(t.missing_parent_ids or not t.has_root for t in self.traces)

    @property
    def verdict(self) -> str:
        return "ORPHANED" if self.orphaned else "OK"


def analyse(spans: Sequence[Any]) -> ExportReport:
    """Check that every exported span's parent is in the exported set.

    Works on anything shaped like an OTel ``ReadableSpan`` (``.context`` and
    ``.parent`` span contexts), so it is independent of the exporter used.
    """
    known = {(s.context.trace_id, s.context.span_id) for s in spans}
    by_trace: dict[int, list[Any]] = defaultdict(list)
    for span in spans:
        by_trace[span.context.trace_id].append(span)

    with_parent = [s for s in spans if s.parent is not None and s.parent.span_id != NO_PARENT]
    resolved = [s for s in with_parent if (s.parent.trace_id, s.parent.span_id) in known]
    report = ExportReport(len(spans), len(with_parent), len(resolved))
    for trace_id, members in sorted(by_trace.items()):
        orphans = [s for s in members if s in with_parent and s not in resolved]
        report.traces.append(
            TraceReport(
                trace_id=f"{trace_id:032x}",
                spans=len(members),
                has_root=any(s.parent is None or s.parent.span_id == NO_PARENT for s in members),
                missing_parent_ids=sorted({f"{s.parent.span_id:016x}" for s in orphans}),
                orphan_parent_is_remote=sorted({s.parent.is_remote for s in orphans}),
            )
        )
    return report


def print_report(title: str, report: ExportReport) -> None:
    print(f"  {title}")
    print(f"    total spans:               {report.total_spans}")
    print(f"    spans with a parent id:    {report.spans_with_parent}")
    print(f"    parents resolved in set:   {report.parents_resolved}")
    print(f"    traces:                    {len(report.traces)}")
    for t in report.traces:
        missing = ",".join(t.missing_parent_ids) or "-"
        remote = ",".join(str(r) for r in t.orphan_parent_is_remote) or "-"
        print(
            f"    trace {t.trace_id[:12]}  spans={t.spans}  root={t.has_root!s:5}  "
            f"missing_parents={len(t.missing_parent_ids)} [{missing}]  "
            f"orphan_parent_is_remote={remote}"
        )


def print_foreign(spans: Sequence[Any]) -> None:
    print(f"  foreign provider received {len(spans)} span(s):")
    for s in spans:
        print(f"    {s.name:<8} trace {s.context.trace_id:032x}  span_id={s.context.span_id:016x}")


# --------------------------------------------------------------------------- mocked LLMs


_OPENAI_BODY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
_ANTHROPIC_BODY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


def _mock_client(httpx_module: Any, body: dict[str, Any]) -> Any:
    """An httpx(-compatible) client that answers every request with ``body``."""
    transport = httpx_module.MockTransport(lambda _: httpx_module.Response(200, json=body))
    return httpx_module.Client(transport=transport)


def _llm_calls() -> list[Callable[[], object]]:
    import anthropic
    import httpx
    import openai

    # anthropic>=1.5 is built on httpx2 and rejects a plain httpx client.
    anthropic_httpx: ModuleType
    try:
        import httpx2 as anthropic_httpx
    except ImportError:
        anthropic_httpx = httpx
    oai = openai.OpenAI(
        api_key="test",
        base_url="http://llm.invalid/v1",
        http_client=_mock_client(httpx, _OPENAI_BODY),
    )
    ant = anthropic.Anthropic(
        api_key="test",
        base_url="http://llm.invalid",
        http_client=_mock_client(anthropic_httpx, _ANTHROPIC_BODY),
    )
    prompt = [{"role": "user", "content": "hi"}]

    def call_openai() -> object:
        return oai.chat.completions.create(model="gpt-4o-mini", messages=prompt)  # type: ignore[arg-type]

    def call_anthropic() -> object:
        return ant.messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=prompt)  # type: ignore[arg-type]

    return [call_openai, call_anthropic]


def run_jobs(start_job: Callable[[int], AbstractContextManager[object]]) -> None:
    """The customer's workload: each job span wraps a few LLM calls."""
    calls = _llm_calls()
    for job in range(JOBS):
        with start_job(job):
            for n in range(LLM_CALLS_PER_JOB):
                calls[n % len(calls)]()


# --------------------------------------------------------------------------- cases


def _init_rius(exporter: Any) -> Any:
    import rius

    return rius.init(
        endpoint="http://rius.invalid",
        api_key="test",
        service_name="rius-1070-repro",
        span_exporter=exporter,
        instruments=["openai", "anthropic"],
        heartbeat=False,
        partial_spans=False,
    )


def _global_job(job: int) -> AbstractContextManager[object]:
    from opentelemetry import trace

    return trace.get_tracer("customer.jobs").start_as_current_span(f"job-{job}")


def run_case(case: str) -> int:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    logging.basicConfig(level=logging.WARNING, format="  [log %(name)s] %(message)s")
    # Langfuse exports to a closed port; its retries are noise, not evidence.
    logging.getLogger("langfuse").setLevel(logging.CRITICAL)
    logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.CRITICAL)
    foreign_exporter: InMemorySpanExporter | None = None
    start_job: Callable[[int], AbstractContextManager[object]] = _global_job

    if case == "foreign":
        foreign_exporter = InMemorySpanExporter()
        foreign = TracerProvider()
        foreign.add_span_processor(SimpleSpanProcessor(foreign_exporter))
        trace.set_tracer_provider(foreign)
    elif case == "langfuse":
        try:
            from langfuse import Langfuse
        except ImportError:
            print("  SKIPPED: langfuse not importable (run with `uv run --with 'langfuse>=3,<4'`)")
            return 0
        from importlib.metadata import version

        lf = Langfuse(public_key="pk-lf-x", secret_key="sk-lf-x", host="http://127.0.0.1:9")
        global_provider = trace.get_tracer_provider()
        print(f"  langfuse {version('langfuse')}; global provider after Langfuse():")
        print(f"    {type(global_provider).__module__}.{type(global_provider).__name__}")
        processors = global_provider._active_span_processor._span_processors  # type: ignore[attr-defined]
        print(f"    processors: {[type(p).__name__ for p in processors]}")
        # Observe what Langfuse's provider receives, next to its own exporter.
        foreign_exporter = InMemorySpanExporter()
        global_provider.add_span_processor(SimpleSpanProcessor(foreign_exporter))  # type: ignore[attr-defined]

        def start_job(job: int) -> AbstractContextManager[object]:
            span: AbstractContextManager[object] = lf.start_as_current_span(name=f"job-{job}")
            return span

    rius_exporter = InMemorySpanExporter()
    client = _init_rius(rius_exporter)
    run_jobs(start_job)
    client.flush()

    report = analyse(rius_exporter.get_finished_spans())
    print_report("spans rius exported:", report)
    if foreign_exporter is not None:
        print_foreign(foreign_exporter.get_finished_spans())
    print(f"  VERDICT[{case}]: {report.verdict}")
    print("RESULT " + json.dumps({"case": case, "verdict": report.verdict}))
    client.shutdown()
    return EXIT_ORPHANED if report.orphaned else 0


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=CASES, help="run one case in this process")
    args = parser.parse_args(argv)
    if args.case:
        return run_case(args.case)

    verdicts: dict[str, int] = {}
    for case in CASES:
        print(f"=== case: {case}", flush=True)
        verdicts[case] = subprocess.run([sys.executable, __file__, "--case", case]).returncode
    print("=== summary")
    for case, code in verdicts.items():
        label = {0: "OK", EXIT_ORPHANED: "ORPHANED"}.get(code, f"ERROR(exit {code})")
        print(f"  {case:<9} {label}")
    return 0 if all(code == 0 for code in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
