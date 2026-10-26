"""Shared plumbing for the instrumentor stopgaps.

``langchain_span_context``, ``litellm_stream_context`` and
``llama_index_span_context`` each patch a third-party instrumentor so its span
is current while the provider call runs. They share the pieces here: the
marker that records the original on a wrapper, the owner check for a context
copied to another thread or task, making a span current without ending it,
and gating on the instrumentor releases that were checked.
"""

from __future__ import annotations

import asyncio
import re
import threading
from typing import Any

from opentelemetry import trace

#: The attribute a wrapper keeps its original under, so re-installing unwraps
#: first instead of stacking, and uninstalling restores the original.
MARK = "__rius_span_context__"


def owner() -> tuple[int, Any]:
    """The thread and asyncio task running now.

    A context copied elsewhere (an executor thread, a child task) carries the
    same context variables, and an attach there would land in a context the
    code that made it cannot detach from. Comparing owners tells them apart.
    """
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), task


def needs_context(span: Any) -> bool:
    """Whether ``span`` exists and is not already the current span.

    None: no span was started for this call. Current already: the instrumentor
    made it current itself (a fixed release), so there is nothing to do. The
    spans are compared by their context, not identity: an instrumentor may make
    a proxy of its span current (llama-index 4.6.1 does).
    """
    if span is None:
        return False
    current = trace.get_current_span()
    return current is not span and current.get_span_context() != span.get_span_context()


def under(span: Any) -> Any:
    """Make ``span`` current for a block, leaving its end and status alone."""
    return trace.use_span(
        span, end_on_exit=False, record_exception=False, set_status_on_exception=False
    )


_VERSION = re.compile(r"v?(\d+(?:\.\d+)*)(.*)")
_PRE_RELEASE = re.compile(r"[-_.]?(a|alpha|b|beta|c|rc|pre|preview|dev)\d*.*")


def version_key(text: str) -> tuple[tuple[int, ...], int] | None:
    """A sortable key for a release string, or None if it doesn't parse.

    The release numbers, without trailing zeros, then -1 for a pre- or dev
    release, 0 for the release itself, and 1 for anything after it (a
    post-release, a local version, an unknown suffix). So ``4.6.0rc1`` sorts
    before ``4.6.0`` and ``4.6.0.post1`` after it.
    """
    match = _VERSION.fullmatch(text.strip().lower())
    if match is None:
        return None
    release = [int(part) for part in match.group(1).split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    suffix = match.group(2)
    if not suffix:
        phase = 0
    elif _PRE_RELEASE.fullmatch(suffix):
        phase = -1
    else:
        phase = 1
    return tuple(release), phase


def verified(installed: str, versions: tuple[str, str]) -> bool:
    """Whether ``installed`` lies within the checked ``(low, high)`` releases."""
    key = version_key(installed)
    low, high = (version_key(v) for v in versions)
    return key is not None and low is not None and high is not None and low <= key <= high
