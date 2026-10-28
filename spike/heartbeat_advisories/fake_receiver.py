"""Local stand-in for the Rius receiver: heartbeats can answer with advisories.

Run it, point the agent's RIUS_ENDPOINT at it, then queue an advisory by hand:

    uv run python spike/heartbeat_advisories/fake_receiver.py
    curl -X POST localhost:8787/admin/advisory -d '{"text": "..."}'

The advisory goes to the most recent instance that pinged (or to "instance_id"
in the body) on its next heartbeat, then is deleted: the same delete-on-read
the receiver spike does with NATS KV. OTLP traces are accepted and dropped.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787

lock = threading.Lock()
pending: dict[str, list[dict[str, Any]]] = {}
last_instance: str | None = None


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        global last_instance
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path == "/v1/heartbeat":
            ping = json.loads(body)
            instance = ping["instance_id"]
            with lock:
                last_instance = instance
                advisories = (
                    pending.pop(instance, []) if "advisories" in ping.get("accepts", []) else []
                )
            if advisories:
                log(
                    f"heartbeat {instance[:8]} open_traces={ping['open_trace_count']} "
                    f"-> {len(advisories)} advisory"
                )
                self._reply(200, {"advisories": advisories})
            else:
                log(f"heartbeat {instance[:8]} open_traces={ping['open_trace_count']}")
                self._reply(204)
        elif self.path == "/admin/advisory":
            item = json.loads(body)
            with lock:
                instance = item.pop("instance_id", None) or last_instance
                if instance is None:
                    self._reply(409, {"error": "no instance has pinged yet"})
                    return
                item.setdefault("id", str(uuid.uuid4()))
                item.setdefault("code", "manual")
                pending.setdefault(instance, []).append(item)
            log(f"queued advisory {item['id'][:8]} for {instance[:8]}")
            self._reply(202, {"instance_id": instance, **item})
        else:  # /v1/traces and anything else
            self._reply(200)

    def _reply(self, status: int, payload: Any = None) -> None:
        data = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(status)
        if data:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        pass


if __name__ == "__main__":
    log(f"fake receiver on http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
