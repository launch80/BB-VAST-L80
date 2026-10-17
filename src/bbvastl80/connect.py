"""M4: make the box's ``/v1`` reachable from *this* machine and prove it.

Two modes, chosen by ``connect.mode``:

* **direct** — Vast maps the container's serving port to a random public port;
  if that mapping exists and answers from here, the endpoint is
  ``http://<public_ip>:<external_port>/v1``.
* **tunnel** — a background ``ssh -N -L <local>:localhost:<port>`` over the
  same SSH access provisioning uses; the endpoint becomes
  ``http://localhost:<local_port>/v1``. Works from any network and keeps the
  box's address off the wire for BetterBench/L80 (they only ever see
  localhost).

``auto`` tries direct first and falls back to the tunnel. Either way the
endpoint is **verified** (``GET /v1/models`` must list the model id) before
anyone benchmarks against it — the "404 endpoint produces a results file of
zeros" trap is closed here, not downstream.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass

import httpx

from bbvastl80 import log
from bbvastl80.providers.base import Box, SSHExecTransport


class ConnectError(Exception):
    """No reachable, verified ``/v1`` could be established."""


@dataclass
class Connection:
    endpoint: str              # always ends in /v1
    mode: str                  # direct | tunnel
    tunnel_pid: int | None = None

    def close(self) -> None:
        close_tunnel(self.tunnel_pid)
        self.tunnel_pid = None


def models_at(endpoint: str, timeout: float = 10.0) -> list[str]:
    """Model ids served at ``endpoint`` (an OpenAI-compatible ``/v1`` base)."""
    r = httpx.get(endpoint.rstrip("/") + "/models", timeout=timeout)
    r.raise_for_status()
    data = r.json()
    items = data.get("data", data) if isinstance(data, dict) else data
    return [m.get("id") for m in items if isinstance(m, dict)]


def verify_v1(endpoint: str, model_id: str, *, timeout_s: float = 60.0,
              poll_s: float = 3.0) -> None:
    """Raise :class:`ConnectError` unless ``endpoint`` lists ``model_id``."""
    if not endpoint.rstrip("/").endswith("/v1"):
        raise ConnectError(f"endpoint must end in /v1 (got {endpoint!r})")
    deadline = time.monotonic() + timeout_s
    last = ""
    while True:
        try:
            ids = models_at(endpoint)
            if model_id in ids:
                return
            last = f"serves {ids!r}, not {model_id!r}"
        except (httpx.HTTPError, ValueError) as e:
            last = str(e)[:120]
        if time.monotonic() >= deadline:
            raise ConnectError(f"{endpoint} did not verify: {last}")
        time.sleep(poll_s)


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def open_tunnel(box: Box, local_port: int, *, wait_s: float = 30.0) -> subprocess.Popen:
    """Start ``ssh -N -L`` in the background and wait for the local port."""
    if not (box.ssh_host and box.ssh_port):
        raise ConnectError(f"instance {box.instance_id} has no SSH host/port yet")
    transport = box.transport if isinstance(box.transport, SSHExecTransport) else SSHExecTransport()
    argv = transport._cmd(
        box, "-N", "-o", "ExitOnForwardFailure=yes",
        "-L", f"{local_port}:localhost:{box.internal_port}",
    )
    # ssh puts the options before user@host; move the forward flags there too.
    user_host = f"{box.ssh_user}@{box.ssh_host}"
    argv = [a for a in argv if a != user_host] + [user_host]
    log.detail("tunnel: " + " ".join(argv))
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            err = (proc.stderr.read() if proc.stderr else "").strip()
            raise ConnectError(f"ssh tunnel exited (rc={proc.returncode}): {err[:200]}")
        if _port_open("127.0.0.1", local_port):
            return proc
        time.sleep(0.5)
    proc.kill()
    raise ConnectError(f"ssh tunnel did not open localhost:{local_port} within {wait_s:.0f}s")


def close_tunnel(pid: int | None) -> None:
    if not pid:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def connect(box: Box, spec, *, verify_timeout_s: float = 60.0) -> Connection:
    """Resolve + verify an endpoint for ``box`` per ``spec.connect``."""
    mode = spec.connect.mode
    model_id = spec.serving.resolved_model_id(spec.model)
    errors: list[str] = []

    if mode in ("auto", "direct"):
        url = box.url()
        if url:
            ep = f"{url}/v1"
            try:
                verify_v1(ep, model_id, timeout_s=verify_timeout_s if mode == "direct" else 20)
                log.success(f"direct endpoint verified: {ep}")
                return Connection(ep, "direct")
            except ConnectError as e:
                errors.append(f"direct: {e}")
        else:
            errors.append("direct: the serving port has no public mapping")
        if mode == "direct":
            raise ConnectError("; ".join(errors))
        log.detail(errors[-1] + " — falling back to an SSH tunnel")

    local = spec.connect.local_port
    if _port_open("127.0.0.1", local, timeout=0.5):
        # Something already listens locally (a stale tunnel?). Verify it or bail.
        ep = f"http://localhost:{local}/v1"
        try:
            verify_v1(ep, model_id, timeout_s=10)
            log.warn(f"localhost:{local} already forwards to the model — reusing it")
            return Connection(ep, "tunnel")
        except ConnectError:
            raise ConnectError(f"localhost:{local} is busy with something else; "
                               "set connect.local_port") from None
    proc = open_tunnel(box, local)
    ep = f"http://localhost:{local}/v1"
    try:
        verify_v1(ep, model_id, timeout_s=verify_timeout_s)
    except ConnectError:
        close_tunnel(proc.pid)
        raise
    log.success(f"tunnel endpoint verified: {ep} (ssh pid {proc.pid})")
    return Connection(ep, "tunnel", tunnel_pid=proc.pid)
