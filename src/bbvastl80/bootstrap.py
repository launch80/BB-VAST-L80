"""M3 orchestrator: take an already-running rented box to a live ``/v1`` server.

This is the "offload a GPU" primitive minus the benchmark. ``provision`` builds
a :class:`Box` handle over the instance, then runs the resolved provider's
``prepare`` → ``launch`` → (poll) ``is_ready``. Readiness is checked *inside* the
box (``curl localhost:PORT``) so it doesn't depend on the external port mapping;
making the port reachable from this machine is M4's job.
"""
from __future__ import annotations

import time

from bbvastl80 import log
from bbvastl80.providers import Box, make_transport, resolve_provider
from bbvastl80.spec import RunSpec
from bbvastl80.vast.instance import ProvisionError
from bbvastl80.vast.models import Instance

SERVER_LOG = "/tmp/bbvastl80/server.log"


def build_box(client, inst: Instance, spec: RunSpec) -> Box:
    """Bind a transport + connection facts to a running instance."""
    transport = make_transport(spec, client)
    ports: dict = {}
    for p in (spec.serving.port, 22):
        ep = inst.external_port(p)
        if ep:
            ports[str(p)] = ep
    return Box(
        instance_id=inst.id,
        transport=transport,
        internal_port=spec.serving.port,
        host_ip=inst.host_ip,
        ssh_host=inst.ssh_host,
        ssh_port=inst.ssh_port,
        ssh_user=spec.ssh.user,
        ports=ports,
    )


def provision(client, inst: Instance, spec: RunSpec, *, ready_timeout_min: float | None = None) -> Box:
    """Drive the box to a live ``/v1``. Idempotent; safe to re-run on a resume."""
    provider = resolve_provider(spec)
    box = build_box(client, inst, spec)
    log.info(f"provisioning with '{provider.name}' on instance {inst.id} "
             f"(transport: {box.transport.name})")
    provider.prepare(box, spec)
    provider.launch(box, spec)
    _wait_ready(box, spec, provider, ready_timeout_min or spec.budget.max_provision_min)
    log.success(f"'{provider.name}' is serving '{spec.model.id}' at /v1 inside the box")
    return box


def _wait_ready(box: Box, spec: RunSpec, provider, timeout_min: float) -> None:
    deadline = time.monotonic() + timeout_min * 60
    start = time.monotonic()
    while time.monotonic() < deadline:
        try:
            if provider.is_ready(box, spec):
                log.success(f"server ready in {time.monotonic() - start:.0f}s")
                return
        except Exception as e:  # a transient exec blip shouldn't kill the poll
            log.detail(f"readiness probe blip: {str(e)[:80]}")
        time.sleep(5)
    try:
        r = box.run(f"tail -n 40 {SERVER_LOG} 2>/dev/null", timeout=20)
        if r.stdout:
            log.detail("server log tail:\n" + r.stdout)
    except Exception:
        pass
    raise ProvisionError(
        f"server not ready after {timeout_min:.0f} min; inspect the box with "
        f"`bbvastl80 logs --instance {box.instance_id}`")
