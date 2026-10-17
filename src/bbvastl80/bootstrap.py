"""M3 orchestrator: take a rented box to a live ``/v1`` server.

Two delivery paths for the *same* provider-rendered bootstrap script:

* **onstart** (default) — ``render_bootstrap`` produces the script at rent
  time and it ships as the instance's ``onstart``, so Vast runs it on first
  boot. The box is usually already pulling weights by the time it reports
  ``running``.
* **ssh** — ``provision`` pushes the script over SSH and runs it detached.
  Because the script is idempotent (exits if ``/v1`` already answers or a
  launch is in flight), ``provision`` *always* does this, which also covers a
  box rented with ``bootstrap: ssh``, a resume after a crash, or an onstart
  that never ran.

Readiness is checked *inside* the box (``curl localhost:PORT``) so it doesn't
depend on the external port mapping; making the port reachable from this
machine is M4's job.
"""
from __future__ import annotations

import time

from bbvastl80 import log
from bbvastl80.providers import Box, make_transport, resolve_provider
from bbvastl80.providers.base import BOOTSTRAP_LOG, BOOTSTRAP_PATH, SERVER_LOG, STATUS_FILE
from bbvastl80.spec import RunSpec
from bbvastl80.vast.instance import ProvisionError
from bbvastl80.vast.models import Instance

SSH_WAIT_MIN = 5.0


def render_bootstrap(spec: RunSpec) -> str | None:
    """The onstart payload for a rental, or None when the spec opts out."""
    if spec.instance.bootstrap != "onstart":
        return None
    return resolve_provider(spec).script(spec, extra_onstart=spec.instance.onstart)


def build_box(client, inst: Instance, spec: RunSpec) -> Box:
    """Bind an SSH transport + connection facts to a running instance."""
    transport = make_transport(spec, client)
    ports: dict = {}
    for p in (spec.serving.port, 22):
        ep = inst.external_port(p)
        if ep:
            ports[str(p)] = ep
    target = inst.ssh_target()
    return Box(
        instance_id=inst.id,
        transport=transport,
        internal_port=spec.serving.port,
        host_ip=inst.host_ip,
        ssh_host=target[0] if target else None,
        ssh_port=target[1] if target else None,
        ssh_user=spec.ssh.user,
        ports=ports,
    )


def wait_for_ssh(box: Box, timeout_min: float = SSH_WAIT_MIN, poll_sec: float = 10.0) -> None:
    """A box reports ``running`` before sshd accepts our key; wait for it."""
    if not box.ssh_host:
        raise ProvisionError(
            f"instance {box.instance_id} has no SSH host/port yet — "
            "wait a little and retry `bbvastl80 provision`")
    deadline = time.monotonic() + timeout_min * 60
    last = ""
    while time.monotonic() < deadline:
        r = box.run("true", timeout=20)
        if r.ok:
            return
        last = (r.stderr or r.stdout).strip().splitlines()
        last = last[-1] if last else f"rc={r.exit_code}"
        log.detail(f"ssh not ready ({last[:80]}) …")
        time.sleep(poll_sec)
    raise ProvisionError(
        f"could not SSH into {box.ssh_user}@{box.ssh_host}:{box.ssh_port} within "
        f"{timeout_min:.0f} min (last: {last[:120]}). Is your public key registered? "
        "`bbvastl80 doctor` checks; `bbvastl80 ssh --instance ID` prints the command."
    )


def provision(client, inst: Instance, spec: RunSpec, *,
              ready_timeout_min: float | None = None) -> Box:
    """Drive the box to a live ``/v1``. Idempotent; safe to re-run on a resume."""
    provider = resolve_provider(spec)
    box = build_box(client, inst, spec)
    log.info(f"provisioning with '{provider.name}' on instance {inst.id} "
             f"(ssh {box.ssh_user}@{box.ssh_host}:{box.ssh_port})")
    wait_for_ssh(box)
    if provider.is_ready(box, spec):
        log.success(f"'{provider.name}' is already serving '{spec.model.id}' at /v1")
        return box
    script = provider.script(spec, extra_onstart=spec.instance.onstart)
    r = box.put_file(script, BOOTSTRAP_PATH)
    if not r.ok:
        raise ProvisionError(f"could not write {BOOTSTRAP_PATH} on the box:\n{r.text()}")
    # The script backgrounds the server itself and returns once it's launched
    # (or exits non-zero on a prepare failure), so run it synchronously but
    # under nohup so a dropped SSH session can't kill a long pip install.
    r = box.run(f"nohup bash {BOOTSTRAP_PATH} </dev/null >/dev/null 2>&1; "
                f"cat {STATUS_FILE} 2>/dev/null", timeout=3600)
    status = (r.stdout or "").strip()
    log.detail(f"bootstrap status: {status or '?'}")
    if status.startswith("prepare failed"):
        _dump_logs(box)
        raise ProvisionError(f"bootstrap {status}; see the log tail above")
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
            st = box.run(f"cat {STATUS_FILE} 2>/dev/null", timeout=15).stdout.strip()
            if st.startswith("prepare failed"):
                _dump_logs(box)
                raise ProvisionError(f"bootstrap {st}")
            log.detail(f"waiting for /v1 ({st or 'starting'}) …")
        except ProvisionError:
            raise
        except Exception as e:  # a transient exec blip shouldn't kill the poll
            log.detail(f"readiness probe blip: {str(e)[:80]}")
        time.sleep(10)
    _dump_logs(box)
    raise ProvisionError(
        f"server not ready after {timeout_min:.0f} min; inspect with "
        f"`bbvastl80 logs --instance {box.instance_id}`")


def _dump_logs(box: Box, n: int = 40) -> None:
    try:
        r = box.run(f"echo '--- {BOOTSTRAP_LOG}'; tail -n {n} {BOOTSTRAP_LOG} 2>/dev/null; "
                    f"echo '--- {SERVER_LOG}'; tail -n {n} {SERVER_LOG} 2>/dev/null", timeout=20)
        if r.stdout:
            log.detail(r.stdout)
    except Exception:
        pass
