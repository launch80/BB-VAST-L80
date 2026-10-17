"""The provisioning contract: a *provider* turns a bare rented box into a live
``/v1`` server, driving it through an *exec transport*.

Two separations keep this flexible:

* **Provider** (vLLM, llama.cpp, or your custom stack) decides *what* to run to
  serve a given model. It only ever talks to the box through :class:`Box`.
* **ExecTransport** decides *how* a command reaches the box. The default is the
  Vast **execute endpoint** (no SSH key needed); SSH is a fallback.

A provider is three idempotent steps: ``prepare`` (make the box ready),
``launch`` (start the server, detached), ``is_ready`` (is ``/v1`` answering?).
"""
from __future__ import annotations

import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime cycle with the vast package
    from bbvastl80.spec import RunSpec


# --- a result from running a command -----------------------------------------
@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def text(self) -> str:
        return self.stdout + (f"\n{self.stderr}" if self.stderr else "")


# --- the box a provider drives ------------------------------------------------
@dataclass
class Box:
    """Everything a provider needs to operate on a rented instance."""

    instance_id: int
    transport: "ExecTransport"
    internal_port: int = 8000
    host_ip: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_user: str = "root"
    ports: dict = field(default_factory=dict)  # str(internal) -> external

    def run(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        return self.transport.run(self, cmd, timeout)

    def run_detached(self, cmd: str, logfile: str = "/tmp/bbvastl80/server.log") -> None:
        self.transport.run_detached(self, cmd, logfile)

    def external_port(self, internal: int):
        v = self.ports.get(str(internal))
        return int(v) if v else None

    def url(self) -> str | None:
        """The externally reachable base URL, if the port mapping is known."""
        ep = self.external_port(self.internal_port)
        if self.host_ip and ep:
            return f"http://{self.host_ip}:{ep}"
        return None


# --- transports ----------------------------------------------------------------
class ExecTransport(ABC):
    """How a command reaches the box. ``run`` is synchronous; ``run_detached``
    starts a long-running process and returns promptly."""

    name = "base"

    @abstractmethod
    def run(self, box: "Box", cmd: str, timeout: float = 120.0) -> ExecResult:
        ...

    @abstractmethod
    def run_detached(self, box: "Box", cmd: str, logfile: str) -> None:
        ...


# Vast merges stdout+stderr on the execute endpoint, so we capture the exit code
# by appending a marker and parsing it back out.
_RC = re.compile(r"__BBRC__(\d+)\s*$")


class VastExecTransport(ExecTransport):
    """Primary transport: the Vast ``PUT /instances/{id}/command/`` endpoint.

    Needs only the API key — no SSH key exchange — which is why it's the default
    for provisioning.
    """

    name = "vast-exec"

    def __init__(self, client):
        self.client = client

    def run(self, box: "Box", cmd: str, timeout: float = 120.0) -> ExecResult:
        wrapped = f"{cmd}\necho \"__BBRC__$?\""
        out = self.client.execute(box.instance_id, wrapped, timeout=timeout) or ""
        m = _RC.search(out)
        code = int(m.group(1)) if m else 0
        body = _RC.sub("", out).rstrip("\n")
        return ExecResult(code, body)

    def run_detached(self, box: "Box", cmd: str, logfile: str) -> None:
        self.client.execute(
            box.instance_id,
            f"mkdir -p $(dirname {logfile}) && nohup {cmd} >{logfile} 2>&1 & echo started",
            timeout=60,
        )


class SSHExecTransport(ExecTransport):
    """Fallback transport: plain ``ssh`` (used when the execute endpoint is
    unavailable, and required for M4's port-forward tunnel)."""

    name = "ssh"

    def __init__(self, identity: str | None = None):
        self.identity = identity

    def _cmd(self, box: "Box", *extra) -> list[str]:
        base = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=15",
        ]
        if self.identity:
            base += ["-i", self.identity]
        host = box.ssh_host or box.host_ip
        if box.ssh_port:
            base += ["-p", str(box.ssh_port)]
        return base + [f"{box.ssh_user}@{host}", *extra]

    def run(self, box: "Box", cmd: str, timeout: float = 120.0) -> ExecResult:
        p = subprocess.run(self._cmd(box, cmd), capture_output=True, text=True,
                           timeout=timeout + 15)
        return ExecResult(p.returncode, p.stdout, p.stderr)

    def run_detached(self, box: "Box", cmd: str, logfile: str) -> None:
        self.run(box, f"mkdir -p $(dirname {logfile}) && nohup {cmd} >{logfile} 2>&1 & echo started",
                 timeout=60)


def make_transport(spec: "RunSpec", client) -> ExecTransport:
    """Pick the command transport. The Vast execute endpoint is the default
    (needs only the API key); an explicit ``ssh.identity`` switches to SSH, which
    M4 also needs for its port-forward tunnel."""
    if spec.ssh and spec.ssh.identity:
        return SSHExecTransport(spec.ssh.identity)
    return VastExecTransport(client)


# --- the provider contract ----------------------------------------------------
class Provider(ABC):
    """A recipe for serving one model from a box. All steps are idempotent so a
    run can be resumed after a crash."""

    name = "base"

    @abstractmethod
    def prepare(self, box: "Box", spec: "RunSpec") -> None:
        """Make the box ready to serve (install deps / pull / stage weights)."""

    @abstractmethod
    def launch(self, box: "Box", spec: "RunSpec") -> None:
        """Start the serving process (detached; ``is_ready`` polls it up)."""

    @abstractmethod
    def is_ready(self, box: "Box", spec: "RunSpec") -> bool:
        """True once the ``/v1`` endpoint answers *inside* the box."""
