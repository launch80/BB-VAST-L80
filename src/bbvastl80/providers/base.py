"""The provisioning contract: a *provider* turns a bare rented box into a live
``/v1`` server, driving it through an *exec transport*.

Two separations keep this flexible:

* **Provider** (vLLM, llama.cpp, or your custom stack) decides *what* to run to
  serve a given model. It describes itself as a list of ``prepare`` commands
  and one ``launch`` command, which the base class renders into a single
  idempotent **bootstrap script**. That script is delivered two ways: as the
  instance's ``onstart`` at rent time (so the box starts serving on first boot,
  with no transport at all) and/or pushed over SSH by ``provision``.
* **ExecTransport** decides *how* an ad-hoc command reaches the box. **SSH is
  the only real transport.** The Vast "execute" endpoint
  (``PUT /instances/command/{id}/``) is *constrained* — it runs ``ls``, ``rm``
  and ``du`` and nothing else (see the official ``vastai execute`` help), so it
  can never install an engine, start a server or probe a port. It is kept only
  as a read-only sanity-check helper.
"""
from __future__ import annotations

import base64
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime cycle with the vast package
    from bbvastl80.spec import RunSpec

REMOTE_DIR = "/tmp/bbvastl80"
BOOTSTRAP_PATH = f"{REMOTE_DIR}/bootstrap.sh"
BOOTSTRAP_LOG = f"{REMOTE_DIR}/bootstrap.log"
SERVER_LOG = f"{REMOTE_DIR}/server.log"
SERVER_PID = f"{REMOTE_DIR}/server.pid"
STATUS_FILE = f"{REMOTE_DIR}/status"


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
    transport: ExecTransport
    internal_port: int = 8000
    host_ip: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_user: str = "root"
    ports: dict = field(default_factory=dict)  # str(internal) -> external

    def run(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        return self.transport.run(self, cmd, timeout)

    def run_detached(self, cmd: str, logfile: str = SERVER_LOG) -> None:
        self.transport.run_detached(self, cmd, logfile)

    def put_file(self, content: str, remote_path: str, mode: str = "755") -> ExecResult:
        """Write ``content`` to ``remote_path`` on the box (via base64, so any
        bytes survive the shell). Uses the plain ``run`` path, so it works over
        any transport that can run ``sh``."""
        b64 = base64.b64encode(content.encode()).decode()
        return self.run(
            f"mkdir -p $(dirname {remote_path}) && echo {b64} | base64 -d > {remote_path} "
            f"&& chmod {mode} {remote_path}",
            timeout=60,
        )

    def external_port(self, internal: int):
        v = self.ports.get(str(internal))
        return int(v) if v else None

    def url(self) -> str | None:
        """The externally reachable base URL, if the port mapping is known."""
        ep = self.external_port(self.internal_port)
        if self.host_ip and ep:
            return f"http://{self.host_ip}:{ep}"
        return None

    def ssh_command(self, identity: str | None = None) -> list[str] | None:
        """The ``ssh`` argv a human would type to get a shell on the box."""
        t = self.transport
        if isinstance(t, SSHExecTransport):
            return t._cmd(self)
        if self.ssh_host and self.ssh_port:
            return SSHExecTransport(identity)._cmd(self)
        return None


# --- transports ----------------------------------------------------------------
class ExecTransport(ABC):
    """How a command reaches the box. ``run`` is synchronous; ``run_detached``
    starts a long-running process and returns promptly."""

    name = "base"

    @abstractmethod
    def run(self, box: Box, cmd: str, timeout: float = 120.0) -> ExecResult:
        ...

    @abstractmethod
    def run_detached(self, box: Box, cmd: str, logfile: str) -> None:
        ...


class SSHExecTransport(ExecTransport):
    """The primary transport: plain ``ssh`` to the box (the same session M4's
    port-forward tunnel rides on). Needs the account/instance to carry our
    public key — see :mod:`bbvastl80.vast.sshkeys`."""

    name = "ssh"

    def __init__(self, identity: str | None = None):
        self.identity = identity

    def _cmd(self, box: Box, *extra: str) -> list[str]:
        base = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",  # every rental is a new host
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=15",
        ]
        if self.identity:
            base += ["-i", self.identity]
        host = box.ssh_host or box.host_ip
        if box.ssh_port:
            base += ["-p", str(box.ssh_port)]
        return base + [f"{box.ssh_user}@{host}", *extra]

    def run(self, box: Box, cmd: str, timeout: float = 120.0) -> ExecResult:
        if not (box.ssh_host or box.host_ip):
            return ExecResult(255, "", "no SSH host known for this instance yet")
        try:
            p = subprocess.run(self._cmd(box, cmd), capture_output=True, text=True,
                               timeout=timeout + 15)
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", f"ssh command timed out after {timeout:.0f}s")
        return ExecResult(p.returncode, p.stdout, p.stderr)

    def run_detached(self, box: Box, cmd: str, logfile: str) -> None:
        # setsid + nohup so the process survives the SSH session ending.
        self.run(box, f"mkdir -p $(dirname {logfile}) && "
                      f"setsid nohup bash -c {shlex.quote(cmd)} >{logfile} 2>&1 < /dev/null & "
                      f"echo started", timeout=60)


class VastExecTransport(ExecTransport):
    """The Vast execute endpoint — **constrained to ``ls``, ``rm`` and ``du``**.

    Kept only for read-only sanity checks (does ``/workspace`` exist? how big is
    the model dir?). Anything else is refused *locally* with a clear message
    rather than letting the API reject it (which surfaces as a 403 from the S3
    result URL and looks like an auth problem).
    """

    name = "vast-exec"
    ALLOWED = ("ls", "rm", "du")

    def __init__(self, client):
        self.client = client

    def run(self, box: Box, cmd: str, timeout: float = 120.0) -> ExecResult:
        head = cmd.strip().split(None, 1)[0] if cmd.strip() else ""
        if head not in self.ALLOWED or any(ch in cmd for ch in ";|&\n`$"):
            return ExecResult(
                126, "",
                f"the Vast execute endpoint only runs {'/'.join(self.ALLOWED)} "
                f"(refused: {cmd[:60]!r}); use the ssh transport",
            )
        out = self.client.execute(box.instance_id, cmd, timeout=timeout) or ""
        return ExecResult(0, out)

    def run_detached(self, box: Box, cmd: str, logfile: str) -> None:
        raise RuntimeError("the Vast execute endpoint cannot start processes; "
                           "use the ssh transport (or the onstart bootstrap)")


def make_transport(spec: RunSpec, client=None) -> ExecTransport:
    """SSH, always. ``spec.ssh.identity`` (or the auto-detected default key)
    is the private key; the ``client`` argument is accepted for call-site
    compatibility but unused."""
    from bbvastl80.vast.sshkeys import resolve_identity

    identity = resolve_identity(spec.ssh.identity, required=False)
    return SSHExecTransport(str(identity) if identity else None)


# --- the provider contract ----------------------------------------------------
class Provider(ABC):
    """A recipe for serving one model from a box.

    Subclasses describe *what* to run (``prepare_cmds`` + ``launch_cmd``); the
    base class turns that into one idempotent bootstrap script (``script``) and
    also offers the step-wise ``prepare``/``launch`` for ad-hoc use.
    """

    name = "base"

    # --- what to run ------------------------------------------------------
    @abstractmethod
    def prepare_cmds(self, spec: RunSpec) -> list[str]:
        """Shell commands that make the box ready (install / stage weights).
        Each must be idempotent; a non-zero exit aborts the bootstrap."""

    @abstractmethod
    def launch_cmd(self, spec: RunSpec) -> str:
        """The foreground command that serves ``/v1`` on ``spec.serving.port``."""

    def health_cmd(self, spec: RunSpec) -> str:
        return f"curl -sf localhost:{spec.serving.port}/v1/models"

    # --- readiness --------------------------------------------------------
    def is_ready(self, box: Box, spec: RunSpec) -> bool:
        """True once the ``/v1`` endpoint answers *inside* the box."""
        return box.run(self.health_cmd(spec), timeout=15).ok

    # --- step-wise driving (ad hoc / tests) --------------------------------
    def prepare(self, box: Box, spec: RunSpec) -> None:
        from bbvastl80.vast.instance import ProvisionError

        for cmd in self.prepare_cmds(spec):
            r = box.run(cmd, timeout=3600)
            if not r.ok:
                raise ProvisionError(f"'{self.name}' prepare step failed (rc={r.exit_code}):\n"
                                     f"  $ {cmd}\n{r.text()}")

    def launch(self, box: Box, spec: RunSpec) -> None:
        box.run_detached(self.launch_cmd(spec))

    # --- the bootstrap script ----------------------------------------------
    def script(self, spec: RunSpec, extra_onstart: str | None = None) -> str:
        """One bash script that takes a fresh box to a serving ``/v1``.

        Idempotent by construction: it exits early if ``/v1`` already answers
        or a launch is already in flight, so it is safe as the instance's
        ``onstart`` (which Vast runs on *every* boot) *and* as the payload
        ``provision`` re-runs over SSH on a resume.
        """
        lines = [
            "#!/bin/bash",
            f"# bbvastl80 bootstrap — provider '{self.name}', model '{spec.model.id}'",
            "set -u",
            f"D={REMOTE_DIR}; mkdir -p $D",
            f"exec >>{BOOTSTRAP_LOG} 2>&1",
            'echo "== bbvastl80 bootstrap $(date -u +%FT%TZ)"',
            f"if {self.health_cmd(spec)} >/dev/null 2>&1; then "
            f"echo serving >{STATUS_FILE}; echo already serving; exit 0; fi",
            f"if [ -f {SERVER_PID} ] && kill -0 $(cat {SERVER_PID}) 2>/dev/null; then "
            "echo launch already in flight; exit 0; fi",
            f"echo preparing >{STATUS_FILE}",
        ]
        if extra_onstart:
            lines += ["# --- user onstart ---", extra_onstart.rstrip("\n")]
        for cmd in self.prepare_cmds(spec):
            lines.append(f"echo + {shlex.quote(cmd)}")
            lines.append(f"{cmd} || {{ echo 'prepare failed: {_sq(cmd)}' >{STATUS_FILE}; exit 1; }}")
        launch = self.launch_cmd(spec)
        lines += [
            "echo launching >" + STATUS_FILE,
            f"echo + {shlex.quote(launch)}",
            f"setsid nohup bash -c {shlex.quote(launch)} >{SERVER_LOG} 2>&1 < /dev/null &",
            f"echo $! >{SERVER_PID}",
            f"echo launched >{STATUS_FILE}",
            'echo "launched pid $(cat ' + SERVER_PID + ')"',
        ]
        return "\n".join(lines) + "\n"


def _sq(s: str) -> str:
    """Make ``s`` safe inside a single-quoted bash string."""
    return s.replace("'", "'\\''")
