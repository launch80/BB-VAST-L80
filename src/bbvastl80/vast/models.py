"""Typed models for the subset of the Vast.ai API we use.

All models allow extra fields (``extra="allow"``) so an upstream schema change
adds a field instead of crashing a parse. Units that trip people up are called
out: ``gpu_total_ram``/``gpu_ram``/``cpu_ram`` are **mebibytes**, ``dph_*`` are
**dollars per hour**, ``disk_space`` is **GB**.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class User(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int | None = None
    email: str | None = None
    credit: float | None = None
    ssh_key: str | None = None


class Offer(BaseModel):
    """One rentable machine offer, as returned by ``POST /bundles``."""

    model_config = ConfigDict(extra="allow")
    id: int
    gpu_name: str | None = None
    num_gpus: int | None = None
    gpu_ram: int | None = None  # MB per GPU
    gpu_total_ram: int | None = None  # MB across all GPUs
    dph_total: float | None = None  # $/hr
    dph_base: float | None = None
    datacenter: str | None = None
    geolocation: str | None = None
    reliability: float | None = None
    verified: bool | None = None
    disk_space: float | None = None  # GB
    cpu_cores: int | None = None
    cpu_ram: int | None = None  # MB
    static_ip: bool | None = None
    direct_port_count: int | None = None

    @property
    def total_ram_gb(self) -> float | None:
        return round(self.gpu_total_ram / 1024.0, 1) if self.gpu_total_ram else None

    @property
    def label(self) -> str:
        gpu = self.gpu_name or "?"
        n = f"x{self.num_gpus}" if self.num_gpus and self.num_gpus > 1 else ""
        return f"{gpu}{n}"


# --- status vocabulary -------------------------------------------------------
# ``actual_status`` is the lifecycle field the live API returns. "loading" is the
# provisioning state; it flips to "running" once the box is up and reachable.
RUNNING = "running"
WAITING = {"loading", "initializing", "pending", "deploying", "starting", "in-progress", "waiting"}
TERMINAL_FAIL = {
    "exited",
    "stopped",
    "destroyed",
    "offline",
    "failed",
    "error",
    "unknown",
    "terminated",
}


class Instance(BaseModel):
    """A rented instance (contract). Tolerant of schema drift.

    The live v0 record names its fields ``actual_status`` / ``public_ipaddr``;
    we alias them to the friendlier ``status`` / ``public_ip`` attributes (and
    ``populate_by_name`` lets tests keep passing the short names).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: int | None = None
    status: str | None = Field(default=None, validation_alias="actual_status")
    intended_status: str | None = None   # what Vast is steering toward (running/stopped)
    cur_state: str | None = None
    status_msg: str | None = None        # host/daemon message (errors land here)
    type: int | str | None = None  # instance-type code (int) or label
    public_ip: str | None = Field(default=None, validation_alias="public_ipaddr")
    ip: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    ports: object = None  # list of {internal, external} or a dict — parsed lazily
    offer: object = None
    disk: float | None = None
    label: str | None = None

    @property
    def is_running(self) -> bool:
        return (self.status or "").lower() == RUNNING

    @property
    def is_waiting(self) -> bool:
        return (self.status or "").lower() in WAITING

    @property
    def is_terminal_failed(self) -> bool:
        return (self.status or "").lower() in TERMINAL_FAIL

    @property
    def host_error(self) -> str | None:
        """The host's failure message when it gave up starting the container.

        Seen live: ``actual_status: created`` forever, ``intended_status:
        stopped`` and a ``status_msg`` like "Error response from daemon: …
        nvidia-container-cli: device error". Polling can't fix that; the
        caller should destroy the instance and rent the next offer.
        """
        if self.is_running:
            return None
        msg = (self.status_msg or "").strip()
        stopped = (self.intended_status or "").lower() == "stopped" or \
            (self.cur_state or "").lower() == "stopped"
        looks_bad = any(t in msg.lower() for t in
                        ("error", "failed", "unable to", "no such", "exit status"))
        if stopped and (looks_bad or (self.status or "").lower() == "created"):
            return msg or f"host stopped the instance while '{self.status}'"
        return None

    @property
    def host_ip(self) -> str | None:
        return self.public_ip or self.ip

    def external_port(self, internal: int) -> int | None:
        """Map an internal container port to its (random) external port.

        Primary source is the ``env`` the API populates (``VAST_TCP_PORT_<n>``);
        we fall back to ``ports``, which the live API returns in Docker's
        shape (``{"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41234"}]}``) and
        which tests/older records may give as ``{internal: external}`` or a
        list of ``{internal, external}`` pairs.
        """
        key = f"VAST_TCP_PORT_{internal}"
        if self.env and self.env.get(key):
            try:
                return int(self.env[key])
            except (TypeError, ValueError):
                pass
        ports = self.ports
        if isinstance(ports, list):
            for p in ports:
                try:
                    if int(p.get("internal")) == internal:
                        return int(p.get("external"))
                except (AttributeError, TypeError, ValueError):
                    continue
        elif isinstance(ports, dict):
            for k, v in ports.items():
                if str(k).split("/")[0] != str(internal):
                    continue
                try:
                    if isinstance(v, list):  # Docker shape
                        return int(v[0]["HostPort"]) if v else None
                    return int(v)
                except (TypeError, ValueError, KeyError, IndexError):
                    return None
        return None

    def ssh_target(self) -> tuple[str, int] | None:
        """``(host, port)`` to SSH into, resolved the way the official CLI does:
        a direct mapping of container port 22 on the public IP wins; otherwise
        Vast's SSH proxy (``ssh_host``/``ssh_port``)."""
        direct = self.external_port(22)
        if direct and self.host_ip:
            return self.host_ip, direct
        if self.ssh_host and self.ssh_port:
            return self.ssh_host, int(self.ssh_port)
        return None
