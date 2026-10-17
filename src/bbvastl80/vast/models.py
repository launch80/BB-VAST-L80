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
    def host_ip(self) -> str | None:
        return self.public_ip or self.ip

    def external_port(self, internal: int) -> int | None:
        """Map an internal container port to its (random) external port.

        Primary source is the ``env`` the API populates (``VAST_TCP_PORT_<n>``);
        we fall back to a ``ports`` list/dict of {internal, external} pairs.
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
                if str(k) == str(internal):
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        return None
        return None
