"""The run spec — the single flexible knob that drives a whole run.

Everything is optional-with-defaults so a 3-line spec works, and a long one
does too. The :class:`ServingSpec` is the pluggable-provider contract (§2.1 of
PLAN.md): ``mode: engine`` (a preset) or ``mode: custom`` (your own stack).
Both reduce to the same goal — *the box serves ``model_id`` at an
OpenAI-compatible ``/v1`` on ``port``* — so the downstream phases
(connect / bench / publish) are identical for either.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


# --- GPU selection -----------------------------------------------------------
class GPUFilter(BaseModel):
    """What GPU to rent. Omit everything to get 'cheapest that exists'."""

    names: list[str] | None = Field(
        None, description="gpu_name values, e.g. ['H100_SXM5', 'A100_SXM4_80GB']"
    )
    count: int = Field(1, description="number of GPUs (num_gpus)")
    min_ram_gb: float | None = Field(None, description="min total GPU RAM in GB")
    max_dph: float | None = Field(None, description="max price, dollars per hour")
    type: Literal["ondemand", "bid", "reserved"] | None = None
    verified: bool | None = None
    min_reliability: float | None = Field(None, ge=0.0, le=1.0)
    min_inet_down: float | None = Field(
        None, ge=0, description="min download speed, Mbps: the engine image and weights come down at "
                                "this rate, and a slow host can miss max_provision_min")
    datacenter: str | None = None
    geolocation: str | None = None
    pick: str = Field("cheapest", description="cheapest | first | <1-based index>")
    limit: int = Field(40, ge=1, le=200)


# --- Model -------------------------------------------------------------------
class ModelSpec(BaseModel):
    id: str = Field(..., description="the model id requested at /v1 (for BetterBench)")
    source: Literal["hf", "url", "local-path"] = "hf"
    hf_repo: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _source_has_target(self) -> "ModelSpec":
        if self.source == "hf" and not self.hf_repo:
            raise ValueError("model.source=hf requires model.hf_repo")
        if self.source == "url" and not self.url:
            raise ValueError("model.source=url requires model.url")
        return self


# --- Serving (the pluggable provider) ---------------------------------------
# Engines with a working preset in providers/engines.py.
ENGINE_PRESETS = ("vllm", "llamacpp")


class EngineServing(BaseModel):
    """mode: engine — a preset. We generate install + launch for you."""

    engine: Literal["vllm", "llamacpp", "sglang", "tgi", "ollama"]
    image: str | None = Field(None, description="override the preset's default image")
    max_model_len: int | None = None
    gpu_mem_util: float | None = Field(None, ge=0.1, le=1.0)
    tool_call_parser: str | None = Field(
        None, description="vLLM --tool-call-parser (enables auto tool choice); QualityBench's bfcl "
                          "scores the server's own parser")
    reasoning_parser: str | None = Field(
        None, description="vLLM --reasoning-parser; needed for a qualitybench step with thinking: on")
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("engine")
    @classmethod
    def _engine_has_preset(cls, v: str) -> str:
        # Reserved in the schema, but there is no preset to provision them yet.
        if v not in ENGINE_PRESETS:
            raise ValueError(
                f"engine {v!r} has no preset yet; available: {', '.join(ENGINE_PRESETS)}. "
                "Use serving.mode: custom with your own image and start command."
            )
        return v


class HealthSpec(BaseModel):
    path: str = "/v1/models"
    expects: str | None = Field(
        None, description="substring that must appear; null => matches serving.model_id"
    )


class WeightsSpec(BaseModel):
    """Lean: how weights reach the box. (hf / url / none; 'scp' is a later milestone.)"""

    source: Literal["hf", "url", "none"] = "none"
    repo: str | None = Field(None, description="HF repo id, or a URL when source=url")


class CustomServing(BaseModel):
    """mode: custom — your own stack. Only ``image`` + ``start`` are required."""

    image: str
    setup: list[str] = Field(
        default_factory=list,
        description="shell commands run in-container before start (git pull, pip, fetch weights…)",
    )
    start: str = Field(..., description="command that launches the server (runs in foreground)")
    health: HealthSpec = Field(default_factory=HealthSpec)
    weights: WeightsSpec = Field(default_factory=WeightsSpec)


class ServingSpec(BaseModel):
    mode: Literal["engine", "custom"] = "engine"
    port: int = 8000
    model_id: str | None = Field(
        None, description="id served at /v1; defaults to model.id"
    )
    engine: EngineServing | None = None
    custom: CustomServing | None = None

    @model_validator(mode="before")
    @classmethod
    def _infer_mode(cls, data):
        # Let a user omit `mode` and just give the block they mean.
        if isinstance(data, dict) and data.get("mode") is None:
            if data.get("custom") is not None:
                data["mode"] = "custom"
        return data

    @model_validator(mode="after")
    def _mode_matches_block(self) -> "ServingSpec":
        if self.mode == "engine" and self.engine is None:
            # A minimal spec (just `model:`) should work: default to the vLLM preset.
            self.engine = EngineServing(engine="vllm")
        if self.mode == "custom" and self.custom is None:
            raise ValueError("serving.mode=custom requires the serving.custom block")
        return self

    def resolved_model_id(self, model: ModelSpec) -> str:
        return self.model_id or model.id


# --- Instance ----------------------------------------------------------------
class InstanceSpec(BaseModel):
    disk_gb: int = Field(200, ge=1)
    runtype: Literal["ssh", "ssh_direct", "ssh_proxy", "jupyter", "args"] = "ssh"
    label: str = "bbvastl80"
    onstart: str | None = Field(
        None, description="extra shell run at boot, before the provider's prepare/launch"
    )
    bootstrap: Literal["onstart", "ssh"] = Field(
        "onstart",
        description="onstart => the provider's bootstrap script is the instance's onstart "
                    "(starts serving on first boot); ssh => only pushed by `provision`",
    )
    price: float | None = Field(None, description="bid price $/hr (mode: bid offers only)")


# --- Benchmark ---------------------------------------------------------------
class BenchmarkSpec(BaseModel):
    quick: bool = False
    passes: int = Field(20, ge=1)
    warmup: int = Field(2, ge=0)
    phases: list[Literal["decode", "prefill", "concurrency"]] = Field(
        default_factory=lambda: ["decode", "prefill", "concurrency"]
    )
    notes: dict[str, str] = Field(default_factory=dict)


# --- Steps (spec v2) ----------------------------------------------------------
# A run is: rent -> provision -> connect -> each step in order -> reap. Steps run
# on this machine against the box's endpoint (through the tunnel), so their
# outputs are local the moment a step ends and a later failure cannot lose them.
class BetterBenchStep(BaseModel):
    type: Literal["betterbench"] = "betterbench"
    quick: bool = False
    passes: int = Field(20, ge=1)
    warmup: int = Field(2, ge=0)
    phases: list[Literal["decode", "prefill", "concurrency"]] = Field(
        default_factory=lambda: ["decode", "prefill", "concurrency"]
    )
    notes: dict[str, str] = Field(default_factory=dict)
    publish: bool = True


QB_TASKS = ("gsm8k", "ifeval", "mmlu_pro", "humaneval", "mbpp", "bfcl", "lcb", "logprobs",
            "needle", "spec_parity")
QB_DEFAULT = tuple(t for t in QB_TASKS if t != "spec_parity")


class QualityBenchStep(BaseModel):
    """`L80 qualitybench` against the rented endpoint. A run with ``reference``
    is a compare: the output is the reference's page with this run as an arm."""

    type: Literal["qualitybench"]
    label: str | None = Field(None, description="names the output folder and the page arm")
    quick: bool = False
    tasks: list[str] = Field(default_factory=list, description="empty => the default suite")
    thinking: Literal["on", "off"] | None = None
    reference: str | None = Field(None, description="a local results file (its .qb/ folder beside it)")
    thresholds: str | None = None
    deadline: str | None = Field(None, description="a duration such as 3h: no work starts after it")
    tokenizer: str | None = Field(None, description="HF repo for logprobs/needle; null => model.hf_repo")
    notes: dict[str, str] = Field(default_factory=dict)
    publish: bool = True

    @field_validator("tasks")
    @classmethod
    def _known_tasks(cls, v: list[str]) -> list[str]:
        bad = [t for t in v if t not in QB_TASKS]
        if bad:
            raise ValueError(f"unknown qualitybench task(s) {', '.join(bad)}; known: {', '.join(QB_TASKS)}")
        return v

    def task_set(self) -> tuple[str, ...]:
        return tuple(self.tasks) or QB_DEFAULT


Step = Annotated[Union[BetterBenchStep, QualityBenchStep], Field(discriminator="type")]


# --- Publish -----------------------------------------------------------------
class PublishSpec(BaseModel):
    template: Literal["bench.betterbench.v1", "bench.report.v1"] = "bench.betterbench.v1"
    title: str | None = None
    summary: str | None = None
    sections: list[str] = Field(default_factory=list, description='"Heading=Body"')
    dry_run: bool = False


# --- Budget / connect / lifecycle -------------------------------------------
class BudgetSpec(BaseModel):
    max_total_usd: float | None = Field(None, ge=0)
    max_provision_min: float = Field(20.0, ge=1)


class ConnectSpec(BaseModel):
    mode: Literal["auto", "direct", "tunnel"] = "auto"
    local_port: int = 8000


class SSHSpec(BaseModel):
    """SSH is how every command reaches the box (the Vast execute endpoint
    only runs ls/rm/du). Defaults find ``~/.ssh/id_ed25519`` (or ecdsa/rsa)
    and register its ``.pub`` with the account before renting."""

    identity: str | None = Field(
        None, description="private key path; null => first of ~/.ssh/id_{ed25519,ecdsa,rsa}"
    )
    user: str = Field("root", description="SSH user on the box")
    public_key: str | None = Field(
        None, description="public key text or .pub path; null => <identity>.pub"
    )


class LifecycleSpec(BaseModel):
    teardown: Literal["auto", "manual"] = "auto"
    keep_running: bool = False


# --- Top-level ---------------------------------------------------------------
class RunSpec(BaseModel):
    name: str | None = None
    gpu: GPUFilter = Field(default_factory=GPUFilter)
    model: ModelSpec
    serving: ServingSpec = Field(default_factory=ServingSpec)
    instance: InstanceSpec = Field(default_factory=InstanceSpec)
    benchmark: BenchmarkSpec = Field(default_factory=BenchmarkSpec)
    steps: list[Step] | None = Field(
        None, description="ordered operations on the box; null => one betterbench step from `benchmark`")
    publish: PublishSpec = Field(default_factory=PublishSpec)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    connect: ConnectSpec = Field(default_factory=ConnectSpec)
    ssh: SSHSpec = Field(default_factory=SSHSpec)
    lifecycle: LifecycleSpec = Field(default_factory=LifecycleSpec)

    @model_validator(mode="before")
    @classmethod
    def _blank_subblocks(cls, data):
        # A YAML key present but empty (``ssh:`` with only comments) parses as
        # null; treat it as an empty block so the sub-model's defaults apply.
        subblocks = ("gpu", "model", "serving", "instance", "benchmark",
                     "publish", "budget", "connect", "ssh", "lifecycle")
        if isinstance(data, dict):
            for key in subblocks:
                if data.get(key) is None:
                    data[key] = {}
        return data

    @model_validator(mode="after")
    def _defaults(self) -> "RunSpec":
        if self.serving.model_id is None:
            self.serving.model_id = self.model.id
        if self.steps is None:
            b = self.benchmark
            self.steps = [BetterBenchStep(quick=b.quick, passes=b.passes, warmup=b.warmup,
                                          phases=list(b.phases), notes=dict(b.notes))]
        if not self.steps:
            raise ValueError("steps is empty: give at least one step, or leave it out for one betterbench step")
        return self

    def needs_logprobs(self) -> bool:
        return any(isinstance(s, QualityBenchStep) and "logprobs" in s.task_set() for s in self.steps or [])


def load_spec(path: str | Path) -> RunSpec:
    """Load + validate a YAML run spec. Raises :class:`SpecError` with details."""
    path = Path(path)
    if not path.exists():
        raise SpecError(f"spec file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise SpecError(f"could not parse {path} as YAML: {e}") from e
    try:
        return RunSpec.model_validate(raw)
    except ValidationError as e:
        raise SpecError(_format_validation_error(e)) from e


class SpecError(Exception):
    """A user-fixible problem with the run spec."""


def _format_validation_error(e: ValidationError) -> str:
    lines = ["spec is invalid:"]
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


__all__ = [
    "RunSpec",
    "SpecError",
    "load_spec",
    "GPUFilter",
    "ModelSpec",
    "ServingSpec",
    "EngineServing",
    "CustomServing",
    "HealthSpec",
    "WeightsSpec",
    "InstanceSpec",
    "BenchmarkSpec",
    "BetterBenchStep",
    "QualityBenchStep",
    "Step",
    "PublishSpec",
    "BudgetSpec",
    "ConnectSpec",
    "SSHSpec",
    "LifecycleSpec",
]
