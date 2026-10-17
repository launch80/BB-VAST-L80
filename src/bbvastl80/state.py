"""Per-run state on disk so a run is resumable and teardown works later.

Layout::

    runs/<run-id>/state.json    # RunState snapshot
    runs/<run-id>/results.json  # the durable BetterBench artifact (M5)

``run-id`` is ``<name-slug>-<timestamp>``. The same id is the key for
``destroy --run <id>``, so a box can be torn down days after the process that
rented it has exited.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

STATUS_PHASES = ["created", "searched", "rented", "provisioned", "connected",
                 "benchmarked", "published", "torn_down", "failed"]


class RunState(BaseModel):
    run_id: str
    created_at: float = Field(default_factory=time.time)
    status: str = "created"
    spec_name: str | None = None
    # Vast
    offer_id: int | None = None
    offer_label: str | None = None
    instance_id: int | None = None
    instance_status: str | None = None
    public_ip: str | None = None
    ports: dict[str, int] = Field(default_factory=dict)  # internal -> external
    # connectivity
    endpoint: str | None = None
    connect_mode: str | None = None
    # benchmark / publish
    results_path: str | None = None
    publish_url: str | None = None
    # cost
    dph: float | None = None
    started_billing_at: float | None = None


class StateStore:
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)

    def new_run_id(self, name: str | None) -> str:
        slug = (name or "run").replace("/", "-").replace(" ", "-")
        return f"{slug}-{time.time():.0f}"

    def path_for(self, run_id: str) -> Path:
        return self.runs_dir / run_id / "state.json"

    def create(self, run_id: str, spec: RunState) -> Path:
        p = self.path_for(run_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(spec.model_dump_json(indent=2))
        return p

    def load(self, run_id: str) -> RunState:
        p = self.path_for(run_id)
        if not p.exists():
            raise FileNotFoundError(f"no state for run '{run_id}' at {p}")
        return RunState.model_validate(json.loads(p.read_text()))

    def save(self, state: RunState) -> Path:
        p = self.path_for(state.run_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(state.model_dump_json(indent=2))
        return p

    def list_run_ids(self) -> list[str]:
        if not self.runs_dir.exists():
            return []
        return sorted(
            (p.parent.name for p in self.runs_dir.glob("*/state.json")),
            reverse=True,
        )
