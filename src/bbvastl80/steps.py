"""Spec v2: run the spec's ``steps`` in order against the box's endpoint.

Each step runs on this machine (through the tunnel) and writes into its own
folder under ``runs/<run-id>/``, so its outputs are local the moment it ends and
a later failure, the watchdog or a cancel cannot lose them. A failed step is
recorded and the next step still runs: the box is already paid for.

Every step change prints one machine-readable line for L80's console::

    bbvastl80:step {"index": 1, "type": "qualitybench", "state": "running", ...}

``state`` is ``running``, ``done``, ``failed`` or ``skipped``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bbvastl80 import log
from bbvastl80.bench import BenchError, run_bench, validate_results
from bbvastl80.publish import PublishError, parse_url, run_publish
from bbvastl80.spec import BetterBenchStep, QualityBenchStep, RunSpec
from bbvastl80.state import RunState, StateStore, StepState

# Stop starting QualityBench work this long before the run's hard deadline, so
# the results file is written (with the cut tasks listed as partial) before
# the watchdog destroys the box.
DEADLINE_MARGIN_S = 15 * 60


class StepError(Exception):
    """A step failed; the run goes on with the next one."""


def emit(st: StepState) -> None:
    d = {"index": st.index, "type": st.type, "state": st.status, "label": st.label}
    for k in ("results", "report", "url", "error"):
        v = getattr(st, k)
        if v:
            d[k] = v
    print("bbvastl80:step " + json.dumps(d), file=sys.stderr, flush=True)


def step_label(i: int, step) -> str:
    name = getattr(step, "label", None) or step.type
    return f"{i:02d}-{name}"


def run_deadline() -> float | None:
    """The run's hard stop (unix seconds), from L80's --max-hours watchdog."""
    v = os.environ.get("BBVASTL80_DEADLINE")
    try:
        return float(v) if v else None
    except ValueError:
        return None


def run_steps(cfg, spec: RunSpec, endpoint: str, store: StateStore, state: RunState, *,
              publish: bool, publish_dry_run: bool | None) -> list[StepState]:
    """Run every step; return their states. Never raises for a step's failure."""
    run_dir = Path(cfg.runs_dir) / state.run_id
    have = {s.index: s for s in state.steps}
    out: list[StepState] = []
    for i, step in enumerate(spec.steps or [], 1):
        label = step_label(i, step)
        st = have.get(i) or StepState(index=i, type=step.type, label=label)
        if st.status == "done":  # resumed run: this step finished earlier
            log.info(f"step {i} ({step.type}) finished earlier: {st.results}")
            emit(st)
            out.append(st)
            continue
        st.status, st.error, st.started_at = "running", None, time.time()
        _save(store, state, st)
        emit(st)
        d = run_dir / label
        d.mkdir(parents=True, exist_ok=True)
        try:
            if isinstance(step, BetterBenchStep):
                _betterbench(cfg, spec, step, endpoint, d, st, publish=publish and step.publish,
                             dry_run=publish_dry_run)
            elif isinstance(step, QualityBenchStep):
                _qualitybench(cfg, spec, step, endpoint, d, st, publish=publish and step.publish,
                              dry_run=publish_dry_run, gpu=state.offer_label)
            st.status = "done"
        except (StepError, BenchError, PublishError) as e:
            st.status, st.error = "failed", str(e)[:800]
            log.error(f"step {i} ({step.type}) failed: {e}")
        st.ended_at = time.time()
        _save(store, state, st)
        emit(st)
        out.append(st)
    return out


def _save(store: StateStore, state: RunState, st: StepState) -> None:
    state.steps = [s for s in state.steps if s.index != st.index] + [st]
    state.steps.sort(key=lambda s: s.index)
    store.save(state)


# --- betterbench ---------------------------------------------------------------
class _BenchView:
    """bench_cmd/publish_cmd read ``spec.benchmark``; give them this step's settings."""

    def __init__(self, spec: RunSpec, step: BetterBenchStep):
        self._spec, self.benchmark = spec, step

    def __getattr__(self, k):
        return getattr(self._spec, k)


def _betterbench(cfg, spec, step, endpoint, d: Path, st: StepState, *, publish, dry_run) -> None:
    out = d / "results.json"
    view = _BenchView(spec, step)
    run_bench(cfg.betterbench_bin, view, endpoint, out)
    summary = validate_results(out)
    st.results = str(out)
    log.success(f"betterbench: {out} · {summary}")
    if publish:
        url = run_publish(cfg.l80_bin, view, out, dry_run=dry_run)
        if url:
            st.url = url
            log.success(f"published: {url}")


# --- qualitybench ----------------------------------------------------------------
def qualitybench_cmd(l80_bin: str, spec: RunSpec, step: QualityBenchStep, endpoint: str,
                     out: Path, *, deadline: float | None = None, gpu: str | None = None) -> list[str]:
    cmd = [l80_bin, "qualitybench", "--endpoint", endpoint,
           "--model", spec.serving.resolved_model_id(spec.model), "--out", str(out), "--no-publish"]
    if step.quick:
        cmd.append("--quick")
    if step.tasks:
        cmd += ["--tasks", ",".join(step.tasks)]
    if step.thinking:
        cmd += ["--thinking", step.thinking]
    if step.reference:
        cmd += ["--reference", str(Path(step.reference).expanduser())]
    if step.thresholds:
        cmd += ["--thresholds", str(Path(step.thresholds).expanduser())]
    dl = _deadline(step.deadline, deadline)
    if dl:
        cmd += ["--deadline", dl]
    tok = step.tokenizer or spec.model.hf_repo
    if tok:
        cmd += ["--tokenizer", tok]
    notes = {"gpu": gpu, "engine": spec.serving.engine.engine if spec.serving.engine else None}
    notes = {k: v for k, v in notes.items() if v}
    notes.update(step.notes or {})
    for k, v in notes.items():
        cmd += ["--note", f"{k}={v}"]
    return cmd


def _deadline(step_deadline: str | None, hard: float | None) -> str | None:
    """The earlier of the step's own budget and the run's hard stop (less a margin)."""
    times = []
    if step_deadline:
        secs = _duration_s(step_deadline)
        if secs is None:
            return step_deadline  # an RFC 3339 time: the tool parses it
        times.append(time.time() + secs)
    if hard:
        times.append(hard - DEADLINE_MARGIN_S)
    if not times:
        return None
    t = max(min(times), time.time() + 60)
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _duration_s(v: str) -> float | None:
    units = {"h": 3600, "m": 60, "s": 1}
    total, num = 0.0, ""
    for ch in v.strip():
        if ch.isdigit() or ch == ".":
            num += ch
        elif ch in units and num:
            total += float(num) * units[ch]
            num = ""
        else:
            return None
    return total if not num and total > 0 else None


def _qualitybench(cfg, spec, step, endpoint, d: Path, st: StepState, *, publish, dry_run,
                  gpu: str | None = None) -> None:
    out = d / "results.json"
    cmd = qualitybench_cmd(cfg.l80_bin, spec, step, endpoint, out, deadline=run_deadline(), gpu=gpu)
    log.info("$ " + " ".join(cmd))
    try:
        # Inherit stdio: L80 streams the tool's qb:progress lines through.
        p = subprocess.run(cmd)
    except FileNotFoundError:
        raise StepError(f"l80 not found at {cfg.l80_bin!r} (set L80_BIN)") from None
    if not out.is_file():
        raise StepError(f"L80 qualitybench exited {p.returncode} and wrote no {out}")
    st.results = str(out)
    rep = out.with_suffix(".html")
    if rep.is_file():
        st.report = str(rep)
    try:
        doc = json.loads(out.read_text())
    except ValueError as e:
        raise StepError(f"{out} is not JSON: {e}") from None
    partial = doc.get("partials") or []
    if p.returncode != 0:
        # The tool writes its results even when a task fails; keep them, say so.
        st.error = f"L80 qualitybench exited {p.returncode}; partial tasks: {', '.join(map(str, partial)) or 'none listed'}"
        log.warn(st.error)
    if publish:
        st.url = _publish_quality(cfg.l80_bin, out, dry_run=dry_run)
    if p.returncode != 0:
        raise StepError(st.error)


def _publish_quality(l80_bin: str, results: Path, *, dry_run: bool | None, timeout_s: float = 300) -> str | None:
    cmd = [l80_bin, "qualitybench", "--results", str(results), "--json"]
    if dry_run:
        cmd.append("--dry-run")
    log.info(("validating with L80 (dry run): " if dry_run else "publishing with L80 (public): ") + " ".join(cmd))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        raise PublishError(f"l80 not found at {l80_bin!r} (set L80_BIN)") from None
    except subprocess.TimeoutExpired:
        raise PublishError(f"l80 exceeded {timeout_s:.0f}s") from None
    text = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise PublishError(f"l80 exited {p.returncode}:\n{text.strip()[:800]}")
    if dry_run:
        log.success("L80 dry run OK")
        return None
    url = parse_url(text)
    if not url:
        raise PublishError(f"l80 succeeded but printed no URL:\n{text.strip()[:800]}")
    log.success(f"published: {url}")
    return url
