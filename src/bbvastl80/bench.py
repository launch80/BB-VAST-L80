"""M5a: run the *pinned* BetterBench against a verified endpoint and refuse to
hand on a results file where nothing succeeded.

We shell out to ``$L80_BETTERBENCH run`` exactly as a human would, so the
harness version stays the one pinned on this machine (comparability). The
flags mirror ``betterbench run --help`` (0.4.0): ``--endpoint``, ``--model``,
``--out``, ``--quick`` *or* ``--passes N`` / ``--warmup N``, phase flags, and
repeatable ``--note k=v``.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from bbvastl80 import log

ALL_PHASES = ("decode", "prefill", "concurrency")


class BenchError(Exception):
    """BetterBench failed, or produced a results file we must not publish."""


def bench_cmd(betterbench_bin: str, spec, endpoint: str, out: Path) -> list[str]:
    b = spec.benchmark
    cmd = [betterbench_bin, "run", "--endpoint", endpoint,
           "--model", spec.serving.resolved_model_id(spec.model), "--out", str(out)]
    if b.quick:
        cmd.append("--quick")
    else:
        cmd += ["--passes", str(b.passes), "--warmup", str(b.warmup)]
    phases = tuple(b.phases)
    if set(phases) != set(ALL_PHASES):
        for p in phases:
            cmd.append(f"--{p}")
    eng = spec.serving.engine
    if eng and eng.max_model_len:
        cmd += ["--max-model-len", str(eng.max_model_len)]
    for k, v in (b.notes or {}).items():
        cmd += ["--note", f"{k}={v}"]
    return cmd


def run_bench(betterbench_bin: str, spec, endpoint: str, out: Path, *,
              timeout_s: float = 3 * 3600) -> Path:
    """Run BetterBench; stream its output; validate; return the results path."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = bench_cmd(betterbench_bin, spec, endpoint, out)
    log.info("$ " + " ".join(cmd))
    try:
        p = subprocess.run(cmd, timeout=timeout_s)
    except FileNotFoundError:
        raise BenchError(f"betterbench not found at {betterbench_bin!r} "
                         "(set L80_BETTERBENCH)") from None
    except subprocess.TimeoutExpired:
        raise BenchError(f"betterbench exceeded {timeout_s / 60:.0f} min") from None
    if p.returncode != 0:
        raise BenchError(f"betterbench exited {p.returncode}")
    if not out.is_file():
        raise BenchError(f"betterbench exited 0 but wrote no {out}")
    validate_results(out)
    return out


def validate_results(path: Path) -> dict:
    """The never-publish-zeros guard.

    A wrong endpoint yields a full-size results.json where every record is
    ``ok: false``. Require at least one successful record in each phase that
    ran; return a small summary for the log.
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise BenchError(f"could not read results {path}: {e}") from e
    summary: dict = {}
    ran = 0
    if "single_stream" in data:
        ran += 1
        recs = [r for cat in (data["single_stream"] or {}).values() for r in (cat or [])]
        ok = [r for r in recs if r.get("ok")]
        summary["single_stream"] = f"{len(ok)}/{len(recs)} ok"
        if not ok:
            raise BenchError(f"every single-stream request failed ({len(recs)} records) — "
                             "wrong endpoint / model id? Not publishing.")
    if "concurrency" in data:
        ran += 1
        levels = data["concurrency"] or []
        ok = sum(int(lv.get("ok") or 0) for lv in levels)
        summary["concurrency"] = f"{ok} ok across {len(levels)} levels"
        if levels and not ok:
            raise BenchError("every concurrency request failed — not publishing.")
    if "prefill" in data:
        ran += 1
        pts = data["prefill"] or []
        summary["prefill"] = f"{len(pts)} depths"
    if not ran:
        raise BenchError(f"{path} has no benchmark phases (keys: {sorted(data)})")
    env = data.get("env") or {}
    summary["model"] = env.get("model")
    summary["betterbench"] = data.get("betterbench_version")
    return summary
