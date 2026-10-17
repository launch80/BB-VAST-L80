"""M5: BetterBench invocation + the never-publish-zeros guard; L80 publish."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from bbvastl80 import bench as bb
from bbvastl80 import publish as pb
from bbvastl80.spec import RunSpec


def spec(**kw) -> RunSpec:
    base = {"model": {"id": "org/m", "hf_repo": "org/m"}}
    base.update(kw)
    return RunSpec(**base)


def results(ok=True, phases=("single_stream", "prefill", "concurrency")) -> dict:
    d = {"schema": 1, "betterbench_version": "0.4.0", "env": {"model": "org/m"}}
    if "single_stream" in phases:
        d["single_stream"] = {"chat": [{"ok": ok, "decode_tps": 50.0}, {"ok": False}]}
    if "prefill" in phases:
        d["prefill"] = [{"depth": 2048, "tps": 1000}]
    if "concurrency" in phases:
        d["concurrency"] = [{"level": 4, "requests": 8, "ok": 8 if ok else 0}]
    return d


def fake_bin(tmp_path: Path, name: str, body: str) -> str:
    p = tmp_path / name
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


# --- bench ---------------------------------------------------------------------
def test_bench_cmd_quick_and_notes_and_phases():
    s = spec(benchmark={"quick": True, "phases": ["decode", "prefill"],
                        "notes": {"engine": "vllm", "tp": "2"}},
             serving={"engine": {"engine": "vllm", "max_model_len": 4096}})
    cmd = bb.bench_cmd("/bb", s, "http://h/v1", Path("/o/results.json"))
    assert cmd[:3] == ["/bb", "run", "--endpoint"] and cmd[3] == "http://h/v1"
    assert cmd[cmd.index("--model") + 1] == "org/m"
    assert "--quick" in cmd and "--passes" not in cmd
    assert "--decode" in cmd and "--prefill" in cmd and "--concurrency" not in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "4096"
    assert cmd.count("--note") == 2 and "engine=vllm" in cmd and "tp=2" in cmd


def test_bench_cmd_full_run_passes_warmup_all_phases_no_flags():
    cmd = bb.bench_cmd("/bb", spec(benchmark={"passes": 7, "warmup": 1}), "http://h/v1", Path("r"))
    assert cmd[cmd.index("--passes") + 1] == "7" and cmd[cmd.index("--warmup") + 1] == "1"
    assert not any(f in cmd for f in ("--decode", "--prefill", "--concurrency", "--quick"))


def test_validate_results_guard(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(results()))
    summary = bb.validate_results(good)
    assert summary["single_stream"] == "1/2 ok" and summary["model"] == "org/m"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(results(ok=False)))
    with pytest.raises(bb.BenchError, match="every single-stream request failed"):
        bb.validate_results(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"env": {}}))
    with pytest.raises(bb.BenchError, match="no benchmark phases"):
        bb.validate_results(empty)


def test_run_bench_with_fake_betterbench(tmp_path):
    out = tmp_path / "runs" / "r1" / "results.json"
    payload = json.dumps(results()).replace("'", "")
    bin_ = fake_bin(tmp_path, "betterbench",
                    'for a in "$@"; do [ "$prev" = "--out" ] && OUT="$a"; prev="$a"; done\n'
                    f"printf '%s' '{payload}' > \"$OUT\"\n")
    assert bb.run_bench(bin_, spec(benchmark={"quick": True}), "http://h/v1", out) == out
    assert json.loads(out.read_text())["single_stream"]["chat"][0]["ok"] is True


def test_run_bench_nonzero_exit_and_missing_file(tmp_path):
    out = tmp_path / "results.json"
    bad = fake_bin(tmp_path, "bb_fail", "exit 3\n")
    with pytest.raises(bb.BenchError, match="exited 3"):
        bb.run_bench(bad, spec(), "http://h/v1", out)
    silent = fake_bin(tmp_path, "bb_silent", "exit 0\n")
    with pytest.raises(bb.BenchError, match="wrote no"):
        bb.run_bench(silent, spec(), "http://h/v1", out)


# --- publish -------------------------------------------------------------------
def test_publish_cmd_flags():
    s = spec(publish={"template": "bench.report.v1", "title": "T", "summary": "S",
                      "sections": ["A=B", "C=D"], "dry_run": True})
    cmd = pb.publish_cmd("/l80", s, Path("/r.json"))
    assert cmd[:4] == ["/l80", "betterbench", "--results", "/r.json"]
    assert cmd[cmd.index("--template") + 1] == "bench.report.v1"
    assert "--json" in cmd and "--dry-run" in cmd and cmd.count("--section") == 2
    assert "--dry-run" not in pb.publish_cmd("/l80", s, Path("/r.json"), dry_run=False)


def test_parse_url_json_and_fallback():
    assert pb.parse_url('{"url": "https://launch80.com/a/abc", "x": 1}') == "https://launch80.com/a/abc"
    assert pb.parse_url('log line\n{"artifact": {"url": "https://launch80.com/a/def"}}') == \
        "https://launch80.com/a/def"
    assert pb.parse_url("published https://launch80.com/a/ghi ok") == "https://launch80.com/a/ghi"
    assert pb.parse_url("nothing here") is None


def test_run_publish_with_fake_l80(tmp_path):
    r = tmp_path / "results.json"
    r.write_text(json.dumps(results()))
    ok = fake_bin(tmp_path, "l80", 'echo \'{"url":"https://launch80.com/a/zzz","guid":"zzz"}\'\n')
    assert pb.run_publish(ok, spec(), r) == "https://launch80.com/a/zzz"
    assert pb.run_publish(ok, spec(), r, dry_run=True) is None
    fail = fake_bin(tmp_path, "l80_fail", "echo 'error: E_RATE_LIMIT' >&2; exit 2\n")
    with pytest.raises(pb.PublishError, match="E_RATE_LIMIT"):
        pb.run_publish(fail, spec(), r)
    nourl = fake_bin(tmp_path, "l80_nourl", "echo done\n")
    with pytest.raises(pb.PublishError, match="no URL"):
        pb.run_publish(nourl, spec(), r)
    assert os.access(ok, os.X_OK)
