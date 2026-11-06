"""Spec v2: ordered steps (betterbench, qualitybench) on one rental."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pytest
from test_run_pipeline import env, fake_bin, spec  # noqa: F401  (fixture)

from bbvastl80 import run as rp
from bbvastl80 import steps as sp
from bbvastl80.config import Config
from bbvastl80.providers.engines import VLLMProvider
from bbvastl80.spec import BetterBenchStep, QualityBenchStep


def test_no_steps_means_one_betterbench_step_from_benchmark():
    s = spec()
    assert len(s.steps) == 1 and isinstance(s.steps[0], BetterBenchStep) and s.steps[0].quick


def test_steps_parse_in_order_and_reject_unknown_tasks():
    s = spec(steps=[{"type": "qualitybench", "quick": True, "tasks": ["gsm8k"]}, {"type": "betterbench"}])
    assert [st.type for st in s.steps] == ["qualitybench", "betterbench"]
    with pytest.raises(Exception, match="unknown qualitybench task"):
        spec(steps=[{"type": "qualitybench", "tasks": ["nope"]}])
    with pytest.raises(Exception, match="steps is empty"):
        spec(steps=[])


def test_vllm_gets_max_logprobs_and_parsers_only_when_needed():
    s = spec(steps=[{"type": "qualitybench"}],
             serving={"engine": {"engine": "vllm", "tool_call_parser": "hermes", "reasoning_parser": "qwen3"}})
    cmd = VLLMProvider().launch_cmd(s)
    assert "--max-logprobs 64" in cmd and "--tool-call-parser hermes" in cmd
    assert "--enable-auto-tool-choice" in cmd and "--reasoning-parser qwen3" in cmd
    s2 = spec(steps=[{"type": "qualitybench", "tasks": ["gsm8k"]}])
    assert "--max-logprobs" not in VLLMProvider().launch_cmd(s2)
    s3 = spec(steps=[{"type": "qualitybench"}],
              serving={"engine": {"engine": "vllm", "extra_args": ["--max-logprobs=20"]}})
    assert "--max-logprobs 64" not in VLLMProvider().launch_cmd(s3)


def test_qualitybench_cmd():
    s = spec()
    st = QualityBenchStep(type="qualitybench", quick=True, tasks=["gsm8k", "bfcl"], thinking="off",
                          reference="~/ref.json", notes={"precision": "INT4"})
    cmd = sp.qualitybench_cmd("L80", s, st, "http://h:1/v1", Path("/o/r.json"), gpu="2x H200 NVL")
    j = " ".join(cmd)
    assert j.startswith("L80 qualitybench --endpoint http://h:1/v1 --model org/m --out /o/r.json --no-publish")
    assert "--quick --tasks gsm8k,bfcl --thinking off" in j and "--tokenizer org/m" in j
    assert str(Path("~/ref.json").expanduser()) in j
    assert "--note gpu=2x H200 NVL" in j and "--note precision=INT4" in j


def test_deadline_is_the_earlier_of_step_and_run():
    hard = time.time() + 2 * 3600
    d = sp._deadline("6h", hard)
    t = datetime.strptime(d, "%Y-%m-%dT%H:%M:%SZ").timestamp() - time.timezone
    assert abs(t - (hard - sp.DEADLINE_MARGIN_S)) < 5
    d2 = sp._deadline("30m", hard)
    t2 = datetime.strptime(d2, "%Y-%m-%dT%H:%M:%SZ").timestamp() - time.timezone
    assert abs(t2 - (time.time() + 1800)) < 5
    assert sp._deadline(None, None) is None
    assert sp._deadline("2026-12-01T00:00:00Z", None) == "2026-12-01T00:00:00Z"


def _l80_quality(tmp: Path, code: int = 0) -> str:
    # A fake L80: `qualitybench --out X` writes X; `qualitybench --results` "publishes".
    body = (f'echo "$@" >> {tmp}/l80q.args\n'
            'for a in "$@"; do [ "$prev" = "--out" ] && OUT="$a"; prev="$a"; done\n'
            'case "$*" in *"qualitybench --results"*) echo \'{"url":"https://launch80.com/a/q1"}\'; exit 0;; esac\n'
            'case "$*" in *"qualitybench --endpoint"*) '
            'printf \'{"partials":[]}\' > "$OUT"; echo "<html>" > "${OUT%.json}.html"; exit CODE;; '
            '*) echo \'{"url":"https://launch80.com/a/bb1"}\';; esac\n')
    return fake_bin(tmp, "l80q", body.replace("CODE", str(code)))


def test_pipeline_runs_each_step_and_publishes_each(env, client):  # noqa: F811
    cfg = Config(api_key="k", api_base=env["cfg"].api_base, l80_bin=_l80_quality(env["tmp"]),
                 betterbench_bin=env["cfg"].betterbench_bin, runs_dir=env["tmp"] / "runs")
    s = spec(steps=[{"type": "betterbench", "quick": True},
                    {"type": "qualitybench", "quick": True, "label": "base"}])
    st = rp.run_pipeline(cfg, s, client)
    assert st.status == "torn_down" and env["destroy"].called
    assert [x.status for x in st.steps] == ["done", "done"]
    assert st.steps[1].label == "02-base" and st.steps[1].url == "https://launch80.com/a/q1"
    assert Path(st.steps[1].results).name == "results.json" and st.steps[1].report.endswith(".html")
    assert st.publish_url == "https://launch80.com/a/bb1"  # v1 field from the betterbench step
    args = (env["tmp"] / "l80q.args").read_text()
    assert "--no-publish" in args and "--results" in args


def test_a_failed_step_does_not_stop_the_next(env, client):  # noqa: F811
    bad = fake_bin(env["tmp"], "bb_bad", "exit 5\n")
    cfg = Config(api_key="k", api_base=env["cfg"].api_base, l80_bin=_l80_quality(env["tmp"]),
                 betterbench_bin=bad, runs_dir=env["tmp"] / "runs")
    s = spec(steps=[{"type": "betterbench"}, {"type": "qualitybench", "publish": False}])
    st = rp.run_pipeline(cfg, s, client)
    assert [x.status for x in st.steps] == ["failed", "done"] and env["destroy"].called
    assert st.steps[1].url is None


def test_qualitybench_exit_1_keeps_results_but_fails_the_step(env, client):  # noqa: F811
    cfg = Config(api_key="k", api_base=env["cfg"].api_base, l80_bin=_l80_quality(env["tmp"], code=1),
                 betterbench_bin=env["cfg"].betterbench_bin, runs_dir=env["tmp"] / "runs")
    s = spec(steps=[{"type": "betterbench"}, {"type": "qualitybench", "publish": False}])
    st = rp.run_pipeline(cfg, s, client)
    assert st.steps[1].status == "failed" and Path(st.steps[1].results).is_file()
