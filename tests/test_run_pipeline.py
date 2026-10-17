"""M6: the pipeline end-to-end against a respx Vast API, a fake SSH transport,
and fake betterbench / l80 binaries."""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from respx.models import MockResponse

from conftest import BASE, BASE_V1
from bbvastl80 import bootstrap as bs
from bbvastl80 import run as rp
from bbvastl80.config import Config
from bbvastl80.providers.base import ExecResult, ExecTransport
from bbvastl80.spec import RunSpec
from bbvastl80.state import StateStore
from bbvastl80.vast import sshkeys

PUB = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTests0000000000000000000000000 me"
INSTANCE = {"id": 777, "actual_status": "running", "public_ipaddr": "5.6.7.8",
            "ports": {"22/tcp": [{"HostPort": "41122"}], "8000/tcp": [{"HostPort": "41800"}]}}


class HappyTransport(ExecTransport):
    name = "fake-ssh"

    def __init__(self):
        self.calls = []

    def run(self, box, cmd, timeout=120.0):
        self.calls.append(cmd)
        if "curl -sf http://localhost:8000/v1/models" in cmd:
            # not serving until the bootstrap has been pushed
            pushed = any("base64 -d" in c for c in self.calls)
            return ExecResult(0 if pushed else 7, '{"data":[{"id":"org/m"}]}' if pushed else "")
        if "cat /tmp/bbvastl80/status" in cmd:
            return ExecResult(0, "launched")
        return ExecResult(0, "")

    def run_detached(self, box, cmd, logfile):
        self.calls.append("DETACHED " + cmd)


def fake_bin(d: Path, name: str, body: str) -> str:
    p = d / name
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


@pytest.fixture
def env(tmp_path, monkeypatch, router):
    # local ssh key
    monkeypatch.setattr(sshkeys.Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "id_ed25519").write_text("k")
    (tmp_path / ".ssh" / "id_ed25519.pub").write_text(PUB)
    # no sleeping, fake transport
    for mod in (rp, bs):
        monkeypatch.setattr(mod.time, "sleep", lambda s: None, raising=False)
    import bbvastl80.vast.instance as vi
    monkeypatch.setattr(vi.time, "sleep", lambda s: None)
    t = HappyTransport()
    monkeypatch.setattr(bs, "make_transport", lambda spec, client=None: t)
    # connect: direct works
    import bbvastl80.connect as cn
    monkeypatch.setattr(cn.time, "sleep", lambda s: None)
    router.get("http://5.6.7.8:41800/v1/models").mock(
        return_value=MockResponse(200, content=json.dumps({"data": [{"id": "org/m"}]})))
    # Vast API
    router.get(f"{BASE}/users/current/").mock(
        return_value=MockResponse(200, content=json.dumps({"id": 1, "credit": 50.0})))
    router.get(f"{BASE}/ssh/").mock(return_value=MockResponse(200, content="[]"))
    router.post(f"{BASE}/ssh/").mock(return_value=MockResponse(200, content='{"success":true}'))
    router.post(f"{BASE}/bundles").mock(return_value=MockResponse(200, content=json.dumps(
        {"offers": [{"id": 11, "gpu_name": "H200 NVL", "num_gpus": 2, "dph_total": 7.2},
                    {"id": 12, "gpu_name": "H200 NVL", "num_gpus": 2, "dph_total": 8.0}]})))
    asks = router.put(f"{BASE}/asks/11/").mock(
        return_value=MockResponse(200, content=json.dumps({"success": True, "new_contract": 777})))
    router.get(f"{BASE}/instances/777").mock(
        return_value=MockResponse(200, content=json.dumps({"instances": INSTANCE})))
    router.post(f"{BASE}/instances/777/ssh/").mock(return_value=MockResponse(200, content="{}"))
    destroy = router.delete(f"{BASE}/instances/777").mock(return_value=MockResponse(200, content="{}"))
    router.get(f"{BASE_V1}/instances").mock(return_value=MockResponse(200, content='{"instances":[]}'))
    # fake harness + publisher
    payload = json.dumps({"betterbench_version": "0.4.0", "env": {"model": "org/m"},
                          "single_stream": {"chat": [{"ok": True, "decode_tps": 42}]},
                          "concurrency": [{"level": 4, "requests": 4, "ok": 4}]})
    bb = fake_bin(tmp_path, "betterbench",
                  'for a in "$@"; do [ "$prev" = "--out" ] && OUT="$a"; prev="$a"; done\n'
                  f"printf '%s' '{payload}' > \"$OUT\"; echo \"$@\" > {tmp_path}/bb.args\n")
    l80 = fake_bin(tmp_path, "l80",
                   f'echo "$@" > {tmp_path}/l80.args; '
                   'case "$*" in *--dry-run*) echo \'{"ok":true}\';; '
                   '*) echo \'{"url":"https://launch80.com/a/run1"}\';; esac\n')
    cfg = Config(api_key="k", api_base=BASE, l80_bin=l80, betterbench_bin=bb,
                 runs_dir=tmp_path / "runs")
    return dict(cfg=cfg, tmp=tmp_path, transport=t, asks=asks, destroy=destroy)


def spec(**kw) -> RunSpec:
    base = {"name": "e2e", "model": {"id": "org/m", "hf_repo": "org/m"},
            "gpu": {"names": ["H200 NVL"], "count": 2}, "benchmark": {"quick": True},
            "budget": {"max_total_usd": 20}}
    base.update(kw)
    return RunSpec(**base)


def test_full_pipeline_publishes_then_destroys(env, client):
    st = rp.run_pipeline(env["cfg"], spec(), client)
    assert st.status == "torn_down" and st.publish_url == "https://launch80.com/a/run1"
    assert st.connect_mode == "direct" and st.endpoint == "http://5.6.7.8:41800/v1"
    assert Path(st.results_path).is_file()
    assert env["destroy"].called
    # rent body carried the onstart bootstrap + env port mapping
    body = json.loads(env["asks"].calls[0].request.content)
    assert body["env"] == {"-p 8000:8000": "1"} and body["onstart"].startswith("#!/bin/bash")
    assert "--tensor-parallel-size 2" in body["onstart"]
    # the harness was pointed at the verified endpoint, quick mode
    args = (env["tmp"] / "bb.args").read_text()
    assert "--endpoint http://5.6.7.8:41800/v1" in args and "--quick" in args
    # l80 published the file the harness wrote
    assert st.results_path in (env["tmp"] / "l80.args").read_text()
    # state persisted
    assert StateStore(env["cfg"].runs_dir).load(st.run_id).status == "torn_down"


def test_up_to_connect_keeps_box(env, client):
    st = rp.run_pipeline(env["cfg"], spec(), client, up_to="connect")
    assert st.status == "connected" and st.endpoint and not env["destroy"].called
    assert st.results_path is None


def test_keep_flag_skips_teardown(env, client):
    st = rp.run_pipeline(env["cfg"], spec(), client, keep=True)
    assert st.status == "published" and not env["destroy"].called


def test_publish_dry_run_does_not_record_url(env, client):
    st = rp.run_pipeline(env["cfg"], spec(), client, publish_dry_run=True)
    assert st.publish_url is None and st.status == "torn_down"
    assert "--dry-run" in (env["tmp"] / "l80.args").read_text()


def test_bench_failure_tears_down_and_raises(env, client, tmp_path):
    bad = fake_bin(tmp_path, "bb_bad", "exit 5\n")
    cfg = Config(api_key="k", api_base=BASE, l80_bin=env["cfg"].l80_bin, betterbench_bin=bad,
                 runs_dir=tmp_path / "runs")
    with pytest.raises(rp.PipelineError, match="exited 5"):
        rp.run_pipeline(cfg, spec(), client)
    assert env["destroy"].called
    st = rp.latest_state(StateStore(cfg.runs_dir))
    assert st.status == "torn_down"


def test_adopt_existing_instance_skips_rent(env, client):
    st = rp.run_pipeline(env["cfg"], spec(), client, instance=777, up_to="provision")
    assert st.instance_id == 777 and not env["asks"].called


def test_cost_guard(env, client, router):
    router.get(f"{BASE}/users/current/").mock(
        return_value=MockResponse(200, content=json.dumps({"id": 1, "credit": 5.0})))
    with pytest.raises(rp.PipelineError, match="balance"):
        rp.run_pipeline(env["cfg"], spec(), client)


def test_rent_fails_over_when_host_cannot_start_container(env, client, router):
    dead = {"id": 777, "actual_status": "created", "intended_status": "stopped",
            "status_msg": "Error response from daemon: nvidia-container-cli: device error"}
    router.get(f"{BASE}/instances/777").mock(
        return_value=MockResponse(200, content=json.dumps({"instances": dead})))
    destroyed_777 = router.delete(f"{BASE}/instances/777").mock(
        return_value=MockResponse(200, content="{}"))
    good = dict(INSTANCE, id=778)
    router.put(f"{BASE}/asks/12/").mock(
        return_value=MockResponse(200, content=json.dumps({"success": True, "new_contract": 778})))
    router.get(f"{BASE}/instances/778").mock(
        return_value=MockResponse(200, content=json.dumps({"instances": good})))
    router.post(f"{BASE}/instances/778/ssh/").mock(return_value=MockResponse(200, content="{}"))
    router.delete(f"{BASE}/instances/778").mock(return_value=MockResponse(200, content="{}"))
    st = rp.run_pipeline(env["cfg"], spec(), client, up_to="rent")
    assert destroyed_777.called            # the dead box was torn down
    assert st.instance_id == 778 and st.offer_id == 12 and st.status == "rented"
