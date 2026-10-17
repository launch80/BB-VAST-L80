"""M3 orchestration: render_bootstrap, build_box (ssh target), provision flow."""
from __future__ import annotations

import pytest

from bbvastl80 import bootstrap as bs
from bbvastl80.providers.base import BOOTSTRAP_PATH, ExecResult, ExecTransport
from bbvastl80.spec import RunSpec
from bbvastl80.vast.instance import ProvisionError
from bbvastl80.vast.models import Instance


def make_spec(**kw) -> RunSpec:
    base = {"model": {"id": "m", "hf_repo": "org/m"}}
    base.update(kw)
    return RunSpec(**base)


class Scripted(ExecTransport):
    """Answers each command from a rule list: (substring, rc, stdout)."""

    name = "scripted"

    def __init__(self, rules):
        self.rules, self.calls = rules, []

    def run(self, box, cmd, timeout=120.0):
        self.calls.append(cmd)
        for sub, rc, out in self.rules:
            if sub in cmd:
                return ExecResult(rc, out)
        return ExecResult(0, "")

    def run_detached(self, box, cmd, logfile):
        self.calls.append("DETACHED " + cmd)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(bs.time, "sleep", lambda s: None)


@pytest.fixture
def transport(monkeypatch):
    holder = {}

    def fake_make_transport(spec, client=None):
        return holder["t"]
    monkeypatch.setattr(bs, "make_transport", fake_make_transport)
    return holder


def running_instance(**kw):
    base = dict(id=7, actual_status="running", public_ipaddr="5.6.7.8",
                ports={"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41122"}],
                       "8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41800"}]})
    base.update(kw)
    return Instance(**base)


def test_render_bootstrap_default_onstart_and_opt_out():
    s = bs.render_bootstrap(make_spec())
    assert s and s.startswith("#!/bin/bash") and "vllm serve org/m" in s
    assert bs.render_bootstrap(make_spec(instance={"bootstrap": "ssh"})) is None


def test_build_box_resolves_direct_ssh_and_ports(transport):
    transport["t"] = Scripted([])
    box = bs.build_box(None, running_instance(), make_spec())
    assert (box.ssh_host, box.ssh_port) == ("5.6.7.8", 41122)
    assert box.ports == {"8000": 41800, "22": 41122}
    assert box.url() == "http://5.6.7.8:41800"


def test_build_box_falls_back_to_vast_ssh_proxy(transport):
    transport["t"] = Scripted([])
    inst = running_instance(ports={}, ssh_host="ssh5.vast.ai", ssh_port=12345)
    box = bs.build_box(None, inst, make_spec())
    assert (box.ssh_host, box.ssh_port) == ("ssh5.vast.ai", 12345)
    assert box.url() is None


def test_provision_short_circuits_when_already_serving(transport, no_sleep):
    t = Scripted([("curl -sf http://localhost:8000/v1/models", 0, '{"data":[{"id":"m"}]}')])
    transport["t"] = t
    box = bs.provision(None, running_instance(), make_spec())
    assert box.ssh_port == 41122
    assert not any(BOOTSTRAP_PATH in c for c in t.calls)  # nothing pushed


def test_provision_pushes_script_then_waits_for_v1(transport, no_sleep):
    health = {"n": 0}

    class T(Scripted):
        def run(self, box, cmd, timeout=120.0):
            if "curl -sf http://localhost:8000/v1/models" in cmd:
                health["n"] += 1
                self.calls.append(cmd)
                return ExecResult(0 if health["n"] >= 3 else 7, "")
            return super().run(box, cmd, timeout)

    t = T([(f"bash {BOOTSTRAP_PATH}", 0, "launched"), ("cat /tmp/bbvastl80/status", 0, "launched")])
    transport["t"] = t
    bs.provision(None, running_instance(), make_spec())
    pushed = [c for c in t.calls if f"base64 -d > {BOOTSTRAP_PATH}" in c]
    assert len(pushed) == 1
    assert any(f"nohup bash {BOOTSTRAP_PATH}" in c for c in t.calls)
    assert health["n"] >= 3


def test_provision_reports_prepare_failure(transport, no_sleep):
    t = Scripted([
        ("curl -sf", 7, ""),
        (f"bash {BOOTSTRAP_PATH}", 0, "prepare failed: pip install vllm"),
    ])
    transport["t"] = t
    with pytest.raises(ProvisionError, match="prepare failed: pip install vllm"):
        bs.provision(None, running_instance(), make_spec())


def test_provision_needs_ssh_host(transport, no_sleep):
    transport["t"] = Scripted([])
    inst = running_instance(ports={}, ssh_host=None, ssh_port=None)
    with pytest.raises(ProvisionError, match="no SSH host"):
        bs.provision(None, inst, make_spec())


def test_wait_for_ssh_gives_up_with_hint(transport, no_sleep):
    t = Scripted([("true", 255, "Permission denied (publickey)")])
    transport["t"] = t
    box = bs.build_box(None, running_instance(), make_spec())
    with pytest.raises(ProvisionError, match="public key registered"):
        bs.wait_for_ssh(box, timeout_min=0.001)
