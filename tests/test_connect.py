"""M4: endpoint resolution + verification (respx for HTTP, fakes for ssh)."""
from __future__ import annotations

import json

import pytest
from respx.models import MockResponse

from bbvastl80 import connect as cn
from bbvastl80.providers.base import Box, SSHExecTransport
from bbvastl80.spec import RunSpec


def spec(**connect) -> RunSpec:
    return RunSpec(model={"id": "m", "hf_repo": "org/m"}, connect=connect or {"mode": "auto"})


def models(ids):
    return MockResponse(200, content=json.dumps({"data": [{"id": i} for i in ids]}))


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(cn.time, "sleep", lambda s: None)


def test_verify_v1_requires_v1_suffix_and_model(router, fast):
    with pytest.raises(cn.ConnectError, match="end in /v1"):
        cn.verify_v1("http://h:1/", "m", timeout_s=0)
    router.get("http://h:1/v1/models").mock(return_value=models(["other"]))
    with pytest.raises(cn.ConnectError, match="not 'm'"):
        cn.verify_v1("http://h:1/v1", "m", timeout_s=0)
    router.get("http://h:1/v1/models").mock(return_value=models(["m"]))
    cn.verify_v1("http://h:1/v1", "m", timeout_s=0)


def test_connect_direct_when_port_mapped_and_answering(router, fast):
    router.get("http://5.6.7.8:41800/v1/models").mock(return_value=models(["m"]))
    box = Box(instance_id=1, transport=SSHExecTransport(), host_ip="5.6.7.8",
              ssh_host="5.6.7.8", ssh_port=41122, ports={"8000": 41800})
    conn = cn.connect(box, spec())
    assert conn.mode == "direct" and conn.endpoint == "http://5.6.7.8:41800/v1"
    assert conn.tunnel_pid is None


def test_connect_falls_back_to_tunnel(router, fast, monkeypatch):
    # direct mapping exists but doesn't answer -> tunnel
    router.get("http://5.6.7.8:41800/v1/models").mock(return_value=MockResponse(502, content="x"))
    router.get("http://localhost:8000/v1/models").mock(return_value=models(["m"]))
    opened = {}

    class P:
        pid = 4242

    def fake_open(box, local_port, wait_s=30.0):
        opened["port"] = local_port
        return P()
    monkeypatch.setattr(cn, "open_tunnel", fake_open)
    monkeypatch.setattr(cn, "_port_open", lambda h, p, timeout=2.0: False)
    box = Box(instance_id=1, transport=SSHExecTransport(), host_ip="5.6.7.8",
              ssh_host="5.6.7.8", ssh_port=41122, ports={"8000": 41800})
    conn = cn.connect(box, spec(mode="auto", local_port=8000))
    assert conn.mode == "tunnel" and conn.endpoint == "http://localhost:8000/v1"
    assert conn.tunnel_pid == 4242 and opened["port"] == 8000


def test_connect_direct_mode_does_not_fall_back(router, fast):
    box = Box(instance_id=1, transport=SSHExecTransport(), host_ip="5.6.7.8",
              ssh_host="5.6.7.8", ssh_port=41122, ports={})
    with pytest.raises(cn.ConnectError, match="no public mapping"):
        cn.connect(box, spec(mode="direct"))


def test_tunnel_argv_shape():
    box = Box(instance_id=1, transport=SSHExecTransport("/k"), host_ip="5.6.7.8",
              ssh_host="5.6.7.8", ssh_port=41122, internal_port=8000)
    t = box.transport
    argv = t._cmd(box, "-N", "-o", "ExitOnForwardFailure=yes", "-L", "9000:localhost:8000")
    user_host = "root@5.6.7.8"
    argv = [a for a in argv if a != user_host] + [user_host]
    assert argv[-1] == "root@5.6.7.8" and "-N" in argv and "9000:localhost:8000" in argv
    assert argv[argv.index("-p") + 1] == "41122"
