"""M3: the provider + exec-transport layer, tested against fakes (no network)."""
import pytest

from bbvastl80.providers import resolve_image, resolve_provider
from bbvastl80.providers.base import (
    BOOTSTRAP_LOG, SERVER_LOG, SERVER_PID, STATUS_FILE,
    Box, ExecResult, ExecTransport, SSHExecTransport, VastExecTransport,
)
from bbvastl80.providers.custom import CustomProvider
from bbvastl80.providers.engines import LLamaCppProvider, VLLMProvider
from bbvastl80.spec import RunSpec, ServingSpec


class FakeTransport(ExecTransport):
    """Records every command; answers with a canned result."""

    name = "fake"

    def __init__(self, rc=0, out="ok\n"):
        self.rc, self.out = rc, out
        self.calls: list[str] = []
        self.detached: list[str] = []

    def run(self, box, cmd, timeout=120.0):
        self.calls.append(cmd)
        return ExecResult(self.rc, self.out)

    def run_detached(self, box, cmd, logfile):
        self.detached.append(cmd)


def make_spec(**kw) -> RunSpec:
    base = {
        "name": "t",
        "model": {"id": "m", "hf_repo": "org/m"},
        "gpu": {"names": ["NVIDIA L40S"], "count": 1},
    }
    base.update(kw)
    return RunSpec(**base)


def box(t: ExecTransport, **kw) -> Box:
    return Box(instance_id=1, transport=t, internal_port=8000, **kw)


# --- SSH transport: argv shape ------------------------------------------------
def test_ssh_transport_argv_uses_identity_host_port():
    t = SSHExecTransport("/k/id_ed25519")
    argv = t._cmd(box(t, ssh_host="1.2.3.4", ssh_port=41234), "true")
    assert argv[0] == "ssh" and argv[-2:] == ["root@1.2.3.4", "true"]
    assert "-i" in argv and argv[argv.index("-i") + 1] == "/k/id_ed25519"
    assert argv[argv.index("-p") + 1] == "41234"
    assert "BatchMode=yes" in argv and "StrictHostKeyChecking=no" in argv


def test_ssh_transport_without_host_fails_cleanly():
    t = SSHExecTransport()
    r = t.run(box(t), "true")
    assert not r.ok and "no SSH host" in r.stderr


def test_box_put_file_round_trips_via_base64():
    t = FakeTransport()
    box(t).put_file("echo 'hi'\n", "/tmp/bbvastl80/x.sh")
    cmd = t.calls[-1]
    assert "base64 -d > /tmp/bbvastl80/x.sh" in cmd and "chmod 755" in cmd
    import base64
    b64 = cmd.split("echo ")[1].split(" |")[0]
    assert base64.b64decode(b64).decode() == "echo 'hi'\n"


# --- Vast execute endpoint: constrained, never used for provisioning ---------
def test_vast_exec_refuses_anything_but_ls_rm_du():
    class C:
        def execute(self, iid, cmd, timeout=60):
            return "total 0\n"
    t = VastExecTransport(C())
    assert t.run(box(t), "ls -l /workspace").ok
    r = t.run(box(t), "echo hi")
    assert r.exit_code == 126 and "only runs ls/rm/du" in r.stderr
    r = t.run(box(t), "ls; curl localhost")  # shell metachars
    assert not r.ok
    with pytest.raises(RuntimeError):
        t.run_detached(box(t), "vllm serve x", "/tmp/l")


# --- vLLM preset: command building ------------------------------------------
def test_vllm_launch_command():
    cmd = VLLMProvider().launch_cmd(make_spec())
    assert cmd.startswith("vllm serve org/m")
    assert "--served-model-name m" in cmd
    assert "--host 0.0.0.0" in cmd and "--port 8000" in cmd


def test_vllm_launch_multi_gpu_adds_tp():
    cmd = VLLMProvider().launch_cmd(make_spec(gpu={"names": ["NVIDIA L40S"], "count": 2}))
    assert "--tensor-parallel-size 2" in cmd


def test_vllm_launch_engine_knobs():
    spec = make_spec(serving={"mode": "engine",
                              "engine": {"engine": "vllm", "max_model_len": 4096,
                                         "gpu_mem_util": 0.9, "extra_args": ["--enforce-eager"]}})
    cmd = VLLMProvider().launch_cmd(spec)
    assert "--max-model-len 4096" in cmd
    assert "--gpu-memory-utilization 0.9" in cmd
    assert "--enforce-eager" in cmd


def test_vllm_stepwise_launch_is_detached():
    t = FakeTransport()
    VLLMProvider().launch(box(t), make_spec())
    assert t.detached and "vllm serve org/m" in t.detached[-1]


def test_llamacpp_stages_url_weights():
    spec = make_spec(model={"id": "m", "source": "url", "url": "https://x/m.gguf"},
                     serving={"engine": {"engine": "llamacpp", "max_model_len": 2048}})
    p = LLamaCppProvider()
    prep = p.prepare_cmds(spec)
    assert any("curl -fsSL -o /bbvastl80/model.gguf https://x/m.gguf" in c for c in prep)
    cmd = p.launch_cmd(spec)
    assert "-m /bbvastl80/model.gguf" in cmd and "--alias m" in cmd and "-c 2048" in cmd


# --- custom provider ---------------------------------------------------------
def test_custom_provider_launch_and_health():
    t = FakeTransport(out='{"data": [{"id": "m"}]}')
    spec = make_spec(serving={"mode": "custom",
                              "custom": {"image": "my/img", "start": "python server.py"}})
    p = CustomProvider()
    p.launch(box(t), spec)
    assert t.detached[-1] == "python server.py"
    assert p.is_ready(box(t), spec) is True  # 200 + 'm' present in the body
    assert "curl -sf http://localhost:8000/v1/models" in t.calls[-1]


def test_custom_provider_setup_runs_in_order_and_stops_on_failure():
    from bbvastl80.vast.instance import ProvisionError
    t = FakeTransport()
    spec = make_spec(serving={"mode": "custom",
                              "custom": {"image": "my/img", "setup": ["a", "b"],
                                         "start": "s"}})
    CustomProvider().prepare(box(t), spec)
    assert t.calls == ["a", "b"]
    t2 = FakeTransport(rc=1, out="err")
    with pytest.raises(ProvisionError):
        CustomProvider().prepare(box(t2), spec)


# --- the bootstrap script -----------------------------------------------------
def test_bootstrap_script_is_idempotent_and_ordered():
    spec = make_spec(instance={"onstart": "echo user-hook"},
                     serving={"mode": "custom",
                              "custom": {"image": "i", "setup": ["apt-get install -y x"],
                                         "start": "python serve.py --port 8000",
                                         "health": {"path": "/health"}}})
    s = CustomProvider().script(spec, extra_onstart=spec.instance.onstart)
    assert s.startswith("#!/bin/bash")
    # exits early when already serving / launching
    assert "curl -sf http://localhost:8000/health" in s and "already serving; exit 0" in s
    assert f"kill -0 $(cat {SERVER_PID})" in s
    # order: user hook -> prepare -> launch
    i_hook, i_prep, i_launch = (s.index("echo user-hook"), s.index("apt-get install -y x"),
                                s.index("python serve.py --port 8000"))
    assert i_hook < i_prep < i_launch
    assert f"'prepare failed: apt-get install -y x' >{STATUS_FILE}; exit 1" in s
    assert f">{SERVER_LOG} 2>&1" in s and f"exec >>{BOOTSTRAP_LOG}" in s
    assert "setsid nohup bash -c 'python serve.py --port 8000'" in s


def test_bootstrap_script_quotes_launch_safely():
    spec = make_spec(serving={"engine": {"engine": "vllm", "extra_args": ["--chat-template", "x y"]}})
    import shlex
    launch = VLLMProvider().launch_cmd(spec)
    assert launch.endswith("--chat-template 'x y'")  # extra args are shell-quoted
    s = VLLMProvider().script(spec)
    assert f"setsid nohup bash -c {shlex.quote(launch)} >" in s  # and the whole launch again


# --- resolution --------------------------------------------------------------
def test_resolve_provider_engine_and_custom():
    assert type(resolve_provider(make_spec())) is VLLMProvider
    custom = make_spec(serving={"mode": "custom",
                                "custom": {"image": "x", "start": "s"}})
    assert type(resolve_provider(custom)) is CustomProvider


def test_resolve_image_default_and_override():
    eng = ServingSpec(mode="engine", engine={"engine": "vllm"})
    assert resolve_image(eng) == "vllm/vllm-openai:latest"
    eng2 = ServingSpec(mode="engine", engine={"engine": "vllm", "image": "my/vllm:1"})
    assert resolve_image(eng2) == "my/vllm:1"
