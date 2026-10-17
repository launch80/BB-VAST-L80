"""M3: the provider + exec-transport layer, tested against fakes (no network)."""
import pytest

from bbvastl80.providers import resolve_image, resolve_provider
from bbvastl80.providers.base import Box, VastExecTransport
from bbvastl80.providers.custom import CustomProvider
from bbvastl80.providers.engines import VLLMProvider
from bbvastl80.spec import RunSpec, ServingSpec


class FakeClient:
    """Stands in for VastClient.execute: returns a canned (marker-bearing) blob."""

    def __init__(self, output="hello\n__BBRC__0"):
        self.output = output
        self.calls: list[str] = []

    def execute(self, iid, cmd, timeout=60):
        self.calls.append(cmd)
        return self.output


def make_spec(**kw) -> RunSpec:
    base = {
        "name": "t",
        "model": {"id": "m", "hf_repo": "org/m"},
        "gpu": {"names": ["NVIDIA L40S"], "count": 1},
    }
    base.update(kw)
    return RunSpec(**base)


# --- transport: exit-code capture --------------------------------------------
def test_vast_transport_parses_and_strips_marker():
    c = FakeClient(output="line1\nline2\n__BBRC__0")
    box = Box(instance_id=1, transport=VastExecTransport(c))
    r = box.run("echo hi")
    assert r.ok and r.exit_code == 0
    assert "line1" in r.stdout
    assert "__BBRC__" not in r.stdout  # marker removed from the body


def test_vast_transport_nonzero_exit():
    c = FakeClient(output="boom\n__BBRC__1")
    box = Box(instance_id=1, transport=VastExecTransport(c))
    r = box.run("false")
    assert not r.ok and r.exit_code == 1


def test_vast_transport_missing_marker_defaults_ok():
    c = FakeClient(output="just output, no marker")
    box = Box(instance_id=1, transport=VastExecTransport(c))
    r = box.run("echo x")
    assert r.exit_code == 0  # no marker -> assume success, keep the text


# --- vLLM preset: command building ------------------------------------------
def test_vllm_launch_command():
    c = FakeClient()
    box = Box(instance_id=1, transport=VastExecTransport(c), internal_port=8000)
    VLLMProvider().launch(box, make_spec())
    cmd = c.calls[-1]
    assert "vllm serve org/m" in cmd
    assert "--served-model-name m" in cmd
    assert "--port 8000" in cmd
    assert "nohup" in cmd  # launched detached


def test_vllm_launch_multi_gpu_adds_tp():
    c = FakeClient()
    box = Box(instance_id=1, transport=VastExecTransport(c), internal_port=8000)
    VLLMProvider().launch(box, make_spec(gpu={"names": ["NVIDIA L40S"], "count": 2}))
    assert "--tensor-parallel-size 2" in c.calls[-1]


def test_vllm_launch_engine_knobs():
    c = FakeClient()
    box = Box(instance_id=1, transport=VastExecTransport(c), internal_port=8000)
    spec = make_spec(serving={"mode": "engine",
                              "engine": {"engine": "vllm", "max_model_len": 4096,
                                         "gpu_mem_util": 0.9, "extra_args": ["--enforce-eager"]}})
    VLLMProvider().launch(box, spec)
    cmd = c.calls[-1]
    assert "--max-model-len 4096" in cmd
    assert "--gpu-memory-utilization 0.9" in cmd
    assert "--enforce-eager" in cmd


# --- custom provider ---------------------------------------------------------
def test_custom_provider_launch_and_health():
    c = FakeClient(output='{"data": [{"id": "m"}]}\n__BBRC__0')
    box = Box(instance_id=1, transport=VastExecTransport(c), internal_port=8000)
    spec = make_spec(serving={"mode": "custom",
                              "custom": {"image": "my/img", "start": "python server.py"}})
    p = CustomProvider()
    p.launch(box, spec)
    assert "python server.py" in c.calls[-1]
    assert p.is_ready(box, spec) is True  # 200 + 'm' present in the body


def test_custom_provider_setup_runs_in_order_and_stops_on_failure():
    from bbvastl80.vast.instance import ProvisionError
    c = FakeClient(output="ok\n__BBRC__0")
    box = Box(instance_id=1, transport=VastExecTransport(c), internal_port=8000)
    spec = make_spec(serving={"mode": "custom",
                              "custom": {"image": "my/img", "setup": ["a", "b"],
                                         "start": "s"}})
    CustomProvider().prepare(box, spec)
    assert c.calls == ["a\necho \"__BBRC__$?\"", "b\necho \"__BBRC__$?\""]
    # a failing step raises
    c2 = FakeClient(output="err\n__BBRC__1")
    box2 = Box(instance_id=1, transport=VastExecTransport(c2), internal_port=8000)
    with pytest.raises(ProvisionError):
        CustomProvider().prepare(box2, spec)


# --- resolution --------------------------------------------------------------
def test_resolve_provider_engine_and_custom():
    assert type(resolve_provider(make_spec())) is VLLMProvider
    custom = make_spec(serving={"mode": "custom",
                                "custom": {"image": "x", "start": "s"}})
    assert type(resolve_provider(custom)) is CustomProvider


def test_resolve_image_default_and_override():
    eng = ServingSpec(mode="engine", engine={"engine": "vllm"})
    assert resolve_image(eng).startswith("ghcr.io/vllm")
    eng2 = ServingSpec(mode="engine", engine={"engine": "vllm", "image": "my/vllm:1"})
    assert resolve_image(eng2) == "my/vllm:1"
