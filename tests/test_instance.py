from __future__ import annotations

import json

from respx.models import MockResponse

from conftest import BASE_V1
from bbvastl80.spec import CustomServing, EngineServing, InstanceSpec, ServingSpec
from bbvastl80.vast.instance import ProvisionError, rent, rent_body, wait_for_running
from bbvastl80.vast.models import Instance, Offer


def engine_serving():
    return ServingSpec(model_id="m", port=8000, engine=EngineServing(engine="vllm"))


def js(obj):
    return json.dumps(obj)


def test_rent_body_engine():
    body = rent_body(Offer(id=1, gpu_name="H100"), engine_serving(), InstanceSpec())
    assert body == {
        "image": "ghcr.io/vllm/vllm-openai:latest", "disk": 200, "runtype": "ssh",
        "label": "bbvastl80", "args": ["-p", "8000:8000"],
    }


def test_rent_body_custom_uses_custom_image():
    s = ServingSpec(model_id="m", port=9000, custom=CustomServing(
        image="registry.example/svc:latest", start="python serve.py --port 9000"))
    body = rent_body(Offer(id=1, gpu_name="H100"), s, InstanceSpec(price=1.5, onstart="echo hi"))
    assert body["image"] == "registry.example/svc:latest"
    assert body["price"] == 1.5
    assert body["onstart"] == "echo hi"


def test_rent_returns_instance_id(client, router):
    router.post(f"{BASE_V1}/asks/11/").mock(
        return_value=MockResponse(200, content=js({"new_contract": 555, "contract_id": 555})))
    inst = rent(client, Offer(id=11, gpu_name="H100"), engine_serving(), InstanceSpec())
    assert inst == 555


def test_wait_for_running_reaches_running(client, router):
    provisioning = {"id": 555, "status": "provisioning", "type": 0}
    running = {"id": 555, "status": "running", "type": 0, "public_ip": "1.2.3.4",
               "ports": {"8000": 44321, "22": 44322}}
    router.get(f"{BASE_V1}/instances/555").mock(side_effect=[
        MockResponse(200, content=js(provisioning)),
        MockResponse(200, content=js(running)),
    ])
    inst = wait_for_running(client, 555, timeout_min=0.05, poll_sec=0.01)
    assert inst.is_running
    assert inst.host_ip == "1.2.3.4"
    assert inst.external_port(8000) == 44321


def test_wait_for_running_survives_404(client, router):
    running = {"id": 555, "status": "running", "type": 0, "public_ip": "9.9.9.9"}
    router.get(f"{BASE_V1}/instances/555").mock(side_effect=[
        MockResponse(404, content=js({"detail": "not found"})),
        MockResponse(200, content=js(running)),
    ])
    inst = wait_for_running(client, 555, timeout_min=0.05, poll_sec=0.01)
    assert inst.host_ip == "9.9.9.9"


def test_wait_for_running_failed(client, router):
    router.get(f"{BASE_V1}/instances/555").mock(
        return_value=MockResponse(200, content=js({"id": 555, "status": "canceled", "type": 0})))
    import pytest
    with pytest.raises(ProvisionError):
        wait_for_running(client, 555, timeout_min=0.05, poll_sec=0.01)


def test_external_port_prefers_env():
    inst = Instance(id=1, status="running", public_ip="1.2.3.4",
                    env={"VAST_TCP_PORT_8000": "60000"}, ports={"8000": 44321})
    assert inst.external_port(8000) == 60000  # env wins
