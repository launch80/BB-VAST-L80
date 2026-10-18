from __future__ import annotations

import pytest
from pydantic import ValidationError

from bbvastl80.spec import RunSpec, load_spec

MINIMAL = {
    "name": "t",
    "model": {"id": "Qwen/Qwen3-8B", "source": "hf", "hf_repo": "Qwen/Qwen3-8B"},
}


def test_minimal_spec_defaults_to_vllm():
    s = RunSpec(**MINIMAL)
    assert s.serving.mode == "engine"
    assert s.serving.engine is not None
    assert s.serving.engine.engine == "vllm"
    assert s.serving.model_id == "Qwen/Qwen3-8B"  # auto-filled
    assert s.serving.port == 8000


def test_engine_spec_explicit():
    s = RunSpec(**{
        "name": "t",
        "model": {"id": "m", "source": "hf", "hf_repo": "r"},
        "serving": {
            "mode": "engine", "port": 8080,
            "engine": {"engine": "llamacpp", "max_model_len": 4096},
        },
    })
    assert s.serving.engine.engine == "llamacpp"
    assert s.serving.engine.max_model_len == 4096
    assert s.serving.port == 8080


@pytest.mark.parametrize("engine", ["sglang", "tgi", "ollama"])
def test_engine_without_preset_is_rejected(engine):
    with pytest.raises(ValidationError, match="has no preset yet"):
        RunSpec(**{"name": "t", "model": {"id": "m"},
                   "serving": {"engine": {"engine": engine}}})


def test_custom_spec_requires_block():
    with pytest.raises(ValidationError):
        RunSpec(**{
            "name": "t",
            "model": {"id": "m", "source": "hf"},
            "serving": {"mode": "custom", "port": 8000},  # no custom block
        })


def test_custom_spec_valid():
    s = RunSpec(**{
        "name": "t",
        "model": {"id": "my-model", "source": "hf", "hf_repo": "a/b"},
        "serving": {
            "mode": "custom", "port": 9000,
            "custom": {
                "image": "registry.example/svc:latest",
                "setup": ["pip install -r /app/requirements.txt"],
                "start": "python /app/serve.py --port 9000",
                "health": {"path": "/v1/models", "expects": "my-model"},
            },
        },
    })
    assert s.serving.mode == "custom"
    assert s.serving.custom.start.startswith("python")


def test_load_spec_from_file(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(
        "name: t\nmodel:\n  id: Qwen/Qwen3-8B\n  source: hf\n  hf_repo: Qwen/Qwen3-8B\n"
    )
    s = load_spec(str(p))
    assert s.name == "t"
    assert s.serving.engine.engine == "vllm"


def test_gpu_filter_defaults():
    s = RunSpec(**MINIMAL)
    assert s.gpu.count == 1
    assert s.gpu.type is None  # not pinned by default
    assert s.gpu.pick == "cheapest"
