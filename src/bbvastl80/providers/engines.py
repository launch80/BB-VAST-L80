"""Engine presets: default image + (M3) install/launch commands.

The images here are reasonable defaults; M3 validates each against a real
bootstrap and may swap them. For ``mode: custom`` the image always comes from
the spec.
"""
from __future__ import annotations

from bbvastl80.spec import ServingSpec

DEFAULT_ENGINE_IMAGES: dict[str, str] = {
    "vllm": "ghcr.io/vllm/vllm-openai:latest",
    "llamacpp": "ghcr.io/ggml-org/llama.cpp:latest",
    "sglang": "ghcr.io/sglang/sglang:latest",
    "tgi": "ghcr.io/huggingface/tgi:latest",
    "ollama": "ollama/ollama:latest",
}


def resolve_image(serving: ServingSpec) -> str:
    """The docker image the rented box should run."""
    if serving.mode == "custom":
        if not serving.custom or not serving.custom.image:
            raise ValueError("serving.mode=custom requires serving.custom.image")
        return serving.custom.image
    if serving.engine is None:
        raise ValueError("serving.mode=engine requires serving.engine")
    if serving.engine.image:
        return serving.engine.image
    return DEFAULT_ENGINE_IMAGES[serving.engine.engine]


def resolve_port(serving: ServingSpec) -> int:
    return serving.port
