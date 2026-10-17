"""Serving providers: how a running box becomes *your model at /v1*.

Two layers:

* **Rent time** — :func:`resolve_image` / :func:`resolve_port` tell Vast which
  image to boot and which port to expose (M2).
* **Provision time** — :func:`resolve_provider` returns a
  :class:`~bbvastl80.providers.base.Provider` (a curated engine preset or your
  custom stack) that drives the box to a live ``/v1`` (M3).

Commands reach the box over SSH (:func:`make_transport`); the provider's
bootstrap script additionally rides along as the instance's ``onstart`` so a
box starts serving on first boot. The Vast execute endpoint is constrained to
``ls``/``rm``/``du`` and is never used for provisioning.
"""
from bbvastl80.providers.base import (
    Box,
    ExecResult,
    ExecTransport,
    Provider,
    SSHExecTransport,
    VastExecTransport,
    make_transport,
)
from bbvastl80.providers.custom import CustomProvider
from bbvastl80.providers.engines import LLamaCppProvider, VLLMProvider, get_provider

_ENGINE_DEFAULTS = {
    "vllm": VLLMProvider.DEFAULT_IMAGE,
    "llamacpp": LLamaCppProvider.DEFAULT_IMAGE,
}


def resolve_image(serving) -> str:
    """The image to boot the box with (rent time). Takes the *serving block*."""
    if serving.mode == "custom":
        if not serving.custom or not serving.custom.image:
            raise ValueError("serving.custom.image is required when mode: custom")
        return serving.custom.image
    if not serving.engine:
        raise ValueError("serving.engine.engine is required when mode: engine")
    return serving.engine.image or _ENGINE_DEFAULTS.get(serving.engine.engine, serving.engine.image)


def resolve_port(serving) -> int:
    """The internal port the server listens on (rent time)."""
    return serving.port


def resolve_provider(spec) -> Provider:
    """The provider that will take a box to a live ``/v1`` (provision time)."""
    if spec.serving.mode == "custom":
        return CustomProvider()
    if not spec.serving.engine:
        raise ValueError("serving.engine.engine is required when mode: engine")
    return get_provider(spec.serving.engine.engine)


__all__ = [
    "Box", "ExecResult", "ExecTransport", "Provider",
    "SSHExecTransport", "VastExecTransport", "make_transport",
    "CustomProvider", "VLLMProvider", "LLamaCppProvider", "get_provider",
    "resolve_image", "resolve_port", "resolve_provider",
]
