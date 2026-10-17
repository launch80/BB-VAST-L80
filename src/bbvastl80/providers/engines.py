"""Curated engine presets. Each is a :class:`~bbvastl80.providers.base.Provider`
that knows how to serve a model with one specific runtime, reading its knobs from
the spec's ``serving.engine`` block (``EngineServing``) and the model from
``spec.model``.

* **vLLM** — the default. ``vllm serve`` pulls the model straight from the Hub,
  so ``prepare`` is a no-op when the box runs the official vLLM image.
* **llama.cpp** — GGUF; ``prepare`` stages the weights file if it's a URL.

Both speak the same OpenAI-compatible ``/v1`` contract downstream.
"""
from __future__ import annotations

from .base import Box, Provider


def _health_cmd(spec) -> str:
    path = "/v1/models"
    if spec.serving.mode == "custom" and spec.serving.custom:
        path = spec.serving.custom.health.path
    return f"curl -sf localhost:{spec.serving.port}{path}"


class VLLMProvider(Provider):
    name = "vllm"
    DEFAULT_IMAGE = "ghcr.io/vllm/vllm-openai:latest"

    def prepare(self, box: Box, spec) -> None:
        # The official image ships vLLM; only install if it's a bare image.
        r = box.run("command -v vllm || python -c 'import vllm' 2>/dev/null", timeout=60)
        if not r.ok:
            box.run("pip install vllm", timeout=1800)

    def launch(self, box: Box, spec) -> None:
        eng = spec.serving.engine
        repo = spec.model.hf_repo or spec.model.id
        parts = [
            "vllm serve", repo,
            f"--served-model-name {spec.model.id}",
            f"--port {spec.serving.port}",
        ]
        if (spec.gpu.count or 1) > 1:
            parts.append(f"--tensor-parallel-size {spec.gpu.count}")
        if eng.max_model_len:
            parts.append(f"--max-model-len {eng.max_model_len}")
        if eng.gpu_mem_util:
            parts.append(f"--gpu-memory-utilization {eng.gpu_mem_util}")
        parts.extend(eng.extra_args or [])
        box.run_detached(" ".join(parts))

    def is_ready(self, box: Box, spec) -> bool:
        return box.run(_health_cmd(spec), timeout=15).ok


class LLamaCppProvider(Provider):
    name = "llamacpp"
    DEFAULT_IMAGE = "ghcr.io/ggml-org/llama.cpp:latest"
    WEIGHTS = "/bbvastl80/model.gguf"

    def _weights(self, spec) -> str:
        p = spec.model.path or ""
        return self.WEIGHTS if p.startswith(("http://", "https://")) else (p or self.WEIGHTS)

    def prepare(self, box: Box, spec) -> None:
        r = box.run("command -v llama-server", timeout=30)
        if not r.ok:
            box.run("apt-get update && apt-get install -y llama.cpp 2>/dev/null "
                    "|| pip install llama.cpp", timeout=1800)
        src = spec.model.path or ""
        if src.startswith(("http://", "https://")):
            box.run(f"mkdir -p $(dirname {self.WEIGHTS}) && curl -fsSL -o {self.WEIGHTS} {src}",
                    timeout=3600)

    def launch(self, box: Box, spec) -> None:
        eng = spec.serving.engine
        parts = [
            "llama-server", f"-m {self._weights(spec)}",
            "--host 0.0.0.0", f"--port {spec.serving.port}",
            f"--name {spec.model.id}",
        ]
        if eng.max_model_len:
            parts.append(f"-c {eng.max_model_len}")
        parts.extend(eng.extra_args or [])
        box.run_detached(" ".join(parts))

    def is_ready(self, box: Box, spec) -> bool:
        return box.run(_health_cmd(spec), timeout=15).ok


# --- registry (keys match the spec's engine Literal) -------------------------
def get_provider(name: str) -> Provider:
    table = {
        "vllm": VLLMProvider(),
        "llamacpp": LLamaCppProvider(),
    }
    if name not in table:
        raise KeyError(f"engine {name!r} has no preset yet; available: {', '.join(sorted(table))}. "
                       "Use mode: custom for anything else.")
    return table[name]
