"""Curated engine presets. Each is a :class:`~bbvastl80.providers.base.Provider`
that knows how to serve a model with one specific runtime, reading its knobs from
the spec's ``serving.engine`` block (``EngineServing``) and the model from
``spec.model``.

* **vLLM** — the default. ``vllm serve`` pulls the model straight from the Hub,
  so ``prepare`` only installs vLLM when the box runs a bare image.
* **llama.cpp** — GGUF; ``prepare`` stages the weights file if it's a URL.

Both speak the same OpenAI-compatible ``/v1`` contract downstream, and both
describe themselves as *commands* so the base class can render them into the
onstart / SSH bootstrap script.
"""
from __future__ import annotations

import shlex

from .base import Provider


class VLLMProvider(Provider):
    name = "vllm"
    DEFAULT_IMAGE = "vllm/vllm-openai:latest"  # Docker Hub (ghcr.io has no public image)

    def prepare_cmds(self, spec) -> list[str]:
        # The official image ships vLLM; only install if it's a bare image.
        return ["command -v vllm >/dev/null 2>&1 || python -c 'import vllm' 2>/dev/null "
                "|| pip install vllm"]

    def launch_cmd(self, spec) -> str:
        eng = spec.serving.engine
        repo = spec.model.hf_repo or spec.model.id
        parts = [
            "vllm serve", repo,
            f"--served-model-name {spec.model.id}",
            "--host 0.0.0.0",
            f"--port {spec.serving.port}",
        ]
        if (spec.gpu.count or 1) > 1:
            parts.append(f"--tensor-parallel-size {spec.gpu.count}")
        if eng.max_model_len:
            parts.append(f"--max-model-len {eng.max_model_len}")
        if eng.gpu_mem_util:
            parts.append(f"--gpu-memory-utilization {eng.gpu_mem_util}")
        parts.extend(shlex.quote(a) for a in (eng.extra_args or []))
        return " ".join(parts)


class LLamaCppProvider(Provider):
    name = "llamacpp"
    DEFAULT_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
    WEIGHTS = "/bbvastl80/model.gguf"

    def _source(self, spec) -> str:
        return spec.model.url or spec.model.hf_repo or ""

    def _weights(self, spec) -> str:
        src = self._source(spec)
        return self.WEIGHTS if src.startswith(("http://", "https://")) else (src or self.WEIGHTS)

    def prepare_cmds(self, spec) -> list[str]:
        cmds = ["command -v llama-server >/dev/null 2>&1 || "
                "(apt-get update && apt-get install -y llama.cpp) || pip install llama.cpp"]
        src = self._source(spec)
        if src.startswith(("http://", "https://")):
            cmds.append(f"test -s {self.WEIGHTS} || (mkdir -p $(dirname {self.WEIGHTS}) && "
                        f"curl -fsSL -o {self.WEIGHTS} {src})")
        return cmds

    def launch_cmd(self, spec) -> str:
        eng = spec.serving.engine
        parts = [
            "llama-server", f"-m {self._weights(spec)}",
            "--host 0.0.0.0", f"--port {spec.serving.port}",
            f"--alias {spec.model.id}",
        ]
        if eng.max_model_len:
            parts.append(f"-c {eng.max_model_len}")
        parts.extend(shlex.quote(a) for a in (eng.extra_args or []))
        return " ".join(parts)


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
