"""The escape hatch: serve a model with *your* stack, not a curated preset.

A custom provider reads the spec's ``serving.custom`` block (``CustomServing``):
an ``image``, a ``setup`` list of shell commands (run in order on the box), a
``start`` command (the server), and a ``health`` probe (path + optional
substring). The commands are rendered into the same bootstrap script as the
presets, so they run from ``onstart`` on first boot and/or over SSH.
"""
from __future__ import annotations

from .base import Box, Provider


class CustomProvider(Provider):
    """Drives a user-supplied image + commands. Stateless; reads the spec each
    call so it can be shared and resumed."""

    name = "custom"

    def prepare_cmds(self, spec) -> list[str]:
        return list(spec.serving.custom.setup)

    def launch_cmd(self, spec) -> str:
        cfg = spec.serving.custom
        if not cfg.start:
            from bbvastl80.vast.instance import ProvisionError
            raise ProvisionError("a custom provider needs a 'start' command")
        return cfg.start

    def health_cmd(self, spec) -> str:
        return f"curl -sf localhost:{spec.serving.port}{spec.serving.custom.health.path}"

    def is_ready(self, box: Box, spec) -> bool:
        cfg = spec.serving.custom
        r = box.run(self.health_cmd(spec), timeout=15)
        if not r.ok:
            return False
        expects = cfg.health.expects or spec.serving.model_id or spec.model.id
        return (expects in r.stdout) if expects else True
