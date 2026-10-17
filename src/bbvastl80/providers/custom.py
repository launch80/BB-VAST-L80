"""The escape hatch: serve a model with *your* stack, not a curated preset.

A custom provider reads the spec's ``serving.custom`` block (``CustomServing``):
an ``image``, a ``setup`` list of shell commands (run in order on the box), a
``start`` command (the server), and a ``health`` probe (path + optional
substring). Everything runs on the box through the
:class:`~bbvastl80.providers.base.Box`, so it works with any transport.
"""
from __future__ import annotations

from .base import Box, Provider


class CustomProvider(Provider):
    """Drives a user-supplied image + commands. Stateless; reads the spec each
    call so it can be shared and resumed."""

    name = "custom"

    def prepare(self, box: Box, spec) -> None:
        cfg = spec.serving.custom
        for cmd in cfg.setup:
            r = box.run(cmd, timeout=1800)
            if not r.ok:
                from bbvastl80.vast.instance import ProvisionError
                raise ProvisionError(f"custom 'setup' step failed (rc={r.exit_code}):\n"
                                     f"  $ {cmd}\n{r.text()}")

    def launch(self, box: Box, spec) -> None:
        cfg = spec.serving.custom
        if not cfg.start:
            from bbvastl80.vast.instance import ProvisionError
            raise ProvisionError("a custom provider needs a 'start' command")
        box.run_detached(cfg.start)

    def is_ready(self, box: Box, spec) -> bool:
        cfg = spec.serving.custom
        r = box.run(f"curl -sf localhost:{spec.serving.port}{cfg.health.path}", timeout=15)
        if not r.ok:
            return False
        expects = cfg.health.expects or spec.serving.model_id or spec.model.id
        return (expects in r.stdout) if expects else True
