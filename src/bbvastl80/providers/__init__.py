"""Serving providers: how a running box becomes *your model at /v1*.

M2 exposes just what renting needs — :func:`resolve_image` and
:func:`resolve_port`. M3 adds the full provider contract
(``prepare`` / ``launch`` / ``is_ready``) in :mod:`bbvastl80.providers.base`,
with :mod:`...engines` (presets) and :mod:`...custom` (your stack) as the two
implementations.
"""
from bbvastl80.providers.engines import DEFAULT_ENGINE_IMAGES, resolve_image, resolve_port

__all__ = ["resolve_image", "resolve_port", "DEFAULT_ENGINE_IMAGES"]
