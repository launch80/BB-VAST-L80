"""Turn a :class:`~bbvastl80.spec.GPUFilter` into a Vast search, and pick a winner.

The ``POST /bundles`` request takes each filter as a *top-level* object with an
operator (``{"in": [...]}``, ``{"gte": ...}``, ``{"lte": ...}``, ``{"eq": ...}``).
``gpu_total_ram`` is expressed in mebibytes, so ``min_ram_gb`` is scaled by 1024.
"""
from __future__ import annotations

from bbvastl80.spec import GPUFilter
from bbvastl80.vast.models import Offer


def build_search_body(f: GPUFilter, *, limit: int | None = None) -> dict:
    body: dict = {}
    body["limit"] = limit if limit is not None else f.limit
    # ``order`` is a *list of [field, direction] tuples* on the real API
    # (e.g. [["dph_total", "asc"]]); ``type`` is a plain enum value
    # (on_demand / on-demand / ondemand / ask / bid / reserved). Neither is
    # an operator object, unlike the other filters below.
    body["order"] = [["dph_total", "asc"]]  # cheapest first, server-side
    if f.type:
        body["type"] = f.type

    flt: dict[str, dict] = {}
    if f.names:
        flt["gpu_name"] = {"in": f.names} if len(f.names) > 1 else {"eq": f.names[0]}
    if f.count is not None:
        flt["num_gpus"] = {"eq": f.count}
    if f.min_ram_gb is not None:
        flt["gpu_total_ram"] = {"gte": int(f.min_ram_gb * 1024)}
    if f.max_dph is not None:
        flt["dph_total"] = {"lte": f.max_dph}
    if f.min_reliability is not None:
        flt["reliability"] = {"gte": f.min_reliability}
    if f.verified is not None:
        flt["verified"] = {"eq": f.verified}
    if f.datacenter:
        flt["datacenter"] = {"eq": f.datacenter}
    if f.geolocation:
        flt["geolocation"] = {"eq": f.geolocation}
    if flt:
        body.update(flt)
    return body


def pick_offer(offers: list[Offer], pick: str = "cheapest") -> Offer | None:
    """Select an offer. ``cheapest`` sorts by $/hr; ``first`` keeps order; an
    integer picks the 1-based index into the (sorted) list."""
    if not offers:
        return None
    ranked = sorted(offers, key=lambda o: (o.dph_total is None, o.dph_total or 0.0))
    if pick in ("cheapest", "first", ""):
        return ranked[0] if pick == "cheapest" else offers[0]
    try:
        idx = int(pick)
    except ValueError:
        raise ValueError(f"gpu.pick must be 'cheapest', 'first', or a 1-based index (got {pick!r})")
    if not 1 <= idx <= len(ranked):
        raise ValueError(f"gpu.pick index {idx} out of range 1..{len(ranked)}")
    return ranked[idx - 1]
