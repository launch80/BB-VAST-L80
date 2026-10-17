"""Turn a :class:`~bbvastl80.spec.GPUFilter` into a Vast search, and pick a winner.

The ``POST /bundles`` request takes each filter as a *top-level* object with an
operator (``{"in": [...]}``, ``{"gte": ...}``, ``{"lte": ...}``, ``{"eq": ...}``).
``gpu_total_ram`` is expressed in mebibytes, so ``min_ram_gb`` is scaled by 1024.
"""
from __future__ import annotations

from bbvastl80.spec import GPUFilter
from bbvastl80.vast.models import Offer


def build_search_body(f: GPUFilter, *, limit: int | None = None,
                      min_disk_gb: float | None = None) -> dict:
    """``min_disk_gb`` (the spec's ``instance.disk_gb``) drops offers whose
    host can't give us that much disk — renting one of those 400s."""
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
    if min_disk_gb is not None:
        flt["disk_space"] = {"gte": float(min_disk_gb)}
    flt["rentable"] = {"eq": True}
    if flt:
        body.update(flt)
    return body


def rank_offers(offers: list[Offer], pick: str = "cheapest") -> list[Offer]:
    """Return offers ordered by the pick strategy.

    ``cheapest`` sorts ascending by $/hr; ``first`` keeps API order; a 1-based
    integer starts the ranking at that index (into the cheapest-first order) and
    continues with the rest — useful for trying the N-th-cheapest and falling
    through to the next if it has just been taken.
    """
    if pick in ("first", ""):
        return list(offers)
    ranked = sorted(offers, key=lambda o: (o.dph_total is None, o.dph_total or 0.0))
    if pick == "cheapest":
        return ranked
    try:
        idx = int(pick)
    except ValueError:
        raise ValueError(
            f"gpu.pick must be 'cheapest', 'first', or a 1-based index (got {pick!r})")
    if not 1 <= idx <= len(ranked):
        raise ValueError(f"gpu.pick index {idx} out of range 1..{len(ranked)}")
    return ranked[idx - 1:]


def pick_offer(offers: list[Offer], pick: str = "cheapest") -> Offer | None:
    """Select a single offer (the first of :func:`rank_offers`)."""
    ranked = rank_offers(offers, pick)
    return ranked[0] if ranked else None
