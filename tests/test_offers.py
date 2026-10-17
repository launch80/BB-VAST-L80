from __future__ import annotations

from bbvastl80.spec import GPUFilter
from bbvastl80.vast.models import Offer
from bbvastl80.vast.offers import build_search_body, pick_offer, rank_offers


def make_offer(i, dph, **kw):
    base = dict(id=i, gpu_name="H100_SXM5", num_gpus=1, total_ram=143360,
                total_vram=143360, dph_total=dph, rel=0.99, verified=True,
                type="on-demand", datacenter="us", label=f"offer-{i}")
    base.update(kw)
    return Offer(**base)


def test_build_search_body_full():
    g = GPUFilter(names=["H100_SXM5", "A100_SXM4_80GB"], count=1, min_ram_gb=80,
                  max_dph=3.0, type="ondemand", verified=True, min_reliability=0.99,
                  datacenter="us-east-1")
    body = build_search_body(g)
    assert body["gpu_name"] == {"in": ["H100_SXM5", "A100_SXM4_80GB"]}
    assert body["num_gpus"] == {"eq": 1}
    assert body["gpu_total_ram"] == {"gte": 80 * 1024}  # MB
    assert body["dph_total"] == {"lte": 3.0}
    assert body["type"] == "ondemand"  # plain enum, not an operator object
    assert body["verified"] == {"eq": True}  # operator object (dict)
    assert body["reliability"] == {"gte": 0.99}
    assert body["datacenter"] == {"eq": "us-east-1"}
    assert body["order"] == [["dph_total", "asc"]]  # list of [field, dir] tuples


def test_build_search_body_minimal_defaults_to_one_gpu():
    body = build_search_body(GPUFilter())
    assert body["num_gpus"] == {"eq": 1}
    assert body["order"] == [["dph_total", "asc"]]
    assert body["limit"] == 40


def test_pick_cheapest():
    offers = [make_offer(1, 2.0), make_offer(2, 0.9), make_offer(3, 1.5)]
    assert pick_offer(offers, "cheapest").id == 2


def test_pick_first():
    offers = [make_offer(1, 2.0), make_offer(2, 0.9)]
    assert pick_offer(offers, "first").id == 1


def test_rank_offers_cheapest_orders_and_index_starts():
    offers = [make_offer(1, 2.0), make_offer(2, 0.5), make_offer(3, 1.0)]
    assert [o.id for o in rank_offers(offers, "cheapest")] == [2, 3, 1]
    # index 2 starts at the 2nd-cheapest and continues with the rest
    assert [o.id for o in rank_offers(offers, 2)] == [3, 1]
    # 'first' keeps raw API order
    assert [o.id for o in rank_offers(offers, "first")] == [1, 2, 3]


def test_pick_index_is_into_cheapest_first_ranking():
    # index N = N-th cheapest, not N-th in raw API order
    offers = [make_offer(1, 2.0), make_offer(2, 0.9)]
    assert pick_offer(offers, 2).id == 1  # 2nd cheapest is the 2.0 one


def test_pick_index_out_of_range_raises():
    import pytest
    offers = [make_offer(1, 2.0)]
    with pytest.raises(ValueError):
        pick_offer(offers, 5)


def test_pick_empty():
    assert pick_offer([], "cheapest") is None


def test_build_search_body_disk_and_rentable():
    body = build_search_body(GPUFilter(names=["H200 NVL"], count=2), min_disk_gb=120)
    assert body["gpu_name"] == {"eq": "H200 NVL"}
    assert body["disk_space"] == {"gte": 120.0}
    assert body["rentable"] == {"eq": True}
