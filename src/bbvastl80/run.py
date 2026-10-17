"""M6: the pipeline — ``rent → provision → connect → bench → publish → reap``.

Every phase is a small function that reads and writes the run's
:class:`~bbvastl80.state.RunState`, so a run can stop after any phase
(``--up-to``), skip the front (``--instance``), and be torn down days later
(``destroy --run``). The *offload core* (rent → provision → connect) is what
``serve`` runs; ``run`` adds the benchmark consumer (bench → publish) and the
teardown.

Teardown policy: with ``lifecycle.teardown: auto`` (the default) the box is
destroyed **after a successful publish**, and also after a failure (a dead box
still bills) unless ``--keep`` is passed — in which case the reconnect and
destroy commands are printed instead.
"""
from __future__ import annotations

import time
from pathlib import Path

from bbvastl80 import log
from bbvastl80.bench import BenchError, run_bench, validate_results
from bbvastl80.bootstrap import provision, render_bootstrap
from bbvastl80.config import Config, ConfigError
from bbvastl80.connect import ConnectError, close_tunnel, connect
from bbvastl80.publish import PublishError, run_publish
from bbvastl80.spec import RunSpec
from bbvastl80.state import RunState, StateStore
from bbvastl80.vast.client import VastClient, VastError
from bbvastl80.vast.instance import HostError, ProvisionError, rent as rent_box, wait_for_running
from bbvastl80.vast.models import Instance, Offer
from bbvastl80.vast.offers import build_search_body, rank_offers
from bbvastl80.vast.sshkeys import (
    attach_key, ensure_account_key, resolve_identity, resolve_public_key,
)

PHASES = ("rent", "provision", "connect", "bench", "publish", "reap")


class PipelineError(Exception):
    """A phase failed; the message says which and why."""


# --- rent --------------------------------------------------------------------
def rent_phase(cfg: Config, spec: RunSpec, c: VastClient, offer_id: int | None,
               *, name: str | None = None) -> tuple[StateStore, RunState, Instance]:
    """Cost guard → SSH key → search → rent (next offer on a stale one) → wait
    for running → attach key → save state."""
    if spec.budget.max_total_usd is not None:
        user = c.get_user()
        if user.credit is not None and user.credit < spec.budget.max_total_usd:
            raise PipelineError(
                f"balance ${user.credit:.2f} < budget ${spec.budget.max_total_usd:.2f}; "
                "top up at https://cloud.vast.ai or lower budget.max_total_usd")

    # Vast injects the account's keys at creation time: register first.
    try:
        ident = resolve_identity(spec.ssh.identity)
        pub = resolve_public_key(spec.ssh.public_key, ident)
    except ConfigError as e:
        raise PipelineError(str(e)) from e
    try:
        ensure_account_key(c, pub)
    except VastError as e:
        raise PipelineError(f"could not register the SSH key with Vast: {e}") from e
    bootstrap = render_bootstrap(spec)
    if bootstrap:
        log.detail(f"onstart bootstrap: {len(bootstrap)} bytes")

    if offer_id is not None:
        candidates = [Offer(id=offer_id, gpu_name=f"offer-{offer_id}")]
    else:
        try:
            offers = c.search_offers(build_search_body(spec.gpu, min_disk_gb=spec.instance.disk_gb))
        except VastError as e:
            raise PipelineError(f"search failed: {e}") from e
        if not offers:
            raise PipelineError("no offers matched — relax the gpu filter")
        candidates = rank_offers(offers, spec.gpu.pick)
        if not candidates:
            raise PipelineError("no offer selected")

    store = StateStore(cfg.runs_dir)
    run_id = store.new_run_id(name or spec.name)
    state = RunState(run_id=run_id, status="renting", spec_name=spec.name)
    chosen, inst, last_err = None, None, None
    for cand in candidates[:8]:
        price = f" ${cand.dph_total:.3f}/hr" if cand.dph_total is not None else ""
        log.info(f"renting offer {cand.id} ({cand.label}{price})")
        try:
            instance_id = rent_box(c, cand, spec.serving, spec.instance, bootstrap)
        except VastError as e:
            last_err = e
            log.detail(f"  offer {cand.id} not rentable ({' '.join(str(e).split())[:90]}) — next")
            continue
        # save *before* waiting so a crash still knows the box
        state.status, state.offer_id, state.offer_label = "rented", cand.id, cand.label
        state.instance_id, state.dph, state.started_billing_at = instance_id, cand.dph_total, time.time()
        store.save(state)
        log.info(f"instance {instance_id} created (run {run_id}) — waiting up to "
                 f"{spec.budget.max_provision_min:.0f} min for it to run…")
        try:
            inst = wait_for_running(c, instance_id, timeout_min=spec.budget.max_provision_min)
            chosen = cand
            break
        except HostError as e:
            # The host is broken, not us: destroy the dead box, try the next offer.
            last_err = e
            log.warn(f"{e}\n  destroying {instance_id} and trying the next offer")
            try:
                c.destroy_instance(instance_id)
            except VastError as de:
                log.error(f"could not destroy {instance_id}: {de} — run "
                          f"`bbvastl80 destroy --instance {instance_id} --yes`")
            state.status, state.instance_id = "host_error", None
            store.save(state)
        except ProvisionError as e:
            state.status = "failed"
            store.save(state)
            raise PipelineError(f"{e}\n(the instance may still be starting; re-check with "
                                f"`bbvastl80 status --instance {instance_id}`)") from e
    if chosen is None or inst is None or state.instance_id is None:
        state.status = "failed"
        store.save(state)
        raise PipelineError(f"no offer could be rented and started "
                            f"({min(len(candidates), 8)} tried): {last_err}")
    instance_id = state.instance_id
    attach_key(c, instance_id, pub)
    record_instance(state, inst, spec)
    store.save(state)
    return store, state, inst


def record_instance(state: RunState, inst: Instance, spec: RunSpec) -> None:
    state.instance_status = inst.status
    state.public_ip = inst.host_ip
    ports = {str(p): inst.external_port(p) for p in (spec.serving.port, 22)}
    state.ports = {**state.ports, **{k: v for k, v in ports.items() if v}}
    target = inst.ssh_target()
    if target:
        state.ssh_host, state.ssh_port = target


def state_for_instance(store: StateStore, instance_id: int) -> RunState | None:
    for rid in store.list_run_ids():
        st = store.load(rid)
        if st.instance_id == instance_id:
            return st
    return None


def latest_state(store: StateStore) -> RunState | None:
    for rid in store.list_run_ids():
        st = store.load(rid)
        if st.instance_id:
            return st
    return None


def running_instance(c: VastClient, instance_id: int, spec: RunSpec) -> Instance:
    inst = c.get_instance(instance_id)
    if not inst.is_running:
        log.info(f"instance {instance_id} is '{inst.status}' — waiting for it to run…")
        inst = wait_for_running(c, instance_id, timeout_min=spec.budget.max_provision_min)
    return inst


# --- cost ----------------------------------------------------------------------
def cost_estimate(state: RunState) -> str:
    if not state.started_billing_at:
        return "?"
    hours = (time.time() - state.started_billing_at) / 3600
    if state.dph is None:
        return f"{hours * 60:.0f} min at an unknown $/hr"
    return f"~${state.dph * hours:.2f} ({hours * 60:.0f} min × ${state.dph:.3f}/hr)"


# --- teardown -------------------------------------------------------------------
def teardown(c: VastClient, store: StateStore, state: RunState, *, reason: str) -> None:
    if state.tunnel_pid:
        close_tunnel(state.tunnel_pid)
        state.tunnel_pid = None
    if state.instance_id and state.status != "torn_down":
        log.info(f"destroying instance {state.instance_id} ({reason})")
        try:
            c.destroy_instance(state.instance_id)
            state.status = "torn_down"
            log.success(f"instance {state.instance_id} destroyed · cost {cost_estimate(state)}")
        except VastError as e:
            log.error(f"could not destroy {state.instance_id}: {e} — run "
                      f"`bbvastl80 destroy --instance {state.instance_id} --yes`")
    store.save(state)


def keep_notice(state: RunState) -> None:
    log.warn(f"instance {state.instance_id} is still RUNNING and billing "
             f"({cost_estimate(state)} so far)")
    if state.ssh_host:
        log.info(f"  shell:    bbvastl80 ssh --instance {state.instance_id}")
    if state.endpoint:
        log.info(f"  endpoint: {state.endpoint}")
    log.info(f"  finish:   bbvastl80 destroy --run {state.run_id} --yes")


# --- the pipeline -----------------------------------------------------------------
def dry_run(spec: RunSpec, c: VastClient) -> None:
    """Search + plan, no billing."""
    offers = c.search_offers(build_search_body(spec.gpu, min_disk_gb=spec.instance.disk_gb))
    ranked = rank_offers(offers, spec.gpu.pick)
    if not ranked:
        raise PipelineError("no offers matched — relax the gpu filter")
    rows = [[i, o.label, f"${o.dph_total:.3f}" if o.dph_total is not None else "?",
             f"{o.total_ram_gb}GB" if o.total_ram_gb else "?", o.geolocation or "?",
             f"{o.reliability:.3f}" if o.reliability is not None else "?", o.id]
            for i, o in enumerate(ranked[:10], 1)]
    log.table("Offers (ranked)", ["#", "GPU", "$/hr", "RAM", "geo", "rel", "id"], rows)
    chosen = ranked[0]
    bootstrap = render_bootstrap(spec)
    log.info(f"would rent offer {chosen.id} ({chosen.label}) with image "
             f"'{__import__('bbvastl80.providers', fromlist=['resolve_image']).resolve_image(spec.serving)}', "
             f"disk {spec.instance.disk_gb}GB, onstart {'bootstrap script' if bootstrap else 'none'}")
    if bootstrap:
        log.detail(bootstrap)
    log.info(f"then: provision → connect ({spec.connect.mode}) → bench "
             f"({'quick' if spec.benchmark.quick else f'{spec.benchmark.passes} passes'}) → "
             f"publish ({spec.publish.template}{', dry-run' if spec.publish.dry_run else ''}) → "
             f"{'keep' if spec.lifecycle.keep_running else 'destroy'}")


def run_pipeline(cfg: Config, spec: RunSpec, c: VastClient, *, up_to: str | None = None,
                 keep: bool = False, offer: int | None = None, instance: int | None = None,
                 publish_dry_run: bool | None = None) -> RunState:
    if up_to is not None and up_to not in PHASES:
        raise PipelineError(f"--up-to must be one of {', '.join(PHASES)}")
    keep = keep or spec.lifecycle.keep_running or spec.lifecycle.teardown == "manual"
    store = StateStore(cfg.runs_dir)

    def stop_after(phase: str) -> bool:
        return up_to == phase

    # --- rent (or adopt) ---
    if instance is not None:
        state = state_for_instance(store, instance)
        if state is None:
            state = RunState(run_id=store.new_run_id(spec.name), status="rented",
                             spec_name=spec.name, instance_id=instance,
                             started_billing_at=time.time())
        try:
            inst = running_instance(c, instance, spec)
        except (VastError, ProvisionError) as e:
            raise PipelineError(f"instance {instance}: {e}") from e
        record_instance(state, inst, spec)
        store.save(state)
    else:
        store, state, inst = rent_phase(cfg, spec, c, offer)
    log.success(f"instance {state.instance_id} RUNNING · run {state.run_id} · "
                f"ssh {state.ssh_host}:{state.ssh_port}")
    if stop_after("rent"):
        keep_notice(state)
        return state

    try:
        # --- provision ---
        box = provision(c, inst, spec)
        state.status = "provisioned"
        store.save(state)
        if stop_after("provision"):
            keep_notice(state)
            return state

        # --- connect ---
        conn = connect(box, spec)
        state.endpoint, state.connect_mode, state.tunnel_pid = conn.endpoint, conn.mode, conn.tunnel_pid
        state.status = "connected"
        store.save(state)
        if stop_after("connect"):
            keep_notice(state)
            return state

        # --- bench ---
        out = Path(cfg.runs_dir) / state.run_id / "results.json"
        run_bench(cfg.betterbench_bin, spec, conn.endpoint, out)
        summary = validate_results(out)
        state.results_path = str(out)
        state.status = "benchmarked"
        store.save(state)
        log.success(f"results: {out} · {summary}")
        if stop_after("bench"):
            keep_notice(state)
            return state

        # --- publish ---
        url = run_publish(cfg.l80_bin, spec, out, dry_run=publish_dry_run)
        if url:
            state.publish_url = url
            state.status = "published"
            log.success(f"published: {url}")
        store.save(state)
        if stop_after("publish"):
            keep_notice(state)
            return state
    except (ProvisionError, ConnectError, BenchError, PublishError, VastError, ConfigError) as e:
        state.status = "failed"
        store.save(state)
        log.error(f"pipeline failed: {e}")
        if keep:
            keep_notice(state)
        else:
            teardown(c, store, state, reason="failure with lifecycle.teardown=auto; "
                                                "pass --keep to inspect the box instead")
        raise PipelineError(str(e)) from e

    # --- reap ---
    if keep:
        if state.tunnel_pid:
            log.info(f"  tunnel:   ssh pid {state.tunnel_pid} (closed by destroy --run)")
        keep_notice(state)
    else:
        teardown(c, store, state, reason="publish done, lifecycle.teardown=auto")
    return state
