"""The bbvastl80 command-line interface.

M0–M2 (this stage): ``init``, ``doctor``, ``search``, ``list``, ``status``,
``rent``, ``destroy``, ``reap``. The pipeline commands (``serve``,
``provision``, ``connect``, ``bench``, ``publish``, ``run``) are present as
stubs so the surface is stable; they light up in M3–M6.
"""
from __future__ import annotations

import subprocess
from typing import Annotated, Optional

import typer
from rich.table import Table

from bbvastl80 import __version__, log
from bbvastl80.bootstrap import provision as provision_box
from bbvastl80.config import CRED_FILE, Config, ConfigError
from bbvastl80.spec import RunSpec, SpecError, load_spec
from bbvastl80.state import RunState, StateStore
from bbvastl80.vast import VastAuthError, VastClient, VastError
from bbvastl80.vast.instance import ProvisionError, rent as rent_box, wait_for_running
from bbvastl80.vast.models import Instance, Offer
from bbvastl80.vast.offers import build_search_body, pick_offer, rank_offers

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

SPEC_OPT = Annotated[Optional[str], typer.Option("--spec", "-s", help="Path to a run spec YAML")]
YES_OPT = Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt")]
OFFER_OPT = Annotated[Optional[int], typer.Option("--offer", help="Rent this offer id (skip search)")]
INSTANCE_OPT = Annotated[Optional[int], typer.Option("--instance", help="Target a specific instance id")]
RUN_OPT = Annotated[Optional[str], typer.Option("--run", help="Target a run by id")]
VERBOSE_OPT = Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")]


# --- shared helpers ----------------------------------------------------------
def _cfg() -> Config:
    return Config.load()


def _client(cfg: Config) -> VastClient:
    return VastClient(cfg.require_api_key(), cfg.api_base)


def _spec(spec_path: Optional[str]) -> RunSpec:
    if not spec_path:
        raise ConfigError("this command needs a spec: pass --spec <file.yaml>")
    return load_spec(spec_path)


def _state_store(cfg: Config) -> StateStore:
    return StateStore(cfg.runs_dir)


def _fail(msg: str) -> None:
    log.error(msg)
    raise typer.Exit(1)


def _offer_from_id(cfg: Config, offer_id: int) -> Offer:
    # Minimal offer when the user supplies an id directly (no search).
    return Offer(id=offer_id, gpu_name=f"offer-{offer_id}")


def _resolve_target(cfg: Config, instance: int | None):
    """Find the (run_id, instance_id, state) to provision.

    An explicit ``--instance`` wins; otherwise the most recent run that rented a
    box is used. Returns ``None``-able pieces so the caller can give a precise
    error.
    """
    store = _state_store(cfg)
    if instance is not None:
        state = None
        for rid in store.list_run_ids():
            st = store.load(rid)
            if st.instance_id == instance:
                state = st
                break
        return (state.run_id if state else None), instance, state
    for rid in store.list_run_ids():
        st = store.load(rid)
        if st.instance_id:
            return rid, st.instance_id, st
    _fail("no instance to provision — rent one first (`bbvastl80 rent -s …`) "
          "or pass --instance <id>")


def _print_offers(offers: list[Offer], chosen: Offer | None) -> None:
    rows = []
    for i, o in enumerate(offers, 1):
        mark = "  ←" if chosen is o else ""
        rows.append([
            i,
            o.label,
            f"${o.dph_total:.3f}" if o.dph_total is not None else "?",
            str(o.total_ram_gb) + "GB" if o.total_ram_gb else "?",
            o.datacenter or o.geolocation or "?",
            f"{o.reliability:.2f}" if o.reliability is not None else "?",
            "✓" if o.verified else "",
            str(o.id),
            mark,
        ])
    log.table(
        "Offers (cheapest first)",
        ["#", "GPU", "$/hr", "RAM", "DC/geo", "rel", "vrf", "id", ""],
        rows,
    )


# --- M0: init / doctor -------------------------------------------------------
@app.command()
def init(verbose: VERBOSE_OPT = False) -> None:
    """Create a starter .env and run a connectivity check."""
    cfg = _cfg()
    from pathlib import Path

    env = Path(".env")
    if not env.exists():
        env.write_text(
            "# Vast.ai API key — create one at https://cloud.vast.ai/manage-keys/\n"
            "VAST_API_KEY=\n\n"
            "# Optional overrides\n"
            "# VAST_API_BASE=https://console.vast.ai/api/v0\n"
            f"# L80_BETTERBENCH={cfg.betterbench_bin}\n"
        )
        log.success(f"wrote {env} — add your VAST_API_KEY, then re-run `bbvastl80 doctor`")
    else:
        log.info(f".env already exists at {env}")

    doctor(verbose=verbose)


@app.command()
def doctor(verbose: VERBOSE_OPT = False) -> None:
    """Check credentials and the external tools (Vast key, l80, betterbench, ssh)."""
    cfg = _cfg()
    log.info(f"bbvastl80 {__version__}")

    # Vast key
    if cfg.api_key:
        log.success(f"VAST_API_KEY present ({cfg.api_key[:4]}…{cfg.api_key[-4:]})")
        try:
            with _client(cfg) as c:
                user = c.get_user()
            credit = f"${user.credit:.2f}" if user.credit is not None else "?"
            log.success(f"Vast account OK: {user.email or user.id} · balance {credit}")
        except VastAuthError as e:
            log.error(f"Vast auth failed: {e}")
        except VastError as e:
            log.warn(f"Vast reachable but: {e}")
    else:
        log.warn(
            "no VAST_API_KEY (env or "
            f"{CRED_FILE}). Create one at https://cloud.vast.ai/manage-keys/"
        )

    # external tools
    bins = cfg.check_binaries()
    for name, path in bins.items():
        if path:
            log.success(f"{name}: {path}")
        else:
            log.warn(f"{name}: not found on PATH")

    # BetterBench version probe (the pin must be 0.4.0 for comparability)
    bb = bins.get("betterbench")
    if bb:
        try:
            out = subprocess.run([bb, "--version"], capture_output=True, text=True, timeout=30)
            ver = (out.stdout or out.stderr).strip().splitlines()
            ver = ver[0] if ver else "unknown"
            (log.success if "0.4.0" in ver else log.warn)(f"betterbench version: {ver}")
        except (OSError, subprocess.SubprocessError) as e:
            log.warn(f"betterbench: could not probe version ({e})")


# --- M1: search / list / status ---------------------------------------------
@app.command()
def search(spec: SPEC_OPT = None, verbose: VERBOSE_OPT = False) -> None:
    """Show candidate offers for the spec's GPU filter (no billing)."""
    cfg = _cfg()
    try:
        s = _spec(spec)
    except (SpecError, ConfigError) as e:
        _fail(str(e))
    with _client(cfg) as c:
        try:
            offers = c.search_offers(build_search_body(s.gpu))
        except VastAuthError as e:
            _fail(str(e))
        except VastError as e:
            _fail(f"search failed: {e}")
    if not offers:
        _fail("no offers matched the filter — relax gpu.names / gpu.max_dph / gpu.count")
    chosen = pick_offer(offers, s.gpu.pick)
    _print_offers(offers[:20], chosen)
    if chosen:
        log.success(f"pick={s.gpu.pick} → offer {chosen.id} ({chosen.label})")


@app.command(name="list")
def list_cmd(verbose: VERBOSE_OPT = False) -> None:
    """List all instances on the account."""
    cfg = _cfg()
    with _client(cfg) as c:
        try:
            instances = c.list_instances()
        except VastAuthError as e:
            _fail(str(e))
        except VastError as e:
            _fail(f"list failed: {e}")
    if not instances:
        log.info("no instances")
        return
    rows = [
        [i.id, i.status or "?", i.host_ip or "?",
         ",".join(f"{k}->{v}" for k, v in (i.ports if isinstance(i.ports, dict) else {}).items())
         if isinstance(i.ports, dict) else "?",
         i.label or ""]
        for i in instances
    ]
    log.table("Instances", ["id", "status", "ip", "ports", "label"], rows)


@app.command()
def status(instance: INSTANCE_OPT = None, run: RUN_OPT = None,
           verbose: VERBOSE_OPT = False) -> None:
    """Show one instance (or the one from a run), with its port mappings."""
    cfg = _cfg()
    target = instance
    if target is None and run:
        try:
            st = _state_store(cfg).load(run)
            target = st.instance_id
        except FileNotFoundError as e:
            _fail(str(e))
    if target is None:
        _fail("pass --instance <id> or --run <run-id>")
    with _client(cfg) as c:
        try:
            inst = c.get_instance(target)
        except VastAuthError as e:
            _fail(str(e))
        except VastError as e:
            _fail(f"status failed: {e}")
    _print_instance(inst)


def _print_instance(inst: Instance) -> None:
    t = Table(title=f"instance {inst.id}")
    t.add_column("field")
    t.add_column("value")
    t.add_row("status", inst.status or "?")
    t.add_row("type", inst.type or "?")
    t.add_row("public ip", inst.host_ip or "?")
    t.add_row("label", inst.label or "-")
    for port in (8000, 8080, 22):
        ext = inst.external_port(port)
        if ext:
            t.add_row(f"port {port}", f"{inst.host_ip}:{ext}" if inst.host_ip else str(ext))
    if inst.env:
        vast_ports = {k: v for k, v in inst.env.items() if k.startswith("VAST_TCP_PORT_")}
        if vast_ports:
            t.add_row("vast ports", "; ".join(f"{k}={v}" for k, v in vast_ports.items()))
    log.console().print(t)


# --- M2: rent / destroy / reap ----------------------------------------------
def _rent_phase(cfg, s, c, offer):
    """Cost guard → search → rent (try the next offer on a stale one) → wait for
    running → save state. Returns ``(store, run_id, state, chosen, instance)``.
    Shared by ``rent`` and ``serve``."""
    if s.budget.max_total_usd is not None:
        try:
            user = c.get_user()
        except VastAuthError as e:
            _fail(str(e))
        if user.credit is not None and user.credit < s.budget.max_total_usd:
            _fail(f"balance ${user.credit:.2f} < budget ${s.budget.max_total_usd:.2f}; "
                  "top up at https://cloud.vast.ai or lower budget.max_total_usd")

    if offer is not None:
        candidates = [_offer_from_id(cfg, offer)]
    else:
        try:
            offers = c.search_offers(build_search_body(s.gpu))
        except VastError as e:
            _fail(f"search failed: {e}")
        if not offers:
            _fail("no offers matched — relax the gpu filter")
        candidates = rank_offers(offers, s.gpu.pick)
        if not candidates:
            _fail("no offer selected")

    chosen, instance_id, last_err = None, None, None
    for cand in candidates:
        log.info(f"renting offer {cand.id} ({cand.label})")
        try:
            instance_id = rent_box(c, cand, s.serving, s.instance)
            chosen = cand
            break
        except VastAuthError as e:
            _fail(str(e))  # an auth problem won't clear by trying another offer
        except VastError as e:
            last_err = e
            log.detail(f"  offer {cand.id} not rentable ({' '.join(str(e).split())[:90]}) — trying next")
    if instance_id is None:
        _fail(f"no offer could be rented ({len(candidates)} tried): {last_err}")

    log.info(f"instance {instance_id} created — waiting up to "
             f"{s.budget.max_provision_min:.0f} min for it to run…")
    try:
        inst = wait_for_running(c, instance_id, timeout_min=s.budget.max_provision_min)
    except ProvisionError as e:
        _fail(f"{e}\n(the instance may still be starting; re-check with "
              f"`bbvastl80 status --instance {instance_id}`)")

    store = _state_store(cfg)
    run_id = store.new_run_id(s.name)
    ports = {str(p): inst.external_port(p) for p in (s.serving.port, 22) if inst.external_port(p)}
    state = RunState(
        run_id=run_id, status="rented", spec_name=s.name,
        offer_id=chosen.id, offer_label=chosen.label,
        instance_id=instance_id, instance_status=inst.status,
        public_ip=inst.host_ip, ports={k: v for k, v in ports.items() if v},
        dph=chosen.dph_total,
    )
    store.save(state)
    return store, run_id, state, chosen, inst


@app.command()
def rent(spec: SPEC_OPT = None, offer: OFFER_OPT = None, verbose: VERBOSE_OPT = False) -> None:
    """Rent an offer (or search for one), wait until it's running, save state."""
    cfg = _cfg()
    try:
        s = _spec(spec)
    except (SpecError, ConfigError) as e:
        _fail(str(e))
    with _client(cfg) as c:
        store, run_id, state, _chosen, inst = _rent_phase(cfg, s, c, offer)
        log.success(f"instance {state.instance_id} is RUNNING · run id: {run_id}")
        _print_instance(inst)
        log.info(f"state: {store.path_for(run_id)}")


@app.command()
def destroy(instance: INSTANCE_OPT = None, run: RUN_OPT = None, yes: YES_OPT = False,
            verbose: VERBOSE_OPT = False) -> None:
    """Tear down an instance (or the one from a run). Irreversible."""
    cfg = _cfg()
    target = instance
    label = None
    if target is None and run:
        try:
            st = _state_store(cfg).load(run)
            target = st.instance_id
            label = "run " + run
        except FileNotFoundError as e:
            _fail(str(e))
    if target is None:
        _fail("pass --instance <id> or --run <run-id>")
    if not yes:
        typer.confirm(f"Destroy instance {target} ({label or 'no run'})? Data is deleted.",
                      abort=True)
    with _client(cfg) as c:
        try:
            c.destroy_instance(target)
        except VastAuthError as e:
            _fail(str(e))
        except VastError as e:
            _fail(f"destroy failed: {e}")
    if run:
        st = _state_store(cfg).load(run)
        st.status = "torn_down"
        _state_store(cfg).save(st)
    log.success(f"instance {target} destroyed")


@app.command()
def reap(yes: YES_OPT = False, verbose: VERBOSE_OPT = False) -> None:
    """Destroy every instance carrying our label (sweeps up forgotten boxes)."""
    cfg = _cfg()
    label = "bbvastl80"
    with _client(cfg) as c:
        try:
            instances = c.list_instances()
        except VastAuthError as e:
            _fail(str(e))
        except VastError as e:
            _fail(f"list failed: {e}")
    mine = [i for i in instances if i.label == label]
    if not mine:
        log.info(f"no instances labeled '{label}' — nothing to reap")
        return
    rows = [[i.id, i.status or "?", i.host_ip or "?"] for i in mine]
    log.table(f"Will destroy {len(mine)} instance(s) labeled '{label}'",
              ["id", "status", "ip"], rows)
    if not yes:
        typer.confirm("Destroy all of the above?", abort=True)
    with _client(cfg) as c:
        for i in mine:
            try:
                c.destroy_instance(i.id)
                log.success(f"destroyed {i.id}")
            except VastError as e:
                log.warn(f"could not destroy {i.id}: {e}")


# --- M3+ stubs (surface is stable; logic lands in later milestones) ---------
def _stub(what: str, milestone: str) -> None:
    log.warn(f"`{what}` is planned for {milestone} — not built in this stage yet.")
    raise typer.Exit(0)


@app.command()
def serve(spec: SPEC_OPT = None, offer: OFFER_OPT = None, instance: INSTANCE_OPT = None,
          verbose: VERBOSE_OPT = False) -> None:
    """Offload: rent (if needed) + provision the model, then report the /v1.

    This is the reusable "rent a GPU and get a live endpoint" primitive. Making
    the port reachable from this machine (direct vs SSH tunnel) is ``connect``.
    """
    cfg = _cfg()
    try:
        s = _spec(spec)
    except (SpecError, ConfigError) as e:
        _fail(str(e))
    with _client(cfg) as c:
        if instance is not None:
            run_id, iid, state = _resolve_target(cfg, instance)
        else:
            store, run_id, state, _chosen, _inst = _rent_phase(cfg, s, c, offer)
            iid = state.instance_id
        inst = c.get_instance(iid)
        if not inst.is_running:
            log.info(f"instance {iid} is '{inst.status}' — waiting for it to run…")
            inst = wait_for_running(c, iid, timeout_min=s.budget.max_provision_min)
        box = provision_box(c, inst, s)
        if state is not None:
            state.status = "provisioned"
            state.public_ip = box.host_ip
            state.ports = {**state.ports, **box.ports}
            state.endpoint = box.url()
            _state_store(cfg).save(state)
        _report_endpoint(box, s)


@app.command()
def provision(spec: SPEC_OPT = None, instance: INSTANCE_OPT = None,
              verbose: VERBOSE_OPT = False) -> None:
    """Bootstrap a running box (from a prior rent, or --instance) into serving
    the spec's model; confirms the /v1 endpoint answers inside the box."""
    cfg = _cfg()
    try:
        s = _spec(spec)
    except (SpecError, ConfigError) as e:
        _fail(str(e))
    with _client(cfg) as c:
        run_id, iid, state = _resolve_target(cfg, instance)
        inst = c.get_instance(iid)
        if not inst.is_running:
            log.info(f"instance {iid} is '{inst.status}' — waiting for it to run…")
            inst = wait_for_running(c, iid, timeout_min=s.budget.max_provision_min)
        box = provision_box(c, inst, s)
        if state is not None:
            state.status = "provisioned"
            state.instance_id = iid
            state.public_ip = box.host_ip
            state.ports = {**state.ports, **box.ports}
            state.endpoint = box.url()
            _state_store(cfg).save(state)
        _report_endpoint(box, s)


def _report_endpoint(box, s) -> None:
    if box.url():
        log.success(f"live endpoint: {box.url()}/v1  (model '{s.model.id}')")
    else:
        log.success(f"model '{s.model.id}' is serving at /v1 inside the box")
        log.info("external port not mapped yet — run `bbvastl80 connect` (M4) to "
                 "expose it here")


@app.command()
def connect(spec: SPEC_OPT = None, instance: INSTANCE_OPT = None,
            verbose: VERBOSE_OPT = False) -> None:
    """Resolve the endpoint (direct port or SSH tunnel) and verify /v1."""
    _stub("connect", "M4")


@app.command()
def bench(spec: SPEC_OPT = None, endpoint: Annotated[Optional[str], typer.Option()] = None,
          verbose: VERBOSE_OPT = False) -> None:
    """Run the pinned BetterBench against an endpoint; write results.json."""
    _stub("bench", "M5")


@app.command()
def publish(results: Annotated[str, typer.Argument()],
            dry_run: Annotated[bool, typer.Option()] = False,
            verbose: VERBOSE_OPT = False) -> None:
    """Publish a results.json via the L80 CLI."""
    _stub("publish", "M5")


@app.command()
def run(spec: SPEC_OPT = None,
        up_to: Annotated[Optional[str], typer.Option()] = None,
        dry_run: Annotated[bool, typer.Option()] = False,
        keep: Annotated[bool, typer.Option()] = False,
        offer: OFFER_OPT = None,
        instance: INSTANCE_OPT = None,
        verbose: VERBOSE_OPT = False) -> None:
    """The full pipeline: serve + bench + publish (the one-command run)."""
    _stub("run", "M6")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
