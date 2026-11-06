"""The bbvastl80 command-line interface.

Read-only (``doctor``, ``search``, ``list``, ``status``), lifecycle (``rent``,
``destroy``, ``reap``), provisioning (``serve``, ``provision``, ``ssh``,
``logs``) and the pipeline (``connect``, ``bench``, ``publish``, ``run``).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.table import Table

from bbvastl80 import __version__, log
from bbvastl80.bench import BenchError, run_bench, validate_results
from bbvastl80.bootstrap import build_box, provision as provision_box
from bbvastl80.connect import ConnectError, connect as connect_box, verify_v1
from bbvastl80.config import CRED_FILE, Config, ConfigError
from bbvastl80.providers.base import BOOTSTRAP_LOG, SERVER_LOG, STATUS_FILE
from bbvastl80.publish import PublishError, run_publish
from bbvastl80.run import (
    PHASES, PipelineError, dry_run as pipeline_dry_run, latest_state, rent_phase, run_pipeline,
)
from bbvastl80.spec import RunSpec, SpecError, load_spec
from bbvastl80.state import StateStore
from bbvastl80.vast import VastAuthError, VastClient, VastError
from bbvastl80.vast.instance import wait_for_running
from bbvastl80.vast.models import Instance, Offer
from bbvastl80.vast.offers import build_search_body, pick_offer
from bbvastl80.vast.sshkeys import key_body, resolve_identity, resolve_public_key

app = typer.Typer(add_completion=False, no_args_is_help=True)

SPEC_OPT = Annotated[Optional[str], typer.Option("--spec", "-s", help="Path to a run spec YAML")]
YES_OPT = Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt")]
OFFER_OPT = Annotated[Optional[int], typer.Option("--offer", help="Rent this offer id (skip search)")]
INSTANCE_OPT = Annotated[Optional[int], typer.Option("--instance", help="Target a specific instance id")]
RUN_OPT = Annotated[Optional[str], typer.Option("--run", help="Target a run by id")]
VERBOSE_OPT = Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")]


MIN_BETTERBENCH = "0.5.0"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"bbvastl80 {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True,
                           help="Print the version and exit")
    ] = False,
) -> None:
    """Rent a Vast.ai GPU, run your stack on it, benchmark with BetterBench, publish with L80."""


# --- shared helpers ----------------------------------------------------------
def _version_at_least(text: str, minimum: str) -> bool:
    """True when the first x.y.z in ``text`` is >= ``minimum``."""
    import re

    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not m:
        return False
    return tuple(map(int, m.groups())) >= tuple(map(int, minimum.split(".")))

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

    # SSH key (the only way commands reach a box)
    try:
        ident = resolve_identity(None)
        pub = resolve_public_key(None, ident)
        log.success(f"ssh key: {ident} ({pub.split()[0]})")
        if cfg.api_key:
            try:
                with _client(cfg) as c:
                    keys = c.list_ssh_keys()
                have = any(key_body(str(k.get("public_key") or k.get("ssh_key") or ""))
                           == key_body(pub) for k in keys)
                (log.success if have else log.warn)(
                    "ssh public key is registered on the Vast account" if have else
                    "ssh public key not yet on the Vast account — `rent` registers it "
                    "automatically (POST /ssh/)")
            except VastError as e:
                log.warn(f"could not check account SSH keys: {e}")
    except ConfigError as e:
        log.warn(str(e))

    # external tools
    bins = cfg.check_binaries()
    for name, path in bins.items():
        if path:
            log.success(f"{name}: {path}")
        else:
            log.warn(f"{name}: not found on PATH")

    # BetterBench version probe: results schema 2 (0.5.0 and later) is what L80 publishes
    bb = bins.get("betterbench")
    if bb:
        try:
            out = subprocess.run([bb, "--version"], capture_output=True, text=True, timeout=30)
            ver = (out.stdout or out.stderr).strip().splitlines()
            ver = ver[0] if ver else "unknown"
            ok = _version_at_least(ver, MIN_BETTERBENCH)
            (log.success if ok else log.warn)(
                f"betterbench version: {ver}" + ("" if ok else f" (need >= {MIN_BETTERBENCH})")
            )
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
            offers = c.search_offers(build_search_body(s.gpu, min_disk_gb=s.instance.disk_gb))
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
def list_cmd(verbose: VERBOSE_OPT = False,
             as_json: Annotated[bool, typer.Option("--json", help="Print a JSON array (id, status, label, dph).")] = False
             ) -> None:
    """List all instances on the account."""
    cfg = _cfg()
    with _client(cfg) as c:
        try:
            instances = c.list_instances()
        except VastAuthError as e:
            _fail(str(e))
        except VastError as e:
            _fail(f"list failed: {e}")
    if as_json:
        # L80 checks its local run records against this, so a destroyed box is never reported as billing.
        print(json.dumps([{"id": i.id, "status": i.status, "label": i.label,
                           "dph": getattr(i, "dph_total", None)} for i in instances]))
        return
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
    target = inst.ssh_target()
    if target:
        t.add_row("ssh", f"root@{target[0]} -p {target[1]}")
    if inst.env:
        vast_ports = {k: v for k, v in inst.env.items() if k.startswith("VAST_TCP_PORT_")}
        if vast_ports:
            t.add_row("vast ports", "; ".join(f"{k}={v}" for k, v in vast_ports.items()))
    log.console().print(t)


# --- M2: rent / destroy / reap ----------------------------------------------
def _rent_phase(cfg, s, c, offer):
    """Thin CLI wrapper over :func:`bbvastl80.run.rent_phase` (cost guard → SSH
    key → search → rent → wait → attach key → save state). Returns
    ``(store, run_id, state, chosen, instance)`` for ``rent`` and ``serve``."""
    try:
        store, state, inst = rent_phase(cfg, s, c, offer)
    except VastAuthError as e:
        _fail(str(e))
    except PipelineError as e:
        _fail(str(e))
    chosen = Offer(id=state.offer_id, gpu_name=state.offer_label, dph_total=state.dph)
    return store, state.run_id, state, chosen, inst


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


# --- M3–M6: serve / provision / connect / bench / publish / run -------------


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
    cmd = box.ssh_command()
    if cmd:
        log.info("shell: " + " ".join(cmd))


def _box_for(cfg, c, s: RunSpec, instance: int | None):
    """Resolve an instance (explicit or the latest run's) into a Box."""
    run_id, iid, state = _resolve_target(cfg, instance)
    try:
        inst = c.get_instance(iid)
    except VastError as e:
        _fail(f"status failed: {e}")
    return build_box(c, inst, s), inst, state


def _spec_or_default(spec_path: Optional[str]) -> RunSpec:
    """`ssh`/`logs` don't need a spec; fall back to defaults when none given."""
    if spec_path:
        return load_spec(spec_path)
    return RunSpec(model={"id": "_", "hf_repo": "_"})


@app.command()
def ssh(spec: SPEC_OPT = None, instance: INSTANCE_OPT = None,
        command: Annotated[Optional[str], typer.Argument(help="run this instead of a shell")] = None,
        verbose: VERBOSE_OPT = False) -> None:
    """Open a shell on the box (or run one command). Prints the ssh line used."""
    cfg = _cfg()
    try:
        s = _spec_or_default(spec)
    except SpecError as e:
        _fail(str(e))
    with _client(cfg) as c:
        box, inst, _state = _box_for(cfg, c, s, instance)
    argv = box.ssh_command()
    if not argv:
        _fail(f"instance {inst.id} has no SSH host/port yet (status '{inst.status}')")
    argv = argv + ([command] if command else [])
    log.info("$ " + " ".join(argv))
    raise typer.Exit(subprocess.call(argv))


@app.command()
def logs(spec: SPEC_OPT = None, instance: INSTANCE_OPT = None,
         lines: Annotated[int, typer.Option("--lines", "-n")] = 60,
         verbose: VERBOSE_OPT = False) -> None:
    """Tail the bootstrap + server logs on the box (over SSH)."""
    cfg = _cfg()
    try:
        s = _spec_or_default(spec)
    except SpecError as e:
        _fail(str(e))
    with _client(cfg) as c:
        box, inst, _state = _box_for(cfg, c, s, instance)
    if not box.ssh_host:
        _fail(f"instance {inst.id} has no SSH host/port yet (status '{inst.status}')")
    r = box.run(
        f"echo \"status: $(cat {STATUS_FILE} 2>/dev/null || echo none)\"; "
        f"echo '--- {BOOTSTRAP_LOG}'; tail -n {lines} {BOOTSTRAP_LOG} 2>/dev/null; "
        f"echo '--- {SERVER_LOG}'; tail -n {lines} {SERVER_LOG} 2>/dev/null",
        timeout=30,
    )
    if not r.ok and not r.stdout:
        _fail(f"ssh failed (rc={r.exit_code}): {r.stderr.strip()[:200]}")
    log.console().print(r.stdout.rstrip())


@app.command()
def connect(spec: SPEC_OPT = None, instance: INSTANCE_OPT = None,
            verbose: VERBOSE_OPT = False) -> None:
    """Resolve the endpoint (direct port or SSH tunnel), verify /v1, print it.

    A tunnel is left running in the background (its pid is saved in the run
    state; `destroy --run` closes it)."""
    cfg = _cfg()
    try:
        s = _spec(spec)
    except (SpecError, ConfigError) as e:
        _fail(str(e))
    with _client(cfg) as c:
        box, inst, state = _box_for(cfg, c, s, instance)
        try:
            conn = connect_box(box, s)
        except ConnectError as e:
            _fail(str(e))
    if state is not None:
        state.endpoint, state.connect_mode, state.tunnel_pid = conn.endpoint, conn.mode, conn.tunnel_pid
        state.status = "connected"
        _state_store(cfg).save(state)
    log.success(f"endpoint: {conn.endpoint}  ({conn.mode}; model '{s.model.id}')")
    log.info(f"try it:  curl {conn.endpoint}/models")


@app.command()
def bench(spec: SPEC_OPT = None,
          endpoint: Annotated[Optional[str], typer.Option(help="OpenAI /v1 base; default: the latest run's")] = None,
          out: Annotated[Optional[str], typer.Option(help="results.json path")] = None,
          verbose: VERBOSE_OPT = False) -> None:
    """Run the pinned BetterBench against an endpoint; write + validate results.json."""
    cfg = _cfg()
    try:
        s = _spec(spec)
    except (SpecError, ConfigError) as e:
        _fail(str(e))
    store = _state_store(cfg)
    state = None
    if endpoint is None:
        state = latest_state(store)
        if state is None or not state.endpoint:
            _fail("no endpoint: pass --endpoint URL or run `bbvastl80 connect` first")
        endpoint = state.endpoint
    target = Path(out) if out else (
        Path(cfg.runs_dir) / state.run_id / "results.json" if state else Path("results.json"))
    try:
        verify_v1(endpoint, s.serving.resolved_model_id(s.model), timeout_s=15)
        run_bench(cfg.betterbench_bin, s, endpoint, target)
        summary = validate_results(target)
    except (ConnectError, BenchError) as e:
        _fail(str(e))
    if state is not None:
        state.results_path, state.status = str(target), "benchmarked"
        store.save(state)
    log.success(f"results: {target} · {summary}")


@app.command()
def publish(results: Annotated[str, typer.Argument(help="a BetterBench results.json")],
            spec: SPEC_OPT = None,
            dry_run: Annotated[bool, typer.Option("--dry-run", help="validate with L80, don't publish")] = False,
            verbose: VERBOSE_OPT = False) -> None:
    """Publish a results.json via the L80 CLI (public!). Validates it first."""
    cfg = _cfg()
    try:
        s = _spec_or_default(spec)
    except SpecError as e:
        _fail(str(e))
    path = Path(results)
    try:
        validate_results(path)
        url = run_publish(cfg.l80_bin, s, path, dry_run=dry_run or None)
    except (BenchError, PublishError) as e:
        _fail(str(e))
    if url:
        log.success(f"published: {url}")
        store = _state_store(cfg)
        for rid in store.list_run_ids():
            st = store.load(rid)
            if st.results_path and Path(st.results_path).resolve() == path.resolve():
                st.publish_url, st.status = url, "published"
                store.save(st)
                break


@app.command()
def run(spec: SPEC_OPT = None,
        up_to: Annotated[Optional[str], typer.Option("--up-to", help="stop after: " + ", ".join(PHASES))] = None,
        dry_run: Annotated[bool, typer.Option("--dry-run", help="search + plan only, no billing")] = False,
        keep: Annotated[bool, typer.Option("--keep", help="leave the box running at the end")] = False,
        publish_dry_run: Annotated[bool, typer.Option("--publish-dry-run", help="L80 dry run instead of publishing")] = False,
        offer: OFFER_OPT = None,
        instance: INSTANCE_OPT = None,
        verbose: VERBOSE_OPT = False) -> None:
    """The full pipeline: rent → provision → connect → the spec's steps → destroy.

    Exits 1 when any step failed (its error is in state.json and on the
    ``bbvastl80:step`` line); the box is torn down either way."""
    cfg = _cfg()
    try:
        s = _spec(spec)
    except (SpecError, ConfigError) as e:
        _fail(str(e))
    with _client(cfg) as c:
        try:
            if dry_run:
                pipeline_dry_run(s, c)
                return
            state = run_pipeline(cfg, s, c, up_to=up_to, keep=keep, offer=offer,
                                 instance=instance, publish_dry_run=True if publish_dry_run else None)
        except VastAuthError as e:
            _fail(str(e))
        except PipelineError as e:
            _fail(str(e))
    log.success(f"run {state.run_id}: {state.status}")
    for st in state.steps:
        line = f"  {st.label}: {st.status}" + (f" · {st.url}" if st.url else "") + \
            (f" · {st.results}" if st.results else "")
        (log.success if st.status == "done" else log.warn)(line)
    if any(st.status == "failed" for st in state.steps):
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
