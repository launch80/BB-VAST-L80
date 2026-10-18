# BB-VAST-L80

Rent a GPU on **Vast.ai**, provision an inference stack on it, benchmark it with
the pinned **BetterBench**, and publish the result with the **L80** CLI — driven
by a single YAML spec.

The GPU and the model server are remote and disposable. BetterBench and L80 stay
**local** (pinned, so numbers are comparable; L80 keeps the endpoint/host on this
machine). The whole thing collapses to one contract: *the box serves
`model_id` at an OpenAI-compatible `/v1` endpoint.* Everything downstream is
engine-agnostic.

## Status

**Built (M0–M6):** project skeleton, config/doctor, the read-only Vast surface
(`search`, `list`, `status`), the rent/teardown lifecycle (`rent`, `destroy`,
`reap`) with per-run state for resumability, and provisioning (`serve`,
`provision`, `ssh`, `logs`): the provider renders one idempotent bootstrap
script that ships as the instance's `onstart` and is also pushed over SSH. `connect`
resolves a verified `/v1` here (direct port or SSH tunnel), `bench` runs the
pinned BetterBench, `publish` pushes through L80, and `run` chains it all and
destroys the box after the publish. See `specs/` for a real run spec; the first
live run published https://launch80.com/a/a23e4a7c-f503-4d78-985f-5fba6478257d.

The Vast client is exercised against the **live API** (search/list/status all
return clean, actionable errors with a valid key).

## Install

The easiest way is through the [L80 CLI](https://github.com/launch80/L80-CLI),
which pins a release of this project and runs it for you (with
[uv](https://docs.astral.sh/uv/) installed):

```bash
L80 vast doctor
L80 vast run --spec run.yaml --max-hours 2
```

Or run it directly:

```bash
uvx --from git+https://github.com/launch80/BB-VAST-L80@v0.1.0 bbvastl80 --help
```

For development:

```bash
git clone https://github.com/launch80/BB-VAST-L80 && cd BB-VAST-L80
uv venv && uv pip install -e ".[dev]"
alias bbvastl80="$PWD/.venv/bin/bbvastl80"   # or add .venv/bin to PATH
```

## Quick start

```bash
bbvastl80 init                 # writes a starter .env; then add your VAST_API_KEY
bbvastl80 doctor               # verify the key + l80/betterbench/ssh
cp run.example.yaml run.yaml   # edit the model + gpu filter
bbvastl80 search -s run.yaml   # see candidate offers (no billing)
bbvastl80 rent   -s run.yaml   # rent the cheapest match, wait for it to run
bbvastl80 status --run <id>    # inspect it
bbvastl80 destroy --run <id>   # tear it down
```

Create an API key at <https://cloud.vast.ai/manage-keys/> (scope:
`instance_read`, `instance_write`). Put it in `~/.config/bbvastl80/credentials`
(the bare key, or `api_key=…`) or export `VAST_API_KEY`; both work from any
directory, including under `L80 vast`. A `.env` with `VAST_API_KEY=…` is also
read, but only from the current directory.

`bbvastl80` publishes through `L80` (or `l80`) on PATH; set `L80_BIN` to use
another. BetterBench comes from `L80_BETTERBENCH` or PATH and must be 0.5.0 or
later (results schema 2); L80 pins 0.6.1.

## The spec

`run.example.yaml` is fully commented. Two shapes for `serving`:

- **`mode: engine`** — pick a preset (`vllm` or `llamacpp`); the app generates
  the install + launch for you. A minimal spec (just `model:`) defaults to vLLM.
  `sglang`, `tgi` and `ollama` are reserved names without a preset yet: the spec
  rejects them, so use `mode: custom` for those stacks.
- **`mode: custom`** — bring your own `image` + `setup` + `start` + `health`.
  This is the "offload *my* stack" path; the lean form is image + setup + start.

`gpu` filters any GPU (names, count, min RAM, $/hr ceiling, type, reliability,
datacenter) with a `pick` policy (`cheapest`/`first`/index). `budget` caps spend
and provision time. `benchmark`/`publish`/`connect`/`lifecycle` round it out.

## CLI

| command   | what it does |
|-----------|--------------|
| `init`    | create a starter `.env`, run `doctor` |
| `doctor`  | check Vast key + l80/betterbench/ssh |
| `search`  | show candidate offers for the spec's GPU filter |
| `list`    | list all instances on the account |
| `status`  | show one instance (or a run's) with its port mappings |
| `rent`    | rent an offer, wait until running, save per-run state |
| `destroy` | tear down an instance (or a run's) — irreversible |
| `reap`    | destroy every instance carrying our label |
| `serve`   | rent + provision → the model serving at `/v1` (connect lands in M4) |
| `provision` | bootstrap a running box into serving the model (idempotent) |
| `ssh`     | open a shell on the box, or run one command |
| `logs`    | tail the bootstrap + server logs on the box |
| `connect` | resolve the endpoint (direct port or SSH tunnel) + verify `/v1` |
| `bench`   | run pinned BetterBench against an endpoint → validated `results.json` |
| `publish` | publish a `results.json` via the L80 CLI (validated first) |
| `run`     | the full pipeline: rent → provision → connect → bench → publish → destroy (`--up-to`, `--dry-run`, `--keep`) |

## Design notes

- **One contract, any engine.** A provider (`providers/`) is just
  `prepare_cmds + launch_cmd + is_ready`; the base class renders them into one
  idempotent bootstrap script. Engine presets and the lean custom provider are
  interchangeable; `serve`/`run` never know which is in use.
- **SSH is the transport; `onstart` is the fast path.** The Vast "execute"
  endpoint only runs `ls`/`rm`/`du`, so every command reaches a box over SSH.
  `rent` registers your `~/.ssh/id_ed25519.pub` with the account first and
  hands the bootstrap script to Vast as `onstart`, so the box starts serving on
  first boot; `provision` re-runs the same script over SSH (a no-op when
  healthy, a repair otherwise).
- **Resumable + crash-safe.** Each run writes `runs/<id>/state.json`. Phases are
  idempotent; `destroy --run <id>` and `reap` work even after a crash.
- **No watchdog if the process dies.** If `bbvastl80` is killed (closed
  terminal, lost SSH session, power loss), the rented box keeps billing until
  you run `destroy --run <id>` or `reap`. `L80 vast run --max-hours N` adds a
  wall-clock watchdog that destroys the instance when time runs out, and
  `L80 vast status` lists runs that may still be billing.
- **The 404 trap.** Before publishing, we validate that at least one
  `single_stream[].ok` is true — a mis-`/v1` endpoint produces a full-size
  results file where every record is `ok: false`, and we must not publish that.
- **Live-API-verified client.** The Vast client follows 301→trailing-slash
  redirects, treats the API's 404-with-`auth_error` as an auth failure, and
  reports non-JSON bodies cleanly. `order` is a list of `[field, dir]` tuples and
  `type` is a plain enum — both confirmed against the live endpoint.

## Layout

```
src/bbvastl80/
  cli.py            # typer CLI (the surface above)
  config.py         # Config.load(): key + l80/betterbench/ssh + runs dir
  spec.py           # RunSpec + sub-models (pydantic), load_spec()
  state.py          # RunState + StateStore (resumability)
  log.py            # rich logging helpers
  vast/
    client.py       # VastClient (httpx) + error hierarchy
    models.py       # Offer / Instance / User (tolerant of schema drift)
    offers.py       # build_search_body() + pick_offer()
    instance.py     # rent_body / rent / wait_for_running
    sshkeys.py      # find local key, register on account, attach to instance
  providers/
    base.py         # Box, SSHExecTransport, Provider (+ bootstrap script renderer)
    engines.py      # vLLM / llama.cpp presets
    custom.py       # your image + setup[] + start + health
  bootstrap.py      # render_bootstrap (onstart) + provision (SSH push + wait)
  connect.py        # direct vs tunnel, verify_v1
  bench.py          # spawn pinned BetterBench + validate_results (never publish zeros)
  publish.py        # spawn l80, parse the page URL
  run.py            # the pipeline + teardown policy + cost estimate
specs/              # real run specs (e.g. 2x H200 NVL quick bench)
tests/              # 89 tests: spec, offers, client (respx), instance, providers,
                    # bootstrap, sshkeys, connect, bench/publish, pipeline e2e
run.example.yaml    # fully-commented starter spec
```
