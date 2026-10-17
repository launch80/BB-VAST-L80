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

**Built (M0–M3):** project skeleton, config/doctor, the read-only Vast surface
(`search`, `list`, `status`), the rent/teardown lifecycle (`rent`, `destroy`,
`reap`) with per-run state for resumability, and provisioning (`serve`,
`provision`, `ssh`, `logs`): the provider renders one idempotent bootstrap
script that ships as the instance's `onstart` and is also pushed over SSH. The
remaining pipeline commands (`connect`, `bench`, `publish`, `run`) are stubs
so the CLI surface is stable; they light up in M4–M6.

The Vast client is exercised against the **live API** (search/list/status all
return clean, actionable errors with a valid key).

## Install

```bash
cd BB-VAST-L80
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
`instance_read`, `instance_write`). Put it in `.env` as `VAST_API_KEY=…` or in
`~/.config/bbvastl80/credentials` as `api_key=…`.

## The spec

`run.example.yaml` is fully commented. Two shapes for `serving`:

- **`mode: engine`** — pick a preset (`vllm`/`llamacpp`/`sglang`/`tgi`/`ollama`);
  the app generates the install + launch for you. A minimal spec (just `model:`)
  defaults to vLLM.
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
| `connect` | *(M4)* resolve the endpoint (direct port or SSH tunnel) + verify `/v1` |
| `bench`   | *(M5)* run pinned BetterBench against an endpoint → `results.json` |
| `publish` | *(M5)* publish a `results.json` via the L80 CLI |
| `run`     | *(M6)* the full pipeline: serve + bench + publish |

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
  # M4+ will add: connect.py (tunnel), bench.py, publish.py, run.py
tests/              # 59 tests: spec, offers, client (respx), instance, providers,
                    # bootstrap, sshkeys
run.example.yaml    # fully-commented starter spec
```
