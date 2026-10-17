# BB-VAST-L80 — Plan

A local orchestrator that rents **any** GPU on Vast.ai, provisions a model
serving stack on it, connects back to this machine, runs **BetterBench**
against it, and publishes the results with the **L80** CLI — then tears the
rental down. One spec file drives the whole loop, so a new run on a new GPU
is a one-liner.

```
            ┌────────────────────────── this machine ──────────────────────────┐
  Vast.ai   │                                                                    │
  console   │   bbvastl80 (Python)                                              │
  /api/v0   │     │  search → rent → provision → connect → bench → publish → reap
  ────────► │     ▼                                                                    │
  REST      │  Vast GPU box                      http://…/v1      BetterBench 0.4.0   │
  (rent,    │  (disposable                       ────────────────►  (pinned, local)   │
   exec,    │  model server)     SSH tunnel / direct port            │  results.json   │
   ssh)     │     ▲  0.0.0.0:8000  (vLLM/llama.cpp/…)                ▼                 │
            │     └──────────────────────────────────────────────  L80 publish ────────┼──► launch80.com
            │                                                                            │
            └────────────────────────────────────────────────────────────────────────────┘
```

**Core design decision:** keep *BetterBench and L80 local*. Only the GPU and
the model server are remote. The remote box is a disposable OpenAI-compatible
`/v1` server; we reach it over a Vast public port **or** an SSH tunnel that
makes it look like `http://localhost:8000/v1`. This preserves the pinned
harness (version comparability), keeps L80's "endpoint/host never leave this
machine" privacy behavior intact, and means the entire existing L80 workflow
works unchanged.

**The one contract.** Everything downstream (BetterBench, L80, and any other
consumer) depends on exactly one thing: *the box is serving your model at an
OpenAI-compatible `/v1` endpoint, and we know its URL and the model id*. How
the box gets there is a **pluggable provider** — a preset for a known engine,
or a fully user-driven "custom" spec (your image / Dockerfile / setup / start
command). This is what makes "just offload GPU needs to a rented box and run
*my* stack" a first-class use case, not a special case. See §2.1.

---

## 1. What already exists (grounding facts)

Verified on this machine:

| Thing | Status |
|---|---|
| `l80` CLI | `~/.local/bin/l80`, v0.4.0, token at `~/.config/L80/credentials.json` (`l80 doctor` → reachable) |
| BetterBench | pinned via `L80_BETTERBENCH=$HOME/bench/BetterBench/.venv/bin/betterbench`, version **0.4.0** |
| `python3` / `ssh` / `uv` | 3.14.4 / present / present |

L80 subcommands we drive (shelling out, exactly as a human would):
- `l80 betterbench --results <results.json>` → publish an existing run
- `l80 betterbench --results <f> --dry-run` → cheap validation (no run, no publish)
- `l80 publish <f> --template <id>`, `l80 templates list`, `l80 doctor`
- templates: `bench.betterbench.v1` (raw file, 4 MB cap) · `bench.report.v1` (summary)
- free tier: 10 publishes/rolling day, pages immutable, auto-deleted after 5 days
- **trap (from harness notes):** endpoint must end in `/v1`; always check
  `single_stream.<cat>[].ok` before trusting/publishing a results file (a
  404 endpoint produces a full-size file where every request failed).

BetterBench results.json (v0.4.0) top-level keys:
`schema, betterbench_version, corpus_version, config, env, single_stream,
prefill, concurrency, sample_gate`. `env` carries `endpoint, model, gpu,
host, os, python, notes, max_model_len`.

Vast.ai API — base `https://console.vast.ai/api/v0/`, auth
`Authorization: Bearer $VAST_API_KEY`. Endpoints we use (confirmed from the
published OpenAPI spec):

| Operation | Method + path |
|---|---|
| account (credit, ssh_key) | `GET /users/current/` |
| search offers | `POST /bundles` (filters + `order`, `limit`, `type`) |
| rent an offer | `POST /asks/{offer_id}/` → returns `new_contract` (instance id) |
| instance status / ip / ports | `GET /instances/{id}` |
| list my instances | `GET /instances` |
| run a command on the box | **SSH** (`root@<ip> -p <22/tcp external>`, else `ssh_host:ssh_port`). `PUT /instances/command/{id}` only runs `ls`/`rm`/`du` |
| register / attach SSH public key | `POST /ssh/` (account) · `POST /instances/{id}/ssh/` (one instance), body `{ssh_key}` |
| destroy (irreversible) | `DELETE /instances/{id}` |
| change bid | `PUT /instances/bid_price/{id}` |
| logs | `GET /instances/request_logs/{id}` |

`POST /bundles` filter fields (each is `{"op": value}`, ops: `eq neq gt gte
lt lte in notin`): `gpu_name`, `num_gpus`, `gpu_total_ram`, `dph_total`,
`datacenter`, `geolocation`, `reliability`, `verified`, `disk_space`,
`cpu_cores`, `cpu_ram`, `type` (`ondemand|bid|reserved`). Sort via `order`
(e.g. `dph_total`).

`POST /asks/{id}` body: `image` (required; default `vastai/base-image`),
`disk` (GB), `runtype` (`ssh|ssh_direct|ssh_proxy|jupyter|args`),
`args`/`args_str` (docker run args — this is how we open port 8000),
`env`, `label`, `onstart` (boot script), `price`, `template_hash_id`,
`target_state`.

**Networking model (the crux of connectivity):**
- Vast docker boxes share a public IP; each *internal* port is mapped to a
  *random external* port. `runtype: ssh` opens internal 22.
- We open the serving port too — in `env`, the way the official CLI does:
  `env: {"-p 8000:8000": "1"}` (`args` is the container command line, not
  docker flags).
- The instance reports its mappings as env vars: `VAST_TCP_PORT_22`
  (external SSH port) and `VAST_TCP_PORT_8000` (external serving port).
- Two ways back to us:
  - **direct** — `http://<public_ip>:<VAST_TCP_PORT_8000>/v1` (fastest; needs
    a machine with an open port).
  - **tunnel** — `ssh -p <VAST_TCP_PORT_22> root@<ip> -L 8000:localhost:8000 -N`
    → local `http://localhost:8000/v1` (works on every machine, private).

The `vastai` CLI (`curl -fsSL https://vast.ai/install.sh | bash`, `--raw`
for JSON) is a handy escape hatch / manual fallback, but the app talks to the
REST API directly for control and testability.

---

## 2. The run spec (the flexibility knob)

A single YAML file (or flags) describes a run. Everything is optional with
sane defaults, so a 3-line spec works and a 60-line spec is also fine.

```yaml
# run.yaml  (run.example.yaml ships fully commented)
name: qwen3-27b-on-h100

gpu:                              # "whatever GPU I decide to rent"
  names: [H100_SXM5, A100_SXM4_80GB]  # gpu_name  (omit = cheapest that fits)
  count: 1                          # num_gpus
  min_ram_gb: 80                    # gpu_total_ram
  max_dph: 2.5                      # $/hr ceiling (dph_total)
  type: ondemand                    # ondemand | bid | reserved
  verified: true
  min_reliability: 0.99
  datacenter: null                  # optional pin
  geolocation: null                 # optional pin
  pick: cheapest                    # cheapest | first | <index>

model:
  id: Qwen/Qwen3-27B               # the id BetterBench requests at /v1
  source: hf                        # hf | url | local-path
  hf_repo: Qwen/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16

serving:
  mode: engine                      # engine (preset) | custom (your own stack)
  port: 8000
  model_id: Qwen/Qwen3-27B         # id served at /v1 — used to verify readiness + for BetterBench

  # --- mode: engine — pick a preset; we generate the install + launch cmds ---
  engine: vllm                      # vllm | llamacpp | sglang | tgi | ollama
  max_model_len: 8192
  gpu_mem_util: 0.90
  extra_args: []                    # appended to the preset launch command

  # --- mode: custom — bring your own setup. Only image + start are required; ---
  #     everything else is optional. This is the "offload my stack to a GPU" path.
  image: null                       # a full image (already has your stack) OR a base to provision
  dockerfile: null                  # path/URL → build an image on the box first, then run it
  setup: []                         # shell cmds run in-container before start (git pull, pip, weights…)
  start: ""                         # the command that launches the server (runs in foreground)
  health:                           # how we know it's ready (default polls /v1/models)
    path: /v1/models
    expects: null                   # substring that must appear; null → matches model_id
  weights:                          # how model weights get onto the box
    source: hf                      # hf | url | scp | none
    repo: null                      # HF repo id, a URL, or a local path (for scp)

`mode: engine` and `mode: custom` share `port` + `model_id`. A preset engine is
just a curated provider; `custom` is the general provider. Both satisfy the same
contract (§2.1), so `connect`/`bench`/`publish` are identical for either.

instance:
  disk_gb: 200
  runtype: ssh                      # ssh | ssh_direct | ssh_proxy
  label: bbvastl80
  onstart: ""                       # optional extra boot commands

benchmark:
  quick: false
  passes: 20
  warmup: 2
  phases: [decode, prefill, concurrency]   # maps to single_stream/prefill/concurrency
  notes:                            # become L80 chips + results.env.notes
    engine: vllm
    hardware: H100

publish:
  template: bench.betterbench.v1    # bench.betterbench.v1 | bench.report.v1
  title: null                       # default: model (L80 derives it)
  summary: null
  sections: []                      # "Heading=Body"
  dry_run: false

budget:
  max_total_usd: 10.0
  max_provision_min: 20             # give up waiting for the box if it exceeds this

connect:
  mode: auto                        # auto | direct | tunnel   (auto = direct if port is open)
  local_port: 8000

ssh:
  public_key: null                  # null → use the account's key (GET /users/current)

lifecycle:
  teardown: auto                    # auto | manual
  keep_running: false               # if true: leave the box up, print reconnect cmd
```

`pydantic` models in `spec.py` validate this, fill defaults, and reject
contradictions (e.g. `engine: llamacpp` with a CUDA-only `extra_args`).

### 2.1 The provider contract

A **provider** is anything that can take a running Vast box and bring it to:
*"serving `model_id` at an OpenAI-compatible `/v1` on `port`, ready per
`health`."* Two providers ship:

- **`EngineProvider`** (presets) — `vllm`, `llamacpp`, `sglang`, `tgi`,
  `ollama`. Each is a small data record `{default_image, install_cmd,
  launch_cmd(template)}`. Adding a new preset = adding one record; no other
  code changes.
- **`CustomProvider`** (yours) — runs your `dockerfile` (build) → your
  `setup[]` → your `start`, then waits on your `health`. No assumptions about
  what your stack is. This is the "offload my custom setup to a rented GPU"
  path: point it at your image (or a base + a script), and the rest of the
  pipeline is engine-agnostic.

Both implement the same three-method interface — `prepare(box)`,
`launch(box)`, `is_ready(endpoint) -> bool` — so `connect`/`bench`/`publish`
cannot tell them apart. Model weights are handled by a small `weights`
helper shared by both (HF pull / URL download / `scp` from this machine via a
Vast cloud-sync or the SSH tunnel).

---

## 3. CLI surface

```
bbvastl80 init                          generate run.example.yaml; run doctor
bbvastl80 doctor                        check VAST_API_KEY, l80, betterbench pin, ssh
bbvastl80 search   -s run.yaml         show candidate offers (table), no billing
bbvastl80 rent     -s run.yaml [--offer ID]     rent + wait for running; print id, ip, ports
bbvastl80 provision -s run.yaml [--instance ID] bootstrap the box (engine+model+server)
bbvastl80 connect  -s run.yaml [--instance ID] resolve endpoint; tunnel if needed; verify /v1
bbvastl80 serve    -s run.yaml [--offer ID] [--instance ID]   rent+provision+connect; print the live /v1 URL
bbvastl80 bench    -s run.yaml [--endpoint URL] run BetterBench locally → results.json
bbvastl80 publish  <results.json>      thin wrapper: l80 betterbench --results
bbvastl80 status   [--instance ID]     one instance / all
bbvastl80 destroy  [--instance ID | --run ID] [--yes]   tear down
bbvastl80 reap     [--yes]             destroy any leftover box labeled bbvastl80
bbvastl80 run      -s run.yaml [--up-to PHASE] [--dry-run] [--keep] [--offer ID] [--instance ID]
```

`run` is the full pipeline = **`serve` + `bench` + `publish`**. `serve` is the
reusable *offload* primitive — "get me a rented GPU running my serving spec
and hand me a live endpoint" — which you can then use for anything (BetterBench,
your own client, a quick manual test). The individual subcommands let you stop
at any phase, bring your own offer/instance, or recover after a crash.

---

## 4. The pipeline (`run`)

The pipeline has two halves. The **offload core** — `rent → wait →
provision → connect` — turns a spec into a *live `/v1` endpoint* and is
engine-agnostic (it just asks the selected provider to get there). The
**benchmark consumer** — `bench → publish` — then uses that endpoint. `serve`
runs only the offload core; `run` runs both. Each phase is a small,
idempotent function that reads and writes `runs/<run-id>/state.json`, so the
run is **resumable** and teardown works days later. Phases:

1. **doctor** — `VAST_API_KEY` set (or `vastai` configured), `l80 doctor`
   OK, `betterbench --version == 0.4.0`, `ssh` present. Fail fast with a
   precise message otherwise.

2. **search** — `POST /bundles` with the `gpu` filters → ranked list.
   Print a table (rank, GPU, $/hr, ram, datacenter, reliability, id). Apply
   `pick`. `--dry-run` stops here.

3. **cost guard** — `GET /users/current/`; if `credit < budget.max_total_usd`
   abort before we spend anything.

4. **rent** — `POST /asks/{offer_id}/` with `image` (engine default),
   `disk`, `runtype`, `args: ["-p","<port>:<port>"]`, `label`, `onstart`,
   and (for `bid`) `price`. → `new_contract`. Record instance id in state.

5. **wait** — poll `GET /instances/{id}` until `status` is a running state
   and `public_ip` + the `VAST_TCP_PORT_*` mappings are populated. Bound by
   `budget.max_provision_min`. Surface Vast logs on timeout
   (`GET /instances/request_logs/{id}`).

6. **provision** — bring the box to a live `/v1` with an **engine-aware
   bootstrap script** that the provider renders (`Provider.script`). The
   script installs the engine if the image lacks it, stages weights (`hf`
   repo / URL), starts the server bound to `0.0.0.0:<port>` with
   `max_model_len` / `gpu_mem_util` / `extra_args` under `setsid nohup`, and
   is **idempotent** (exits if `/v1` already answers or a launch is in flight).
   It reaches the box two ways — same script, two deliveries:
   - **`onstart`** (default, `instance.bootstrap: onstart`) — the script *is*
     the instance's `onstart`, so Vast runs it on first boot and the box is
     usually pulling weights before it even reports `running`.
   - **SSH** — `provision` always pushes the script (`base64 -d`) and runs it
     over SSH too; thanks to idempotency this is a no-op on a healthy box and
     a repair on a box whose onstart never ran (resume, `bootstrap: ssh`).
   Then poll `curl localhost:<port>/v1/models` *inside* the box until it
   answers; on failure, dump `/tmp/bbvastl80/{bootstrap,server}.log`.
   - **The Vast execute endpoint is NOT a shell.** `PUT /instances/command/{id}`
     is constrained to `ls`, `rm` and `du` (per the official `vastai execute`
     help; `command` ≤ 512 chars). Anything else never produces output, and the
     S3 `result_url` then answers 403 — which is what blocked M3 and looked like
     an auth problem. It is kept only as a read-only sanity helper.
   - **SSH keys.** The account's public keys are injected at creation, so
     `rent` registers `~/.ssh/id_ed25519.pub` (or `ssh.identity`/`public_key`)
     via `POST /ssh/` *before* renting, and attaches it to the instance after
     (`POST /instances/{id}/ssh/`) as belt-and-braces. SSH target = direct
     `22/tcp` mapping on the public IP, else Vast's proxy `ssh_host:ssh_port`.
   - Engine registry (`providers/engines.py`) maps each engine to
     `{default_image, prepare_cmds, launch_cmd}`:
     - `vllm`: `vllm serve <repo> --served-model-name <id> --host 0.0.0.0 --port 8000 --max-model-len N --gpu-memory-utilization U [--tensor-parallel-size count]`
     - `llamacpp`: `llama-server -m <model.gguf> --host 0.0.0.0 --port 8000 --alias <id> -c N`
     - `sglang` / `tgi` / `ollama`: M7

7. **connect** — resolve the local endpoint:
   - `mode: direct` (or auto + a public 8000 port exists) →
     `http://<ip>:<VAST_TCP_PORT_8000>/v1`
   - `mode: tunnel` → write a throwaway `ssh -F` config (host, port,
     identity/key, `ServerAliveInterval`, `ExitOnForwardFailure yes`) and
     spawn `ssh -F cfg -N -L <local_port>:localhost:<port>` in the
     background; wait until the local port accepts; store the PIDs for clean
     shutdown. Endpoint → `http://localhost:<local_port>/v1`.
   - Verify: `GET <endpoint>/models` returns `model.id`. The `/v1` suffix is
     guaranteed here (the harness trap).

8. **bench** — invoke the pinned harness exactly as pinned:
   ```
   $L80_BETTERBENCH run --endpoint <endpoint> --model <model.id> \
       --out runs/<run-id>/results.json \
       [--quick] [--passes N] [--warmup N] [--decode --prefill --concurrency]
   ```
   Then **validate** (the 404 trap): read `results.json`, confirm at least one
   `single_stream.<cat>` record has `ok: true`. If everything failed → abort
   with the endpoint/transport diagnosis *before* publishing (we never publish
   a page of zeros).

9. **publish** — if `publish.dry_run`, `l80 betterbench --results <f> --dry-run`.
   Otherwise `l80 betterbench --results <f>` (template from spec; L80 strips
   `env.endpoint`/`env.host` and prints the URL). Capture the URL into state.
   The local `results.json` remains the durable artifact of record.
   (Publishing is public — the app prints a clear "publishing now" line and
   honors `--dry-run` for a no-surprises preview.)

10. **reap** — if `teardown: auto` (and not `--keep`): close the tunnel,
    `DELETE /instances/{id}`. Print the final URL, the local artifact path,
    and estimated cost (`dph × runtime`). If `keep_running`: print the
    reconnect/tunnel command and the `destroy --run <id>` to finish later.

`--up-to <phase>` stops cleanly after that phase (e.g. `--up-to connect`
hands you a live endpoint to poke at manually). `--dry-run` runs
search+plan without renting. `--offer`/`--instance` skip search/rent.

---

## 5. Project structure

```
BB-VAST-L80/
├── README.md
├── PLAN.md
├── pyproject.toml                  # deps: httpx, typer, pydantic, rich, respx(dev)
├── .env.example                    # VAST_API_KEY=…            (L80 token stays in ~/.config/L80)
├── run.example.yaml                # fully commented starter spec
├── runs/                           # gitignored: <run-id>/state.json, results.json
├── src/bbvastl80/
│   ├── __init__.py
│   ├── cli.py                      # typer app + subcommands
│   ├── config.py                   # VAST_API_KEY / l80 path / betterbench pin / base urls
│   ├── spec.py                     # pydantic RunSpec + defaults + cross-field validation
│   ├── log.py                      # rich/structured logging
│   ├── state.py                    # run-id, state.json read/write, run dir
│   ├── run.py                      # the phase pipeline + --up-to/--dry-run/resume
│   ├── vast/
│   │   ├── client.py               # httpx client: auth header, retries, typed errors
│   │   ├── models.py               # pydantic models for the /api/v0 responses
│   │   ├── offers.py               # build /bundles filters, rank, pick
│   │   ├── instance.py             # rent (+onstart bootstrap), poll, destroy
│   │   └── sshkeys.py              # find local key, register on account, attach to instance
│   ├── providers/
│   │   ├── base.py                 # Provider contract: prepare / launch / is_ready
│   │   ├── engines.py              # built-in presets: vllm/llamacpp/sglang/tgi/ollama
│   │   ├── custom.py               # your stack: image(+dockerfile) + setup[] + start + health
│   │   ├── weights.py              # get model weights onto the box (hf / url / scp)
│   │   └── bootstrap.py            # render the remote provisioning script for a provider
│   ├── connect.py                  # direct vs tunnel; ssh -F config; /v1 verify; cleanup
│   ├── bench.py                    # spawn $L80_BETTERBENCH; parse + validate results
│   └── publish.py                  # spawn l80; parse the published URL
├── scripts/
│   └── bootstrap.sh.tmpl           # the remote script (also embedded in bootstrap.py)
└── tests/
    ├── conftest.py                 # respx fake Vast API; fake l80/betterbench on PATH
    ├── test_spec.py
    ├── test_offers.py
    ├── test_instance.py
    ├── test_provision.py
    ├── test_connect.py
    ├── test_bench.py
    └── test_run_pipeline.py
```

**Stack & why:** Python 3.10+ (matches the BetterBench/ML world; trivial
subprocess + REST + ssh). `httpx` for the Vast API (testable with `respx`),
`typer` for the CLI, `pydantic` v2 for the spec/models, `rich` for tables and
progress. The SSH tunnel is a background `ssh -F … -N -L …` subprocess — no
Python SSH library needed, and it inherits system key/agent handling.

**Testing strategy:** a `respx`-mocked Vast API returns scripted
offer/instance/execute responses; `l80` and `betterbench` are replaced by
tiny fakes on `PATH` that write a canned results.json and print a URL. The
full `run` pipeline is then testable offline and deterministically, including
the "all requests failed → don't publish" guard.

---

## 6. Safety & flexibility features

- **`--dry-run`** on `run`: plan search + rent without billing.
- **`--up-to <phase>`**: stop at any phase.
- **Bring your own**: `--offer <id>` (skip search), `--instance <id>` (skip
  rent+provision; just connect/bench/publish an existing box).
- **Cost guard** before renting; **runtime + $** reported at the end.
- **Resumable** runs; `destroy --run <id>` and `reap` work after a crash.
- **Never publish zeros** (validate `single_stream[].ok` first).
- **Secrets**: `VAST_API_KEY` from env or `~/.config/bbvastl80/`; the L80
  token is read by L80 from disk (never pasted into chat); results always
  saved locally before publishing.
- **Idempotent teardown** + a `reap` that clears any stray `bbvastl80` boxes
  so a forgotten run can't keep billing.

---

## 7. Milestones (build order)

> **Status (2025):** M0–M2 are **built, tested (26 tests), pyflakes-clean, and
> live-validated** against the real Vast API. A real rental was executed
> end-to-end (rent → poll → destroy, box confirmed gone). Live-API contract
> findings baked in: create is `PUT /asks/{id}/` with `client_id:"me"` (v0);
> `get`/`destroy` are v0 (record fields `actual_status`/`public_ipaddr`);
> `list` is the only v1 endpoint; `order` is a list of `[field,dir]` tuples;
> `type` is a plain enum; `verified`/`reliability`/`min_network_*` are operator
> objects; endpoints 301-redirect to a trailing slash (client follows); a bad
> key is a 404 with `{"error":"auth_error"}`. `rent` tries the next ranked offer
> when one is stale. **M3 built** on SSH + `onstart` after the execute-endpoint
> approach proved impossible (it only runs `ls/rm/du`; 403s came from its S3
> result URL, not from the key). Next up: live-validate M3, then **M4 (connect)**.

- **M0 — skeleton.** `pyproject`, package layout, `typer` CLI, `config.py`,
  `spec.py` (full pydantic model + defaults), `init` + `doctor`. ✅ done
- **M1 — read-only Vast.** `vast/client.py`, `offers.py`, `models.py`;
  commands `search`, `list`, `status`. Safe to hit the real API first. ✅ done
- **M2 — rent + lifecycle.** `instance.py` (rent, poll, destroy, logs),
  cost guard; commands `rent`, `destroy`, `reap`. ✅ done, live-validated
  (real rent→poll→destroy round-trip; `PUT /asks/{id}/`+`client_id` contract)
- **M3 — provision.** `engines.py` (vLLM + llama.cpp), `bootstrap.py`,
  provider-rendered bootstrap script delivered as `onstart` + over SSH; SSH key
  registration; commands `provision`, `ssh`, `logs`. ✅ built (59 tests);
  **needs one live rent→serve run to validate** (the execute-endpoint design it
  replaces was a dead end — see §4 step 6).
- **M4 — connect.** `connect.py`: direct + tunnel, `ssh -F` config, `/v1`
  verification, clean shutdown; command `connect`.
- **M5 — bench + publish.** `bench.py` (spawn pinned harness + validation
  guard), `publish.py` (spawn L80, parse URL); commands `bench`, `publish`.
- **M6 — the pipeline.** `run.py` wiring all phases with state/resume,
  `--up-to`, `--dry-run`, `--keep`, `--offer/--instance`; end-to-end test
  against the fakes.
- **M7 — polish.** README, `run.example.yaml`, richer error messages, cost
  reporting, sglang/tgi/ollama engines, interruptible (`bid`) pricing path.

M0–M2 give a working "rent + tear down" you can trust; M3–M5 add the
benchmark+publish value; M6 makes it a one-command product.

---

## 8. Open decisions (confirm before/while building)

1. **Serving default** — `mode: engine` with vLLM as the default preset? And
   how much of the `custom` provider do you want up front (just
   image+setup+start, or also on-box Dockerfile builds + weight `scp`)?
2. **A concrete custom setup** — if you have a specific stack in mind (a
   Dockerfile, a repo with a start script, a quant you maintain), share its
   shape and I'll size the `custom` provider to it exactly rather than
   guessing the surface.
3. **Direct vs tunnel default** — `auto` (direct when the box exposes a
   public 8000 port, else tunnel) is the proposed default. OK?
4. **Publish by default** — should `run` publish automatically, or stop after
   `bench` and require an explicit `publish`? (Publishing is public; a
   `--publish` flag to opt in is the safer default.)
5. **`vastai` CLI vs raw REST** — plan uses raw REST for control. Fine to
   also shell to `vastai --raw` for the few ops it's better at?
