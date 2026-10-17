"""A small httpx client for the Vast.ai REST API.

Versioning is *mixed* (confirmed against the live endpoint and the official
``vast-cli`` Python client, which is the source of truth):

* **Offers search** → **v0**: ``POST /api/v0/bundles``.
* **Instance *list*** → **v1**: ``GET /api/v1/instances/`` (the v0 list 410s).
* **Every other instance op** (single get / create / destroy / command) →
  **v0**. In particular, creating an instance from a *single* offer is
  ``PUT /api/v0/asks/{id}/`` — a **PUT**, not a POST (a POST 404s), and the
  payload must carry ``client_id: "me"``.

The client keeps a v0 base (the default) and derives a v1 base used only by the
list call. Paths are relative and joined explicitly (no httpx base_url) so there
is no RFC-3986 path-joining ambiguity.
"""
from __future__ import annotations

import json as jsonlib
import time
from typing import Any

import httpx

from bbvastl80.vast.models import Instance, Offer, User

# Relative paths. Only the *list* call uses the v1 base; the rest use v0.
P_USER = "users/current/"
P_BUNDLES = "bundles"            # v0
P_ASKS = "asks/{id}/"            # v0 (PUT for a single-offer create)
P_INSTANCE = "instances/{id}"    # v0
P_INSTANCES = "instances"        # v1 (list)
P_COMMAND = "instances/command/{id}"   # v0 — constrained: ls / rm / du only
P_LOGS = "instances/request_logs/{id}"  # v0
P_SSH_KEYS = "ssh/"                     # v0 — account-level public keys
P_INSTANCE_SSH = "instances/{id}/ssh/"  # v0 — attach a key to one instance


class VastError(Exception):
    """Base class for all Vast API failures."""


class VastAuthError(VastError):
    """401/403 — bad or missing/under-scoped API key."""


class VastHTTPError(VastError):
    def __init__(self, method: str, path: str, status: int, detail: str):
        self.method, self.path, self.status = method, path, status
        super().__init__(f"{method} {path} -> HTTP {status}: {detail}")


class VastClient:
    def __init__(self, api_key: str, base_url: str = "https://console.vast.ai/api/v0",
                 timeout: float = 30.0):
        if not api_key:
            raise VastAuthError("no API key supplied")
        self.base_url = base_url.rstrip("/")          # v0 (offers search)
        # Instance lifecycle moved to v1; derive it from the v0 base.
        self.v1_base_url = self.base_url.replace("/v0", "/v1")
        # No httpx base_url: we build explicit full URLs so there is no
        # ambiguity in path joining (RFC 3986 would drop the /v0 segment).
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            follow_redirects=True,  # Vast 301-redirects paths to their trailing-slash form
        )

    def __enter__(self) -> "VastClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # --- transport ----------------------------------------------------------
    def _request(self, method: str, path: str, *, params: dict | None = None,
                 json: Any | None = None, base: str | None = None) -> Any:
        url = f"{(base or self.base_url).rstrip('/')}/{path.lstrip('/')}"
        try:
            r = self._http.request(method, url, params=params, json=json)
        except httpx.HTTPError as e:  # network-level failure
            raise VastError(f"network error calling {method} {url}: {e}") from e
        if r.status_code == 401:
            raise VastAuthError(
                f"{method} /{path} -> HTTP 401 (auth). "
                "Check VAST_API_KEY (scope: instance_read/instance_write)."
            )
        if r.status_code >= 400:
            detail = r.text[:300]
            # 403 is *not* a key-scope signal on this API: the execute endpoint
            # uses it for "blacklisted", create-instance for "not your own
            # machine", and the S3 result URL for "no such object". Surface the
            # body so the real reason is visible instead of blaming the key.
            if r.status_code == 403 and ("auth_error" in detail or "Invalid user key" in detail):
                raise VastAuthError(f"{method} /{path} -> HTTP 403 (auth). Invalid VAST_API_KEY.")
            # Vast reports a bad key as a 404 with {"error": "auth_error"}
            if r.status_code == 404 and ("auth_error" in detail or "Invalid user key" in detail):
                raise VastAuthError(
                    f"{method} /{path} -> HTTP 404 (auth). Invalid VAST_API_KEY."
                )
            raise VastHTTPError(method, path, r.status_code, detail)
        if not r.content:
            return {}
        try:
            return r.json()
        except jsonlib.JSONDecodeError as e:
            raise VastError(f"{method} /{path} returned non-JSON: {r.text[:200]!r}") from e

    # --- endpoints ----------------------------------------------------------
    def get_user(self) -> User:
        return User(**self._request("GET", P_USER))

    def search_offers(self, body: dict[str, Any]) -> list[Offer]:
        data = self._request("POST", P_BUNDLES, json=body)
        offers = data.get("offers", data if isinstance(data, list) else [])
        return [Offer(**o) for o in offers]

    def create_instance(self, offer_id: int, body: dict[str, Any]) -> int:
        # Single-offer create is a PUT on the v0 base (a POST 404s).
        data = self._request("PUT", P_ASKS.format(id=offer_id), json=body)
        new_contract = data.get("new_contract", data.get("id"))
        if new_contract is None:
            raise VastError(f"create-instance for offer {offer_id} returned no id: {data}")
        return int(new_contract)

    def get_instance(self, instance_id: int) -> Instance:
        data = self._request("GET", P_INSTANCE.format(id=instance_id),
                             params={"owner": "me"})
        # v0 wraps a single instance under {"instances": {...}} (null if not
        # resolvable yet); v1 is flat. Unwrap, and surface "not yet" as a 404.
        if isinstance(data, dict) and "instances" in data:
            data = data["instances"]
        if data is None:
            raise VastHTTPError("GET", P_INSTANCE.format(id=instance_id), 404,
                                "instance not resolvable yet")
        return Instance(**data)

    def list_instances(self) -> list[Instance]:
        data = self._request("GET", P_INSTANCES, base=self.v1_base_url)
        if isinstance(data, dict):
            for key in ("instances", "results", "offers"):
                if key in data and isinstance(data[key], list):
                    return [Instance(**x) for x in data[key]]
            # a single object
            return [Instance(**data)]
        if isinstance(data, list):
            return [Instance(**x) for x in data]
        return []

    def destroy_instance(self, instance_id: int) -> None:
        self._request("DELETE", P_INSTANCE.format(id=instance_id))

    # --- SSH keys -----------------------------------------------------------
    def list_ssh_keys(self) -> list[dict[str, Any]]:
        data = self._request("GET", P_SSH_KEYS)
        if isinstance(data, dict):
            for key in ("ssh_keys", "keys", "results"):
                if isinstance(data.get(key), list):
                    return data[key]
            return []
        return data if isinstance(data, list) else []

    def create_ssh_key(self, public_key: str) -> Any:
        return self._request("POST", P_SSH_KEYS, json={"ssh_key": public_key})

    def attach_ssh_key(self, instance_id: int, public_key: str) -> Any:
        return self._request("POST", P_INSTANCE_SSH.format(id=instance_id),
                             json={"ssh_key": public_key})

    def execute(self, instance_id: int, command: str, timeout: float = 60.0) -> str:
        """Run a **constrained** command on an instance and return its output.

        ``PUT /instances/command/{id}/`` only accepts ``ls``, ``rm`` and ``du``
        (per the official ``vastai execute`` help; ``command`` ≤ 512 chars). It
        is *asynchronous*: the response carries a ``result_url`` (S3) which we
        poll with a plain GET — no API auth header, matching the official
        client. S3 answers 403/404 until the object exists, so those are "not
        yet", not errors. If the response has no ``result_url`` we return its
        body. The box merges stdout+stderr, so the result is one text blob.
        """
        rj = self._request("PUT", P_COMMAND.format(id=instance_id), json={"command": command})
        result_url = rj.get("result_url") if isinstance(rj, dict) else None
        if not result_url:
            return rj if isinstance(rj, str) else jsonlib.dumps(rj)
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            time.sleep(0.4)
            rr = httpx.get(result_url, timeout=10)
            if rr.status_code == 200:
                return rr.text
            last = rr.status_code
        raise VastError(f"command result not ready within {timeout:.0f}s "
                        f"(result_url last answered HTTP {last}; the endpoint only runs "
                        f"ls/rm/du — was {command[:40]!r} allowed?)")

    def get_logs(self, instance_id: int, **params: Any) -> Any:
        return self._request("GET", P_LOGS.format(id=instance_id), params=params or None)
