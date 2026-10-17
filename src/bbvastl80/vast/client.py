"""A small httpx client for the Vast.ai REST API.

The API is split across two versions (confirmed against the live endpoint):

* **Offers search** lives on **v0**: ``POST /api/v0/bundles``.
* **Instance lifecycle** (create / get / list / command / destroy) lives on
  **v1**: ``/api/v1/instances…`` and ``/api/v1/asks/{id}``. The v0 equivalents
  are deprecated (list 410s; a v0 ``asks`` POST returns a raw HTML 404).

The client keeps a v0 base for search and derives a v1 base for everything else.
Paths are relative and joined explicitly (no httpx base_url) so there is no
RFC-3986 path-joining ambiguity.
"""
from __future__ import annotations

import json as jsonlib
from typing import Any

import httpx

from bbvastl80.vast.models import Instance, Offer, User

# Relative paths. OFFERS live on the v0 base; INSTANCES on the v1 base.
P_USER = "users/current/"
P_BUNDLES = "bundles"            # v0
P_ASKS = "asks/{id}/"            # v1
P_INSTANCE = "instances/{id}"    # v1
P_INSTANCES = "instances"        # v1
P_COMMAND = "instances/command/{id}"   # v1
P_LOGS = "instances/request_logs/{id}"  # v1


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
        if r.status_code in (401, 403):
            raise VastAuthError(
                f"{method} /{path} -> HTTP {r.status_code} (auth). "
                "Check VAST_API_KEY (scope: instance_read/instance_write)."
            )
        if r.status_code >= 400:
            detail = r.text[:300]
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
        data = self._request("POST", P_ASKS.format(id=offer_id), json=body, base=self.v1_base_url)
        new_contract = data.get("new_contract", data.get("id"))
        if new_contract is None:
            raise VastError(f"create-instance for offer {offer_id} returned no id: {data}")
        return int(new_contract)

    def get_instance(self, instance_id: int) -> Instance:
        data = self._request("GET", P_INSTANCE.format(id=instance_id), base=self.v1_base_url)
        # v0 wrapped a single instance under {"instances": {...}}; v1 is flat.
        # Unwrap if present so both shapes parse.
        if isinstance(data, dict) and isinstance(data.get("instances"), dict):
            data = data["instances"]
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
        self._request("DELETE", P_INSTANCE.format(id=instance_id), base=self.v1_base_url)

    def execute(self, instance_id: int, command: str) -> Any:
        """Queue a command on the box; returns the API response (has a result URL)."""
        return self._request("PUT", P_COMMAND.format(id=instance_id), json={"command": command},
                             base=self.v1_base_url)

    def get_logs(self, instance_id: int, **params: Any) -> Any:
        return self._request("GET", P_LOGS.format(id=instance_id), params=params or None,
                             base=self.v1_base_url)
