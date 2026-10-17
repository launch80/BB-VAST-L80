"""A small httpx client for the Vast.ai REST API.

Base URL is ``https://console.vast.ai/api/v0``; every path below is *relative*
to it (no leading slash), which is what makes httpx join them correctly.

Note on the search endpoint: it is ``POST /bundles`` (not ``/offers``) — that
is the current name in Vast's OpenAPI spec.
"""
from __future__ import annotations

import json as jsonlib
from typing import Any

import httpx

from bbvastl80.vast.models import Instance, Offer, User

# Relative paths (joined onto the api/v0 base url).
P_USER = "users/current/"
P_BUNDLES = "bundles"
P_ASKS = "asks/{id}/"
P_INSTANCE = "instances/{id}"
P_INSTANCES = "instances"
P_COMMAND = "instances/command/{id}"
P_LOGS = "instances/request_logs/{id}"


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
        self.base_url = base_url.rstrip("/")
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
                 json: Any | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
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
        data = self._request("POST", P_ASKS.format(id=offer_id), json=body)
        new_contract = data.get("new_contract", data.get("id"))
        if new_contract is None:
            raise VastError(f"create-instance for offer {offer_id} returned no id: {data}")
        return int(new_contract)

    def get_instance(self, instance_id: int) -> Instance:
        return Instance(**self._request("GET", P_INSTANCE.format(id=instance_id)))

    def list_instances(self) -> list[Instance]:
        data = self._request("GET", P_INSTANCES)
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

    def execute(self, instance_id: int, command: str) -> Any:
        """Queue a command on the box; returns the API response (has a result URL)."""
        return self._request("PUT", P_COMMAND.format(id=instance_id), json={"command": command})

    def get_logs(self, instance_id: int, **params: Any) -> Any:
        return self._request("GET", P_LOGS.format(id=instance_id), params=params or None)
