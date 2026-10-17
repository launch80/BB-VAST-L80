"""Instance lifecycle: rent an offer, wait for it to run, tear it down.

The rent body opens the serving port as a public container port
(``-p <port>:<port>``) so the box can be reached directly; the SSH runtype
additionally opens 22 for the tunnel fallback.
"""
from __future__ import annotations

import time

from bbvastl80 import log
from bbvastl80.providers import resolve_image, resolve_port
from bbvastl80.spec import InstanceSpec, ServingSpec
from bbvastl80.vast.client import VastAuthError, VastClient, VastError
from bbvastl80.vast.models import Instance, Offer


class ProvisionError(Exception):
    """The instance reached a failed terminal state (or we gave up waiting)."""


def rent_body(offer: Offer, serving: ServingSpec, instance: InstanceSpec) -> dict:
    port = resolve_port(serving)
    body: dict = {
        "client_id": "me",  # required by the asks endpoint (see vast-cli)
        "image": resolve_image(serving),
        "disk": instance.disk_gb,
        "runtype": instance.runtype,
        "label": instance.label,
        # open the serving port (and 22 is implicit for the ssh runtypes)
        "args": ["-p", f"{port}:{port}"],
    }
    if instance.onstart:
        body["onstart"] = instance.onstart
    if instance.price is not None:  # honoured for bid/interruptible offers
        body["price"] = instance.price
    return body


def rent(client: VastClient, offer: Offer, serving: ServingSpec,
         instance: InstanceSpec) -> int:
    """Rent ``offer``; returns the new instance (contract) id."""
    body = rent_body(offer, serving, instance)
    log.detail(f"renting offer {offer.id} ({offer.label}) image={body['image']} "
               f"disk={body['disk']}GB runtype={body['runtype']}")
    return client.create_instance(offer.id, body)


def wait_for_running(client: VastClient, instance_id: int, *, timeout_min: float,
                     poll_sec: float = 10.0) -> Instance:
    """Poll until the instance is ``running`` (or a failed terminal state)."""
    deadline = time.time() + timeout_min * 60
    last: Instance | None = None
    while time.time() < deadline:
        try:
            last = client.get_instance(instance_id)
        except VastAuthError:
            raise  # a key/scope problem won't fix itself by polling — fail fast
        except VastError:
            last = None  # transient: network blip, or not resolvable yet (404/null)
        if last is not None:
            if last.is_running:
                return last
            if last.is_terminal_failed:
                raise ProvisionError(
                    f"instance {instance_id} entered failed state '{last.status}' "
                    f"(see 'bbvastl80 status --instance {instance_id}')"
                )
            log.detail(f"instance {instance_id}: {last.status} …")
        time.sleep(poll_sec)
    raise ProvisionError(
        f"instance {instance_id} not running after {timeout_min:.0f} min "
        f"(last status: {last.status if last else 'unknown'})"
    )
