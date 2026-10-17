"""SSH key plumbing: find a local key pair, make sure Vast knows the public
half, and attach it to a running instance.

Why this exists: the Vast execute endpoint cannot run arbitrary commands, so
*every* interaction with a box (provisioning, readiness probes, logs, the M4
tunnel) goes over SSH. Vast injects the account's registered public keys into
a container at creation time — the docs say to register **before** creating
the instance — and Docker instances also accept a key after the fact via
``POST /instances/{id}/ssh/``. We do both, so a key added late still works.
"""
from __future__ import annotations

from pathlib import Path

from bbvastl80 import log
from bbvastl80.config import ConfigError
from bbvastl80.vast.client import VastClient, VastError

DEFAULT_IDENTITIES = ("id_ed25519", "id_ecdsa", "id_rsa")


def resolve_identity(identity: str | None, *, required: bool = True) -> Path | None:
    """The private key to use: ``ssh.identity`` from the spec, else the first
    default key under ``~/.ssh``. Raises :class:`ConfigError` when nothing
    usable exists (and ``required``)."""
    if identity:
        p = Path(identity).expanduser()
        if not p.is_file():
            raise ConfigError(f"ssh.identity {p} does not exist")
        return p
    for name in DEFAULT_IDENTITIES:
        p = Path.home() / ".ssh" / name
        if p.is_file() and p.with_suffix(".pub").is_file():
            return p
    if required:
        raise ConfigError(
            "no SSH key found. Create one with:\n"
            "    ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519\n"
            "or point ssh.identity at an existing private key."
        )
    return None


def resolve_public_key(public_key: str | None, identity: Path | None) -> str:
    """The public key text to register with Vast.

    ``ssh.public_key`` may be the key itself or a path to a ``.pub`` file;
    otherwise it's ``<identity>.pub``.
    """
    if public_key:
        text = public_key.strip()
        if not text.startswith("ssh-") and not text.startswith("ecdsa-"):
            p = Path(text).expanduser()
            if not p.is_file():
                raise ConfigError(f"ssh.public_key {text!r} is neither a key nor a file")
            text = p.read_text().strip()
        return _check_public(text)
    if identity is None:
        raise ConfigError("no public key: set ssh.public_key or create ~/.ssh/id_ed25519")
    pub = identity.with_suffix(identity.suffix + ".pub")
    if not pub.is_file():
        raise ConfigError(f"public key {pub} not found next to {identity}")
    return _check_public(pub.read_text().strip())


def _check_public(text: str) -> str:
    if "PRIVATE KEY" in text:
        raise ConfigError("that is a PRIVATE key — pass the .pub half")
    if not text.lower().startswith(("ssh-", "ecdsa-")):
        raise ConfigError("ssh public key must start with ssh-… / ecdsa-…")
    return text


def key_body(text: str) -> str:
    """``type base64`` — the comparable part of a public key (comment dropped)."""
    parts = text.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else text.strip()


def ensure_account_key(client: VastClient, public_key: str) -> bool:
    """Register ``public_key`` on the account unless it's already there.
    Returns True when a new key was added."""
    body = key_body(public_key)
    try:
        existing = client.list_ssh_keys()
    except VastError as e:
        log.warn(f"could not list account SSH keys ({e}); will register anyway")
        existing = []
    for k in existing:
        if key_body(str(k.get("public_key") or k.get("ssh_key") or "")) == body:
            log.detail("SSH public key already registered on the account")
            return False
    client.create_ssh_key(public_key)
    log.success("registered SSH public key on the Vast account")
    return True


def attach_key(client: VastClient, instance_id: int, public_key: str) -> bool:
    """Attach the key to a running instance (belt and braces). Never raises."""
    try:
        client.attach_ssh_key(instance_id, public_key)
        log.detail(f"attached SSH key to instance {instance_id}")
        return True
    except VastError as e:
        log.detail(f"could not attach SSH key to {instance_id} ({str(e)[:80]}); "
                   "relying on the account key")
        return False
