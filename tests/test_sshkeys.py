"""SSH key resolution + the account/instance key endpoints (respx)."""
from __future__ import annotations

import json

import pytest
from respx.models import MockResponse

from conftest import BASE
from bbvastl80.config import ConfigError
from bbvastl80.vast import sshkeys

PUB = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTests0000000000000000000000000 me@box"


@pytest.fixture
def keydir(tmp_path, monkeypatch):
    monkeypatch.setattr(sshkeys.Path, "home", classmethod(lambda cls: tmp_path))
    d = tmp_path / ".ssh"
    d.mkdir()
    return d


def test_resolve_identity_auto_detects_default_key(keydir):
    (keydir / "id_rsa").write_text("PRIVATE")
    (keydir / "id_rsa.pub").write_text("ssh-rsa AAAA x")
    ident = sshkeys.resolve_identity(None)
    assert ident == keydir / "id_rsa"
    assert sshkeys.resolve_public_key(None, ident) == "ssh-rsa AAAA x"


def test_resolve_identity_prefers_ed25519_and_needs_pub(keydir):
    (keydir / "id_rsa").write_text("x")
    (keydir / "id_rsa.pub").write_text("ssh-rsa AAAA")
    (keydir / "id_ed25519").write_text("x")  # no .pub -> skipped
    assert sshkeys.resolve_identity(None) == keydir / "id_rsa"
    (keydir / "id_ed25519.pub").write_text(PUB)
    assert sshkeys.resolve_identity(None) == keydir / "id_ed25519"


def test_resolve_identity_missing_is_a_config_error(keydir):
    with pytest.raises(ConfigError, match="ssh-keygen"):
        sshkeys.resolve_identity(None)
    assert sshkeys.resolve_identity(None, required=False) is None
    with pytest.raises(ConfigError, match="does not exist"):
        sshkeys.resolve_identity(str(keydir / "nope"))


def test_resolve_public_key_accepts_text_or_path_and_rejects_private(keydir, tmp_path):
    assert sshkeys.resolve_public_key(PUB, None) == PUB
    p = tmp_path / "k.pub"
    p.write_text(PUB + "\n")
    assert sshkeys.resolve_public_key(str(p), None) == PUB
    with pytest.raises(ConfigError, match="PRIVATE"):
        sshkeys.resolve_public_key("-----BEGIN OPENSSH PRIVATE KEY-----", None)


def test_key_body_ignores_comment():
    assert sshkeys.key_body(PUB) == sshkeys.key_body(PUB.rsplit(" ", 1)[0] + " other@host")


def test_ensure_account_key_registers_once(client, router):
    listing = router.get(f"{BASE}/ssh/").mock(
        return_value=MockResponse(200, content=json.dumps([])))
    create = router.post(f"{BASE}/ssh/").mock(
        return_value=MockResponse(200, content=json.dumps({"success": True, "id": 9})))
    assert sshkeys.ensure_account_key(client, PUB) is True
    assert create.called and json.loads(create.calls[0].request.content) == {"ssh_key": PUB}
    # already present (different comment) -> no POST
    listing.mock(return_value=MockResponse(
        200, content=json.dumps([{"id": 9, "public_key": PUB.rsplit(" ", 1)[0] + " x@y"}])))
    create.reset()
    assert sshkeys.ensure_account_key(client, PUB) is False
    assert not create.called


def test_attach_key_posts_and_never_raises(client, router):
    route = router.post(f"{BASE}/instances/42/ssh/").mock(
        return_value=MockResponse(200, content=json.dumps({"success": True})))
    assert sshkeys.attach_key(client, 42, PUB) is True
    assert json.loads(route.calls[0].request.content) == {"ssh_key": PUB}
    route.mock(return_value=MockResponse(404, content=json.dumps({"error": "not_found"})))
    assert sshkeys.attach_key(client, 42, PUB) is False


def test_403_is_not_reported_as_a_key_problem(client, router):
    from bbvastl80.vast.client import VastAuthError, VastHTTPError
    router.put(f"{BASE}/instances/command/1").mock(
        return_value=MockResponse(403, content=json.dumps({"error": "blacklisted"})))
    with pytest.raises(VastHTTPError, match="blacklisted"):
        client.execute(1, "ls")
    router.get(f"{BASE}/users/current/").mock(
        return_value=MockResponse(401, content="{}"))
    with pytest.raises(VastAuthError):
        client.get_user()
