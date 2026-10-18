from __future__ import annotations

from bbvastl80 import config as config_mod
from bbvastl80.config import Config
from bbvastl80.state import RunState, StateStore


def test_config_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "secret-key-123")
    cfg = Config.load()
    assert cfg.api_key == "secret-key-123"
    assert cfg.require_api_key() == "secret-key-123"


def test_config_missing_key_raises(monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    cfg = Config.load()
    if cfg.api_key is None:  # ensure file-based key not picked up in CI
        import pytest
        from bbvastl80.config import ConfigError
        with pytest.raises(ConfigError):
            cfg.require_api_key()


def test_credentials_file_accepts_bare_key_or_assignment(monkeypatch, tmp_path):
    cred = tmp_path / "credentials"
    monkeypatch.setattr(config_mod, "CRED_FILE", cred)
    for text in ("abc123\n", "# comment\napi_key=abc123\n", 'VAST_API_KEY="abc123"\n'):
        cred.write_text(text)
        assert config_mod._read_credentials_file() == "abc123"


def test_l80_bin_prefers_uppercase(monkeypatch):
    monkeypatch.delenv("L80_BIN", raising=False)
    found = {"L80": "/bin/L80", "l80": "/bin/l80"}
    monkeypatch.setattr(config_mod, "_which", lambda n: found.get(n))
    assert Config.load().l80_bin == "L80"
    found.pop("L80")
    assert Config.load().l80_bin == "l80"
    monkeypatch.setenv("L80_BIN", "/opt/L80")
    assert Config.load().l80_bin == "/opt/L80"


def test_state_roundtrip(tmp_path):
    store = StateStore(tmp_path)
    rid = store.new_run_id("my-run")
    assert rid.startswith("my-run-")
    st = RunState(run_id=rid, status="rented", instance_id=555, public_ip="1.2.3.4",
                  ports={"8000": 44321})
    store.save(st)
    loaded = store.load(rid)
    assert loaded.instance_id == 555
    assert loaded.ports == {"8000": 44321}
    assert rid in store.list_run_ids()


def test_state_missing_raises(tmp_path):
    store = StateStore(tmp_path)
    import pytest
    with pytest.raises(FileNotFoundError):
        store.load("nope-1")


def test_client_derives_v1_base_for_instances(tmp_path):
    from bbvastl80.vast.client import VastClient
    c = VastClient("k")
    assert c.base_url == "https://console.vast.ai/api/v0"
    assert c.v1_base_url == "https://console.vast.ai/api/v1"
    c.close()


def test_betterbench_version_gate():
    from bbvastl80.cli import _version_at_least

    assert _version_at_least("BetterBench 0.6.1", "0.5.0")
    assert _version_at_least("betterbench 0.5.0", "0.5.0")
    assert not _version_at_least("BetterBench 0.4.0 · x", "0.5.0")
    assert not _version_at_least("unknown", "0.5.0")
