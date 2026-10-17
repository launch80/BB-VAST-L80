from __future__ import annotations

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
