"""`bbvastl80 list --json`: what L80 checks its local run records against."""
import json

from typer.testing import CliRunner

from bbvastl80 import cli
from bbvastl80.vast.models import Instance


class FakeClient:
    def __init__(self, instances):
        self.instances = instances

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def list_instances(self):
        return self.instances


def test_list_json(monkeypatch):
    insts = [Instance.model_validate({"id": 7, "actual_status": "running", "label": "bbvastl80:x", "dph_total": 0.25})]
    monkeypatch.setattr(cli, "_cfg", lambda: None)
    monkeypatch.setattr(cli, "_client", lambda cfg: FakeClient(insts))
    r = CliRunner().invoke(cli.app, ["list", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout) == [{"id": 7, "status": "running", "label": "bbvastl80:x", "dph": 0.25}]
    monkeypatch.setattr(cli, "_client", lambda cfg: FakeClient([]))
    r = CliRunner().invoke(cli.app, ["list", "--json"])
    assert json.loads(r.stdout) == []
