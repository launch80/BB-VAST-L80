"""M5b: publish a BetterBench results.json through the L80 CLI.

``l80 betterbench --results <f> [--template …] [--title …] [--summary …]
[--section "H=B"]… [--dry-run] --json`` does the work; L80 reads its own token
from ``~/.config/L80/credentials.json`` (never this process) and strips
``env.endpoint``/``env.host`` before upload. We parse the JSON it emits for the
page ``url`` (falling back to the first launch80 URL in the output).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from bbvastl80 import log

_URL = re.compile(r"https?://[^\s\"']*launch80\.com/[^\s\"']+")


class PublishError(Exception):
    """L80 refused, failed, or printed nothing we could read a URL from."""


def publish_cmd(l80_bin: str, spec, results: Path, *, dry_run: bool | None = None) -> list[str]:
    p = spec.publish
    cmd = [l80_bin, "betterbench", "--results", str(results), "--template", p.template, "--json"]
    if p.title:
        cmd += ["--title", p.title]
    if p.summary:
        cmd += ["--summary", p.summary]
    for s in p.sections or []:
        cmd += ["--section", s]
    if (p.dry_run if dry_run is None else dry_run):
        cmd.append("--dry-run")
    return cmd


def run_publish(l80_bin: str, spec, results: Path, *, dry_run: bool | None = None,
                timeout_s: float = 300) -> str | None:
    """Publish; return the page URL (None on a dry run)."""
    cmd = publish_cmd(l80_bin, spec, results, dry_run=dry_run)
    is_dry = "--dry-run" in cmd
    log.info(("validating with L80 (dry run): " if is_dry else "publishing with L80 (public): ")
             + " ".join(cmd))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        raise PublishError(f"l80 not found at {l80_bin!r} (set L80_BIN)") from None
    except subprocess.TimeoutExpired:
        raise PublishError(f"l80 exceeded {timeout_s:.0f}s") from None
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise PublishError(f"l80 exited {p.returncode}:\n{out.strip()[:800]}")
    url = parse_url(out)
    if is_dry:
        log.success("L80 dry run OK" + (f" ({url})" if url else ""))
        return None
    if not url:
        raise PublishError(f"l80 succeeded but printed no URL:\n{out.strip()[:800]}")
    return url


def parse_url(out: str) -> str | None:
    """The page URL from L80's ``--json`` output (or any launch80 URL in it)."""
    for line in reversed([ln for ln in out.splitlines() if ln.strip()]):
        if line.lstrip().startswith("{"):
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            for key in ("url", "share_url", "page_url"):
                if isinstance(obj.get(key), str):
                    return obj[key]
            art = obj.get("artifact") if isinstance(obj.get("artifact"), dict) else None
            if art and isinstance(art.get("url"), str):
                return art["url"]
    try:
        obj = json.loads(out)
        for key in ("url", "share_url", "page_url"):
            if isinstance(obj.get(key), str):
                return obj[key]
    except ValueError:
        pass
    m = _URL.search(out)
    return m.group(0) if m else None
