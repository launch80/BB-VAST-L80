"""Runtime configuration: where secrets and binaries live.

Resolution order for the Vast API key:
  1. ``VAST_API_KEY`` environment variable
  2. ``~/.config/bbvastl80/credentials``  (first non-comment line, or JSON)

The L80 publish token is deliberately NOT read here — the L80 CLI reads it
itself from ``~/.config/L80/credentials.json`` so the token never appears in
this process's environment or in any log we print.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_API_BASE = "https://console.vast.ai/api/v0"
CRED_DIR = Path.home() / ".config" / "bbvastl80"
CRED_FILE = CRED_DIR / "credentials"


class ConfigError(Exception):
    """A configuration problem the user can fix (missing key, no binary, ...)."""


@dataclass(frozen=True)
class Config:
    api_key: str | None
    api_base: str
    l80_bin: str
    betterbench_bin: str
    runs_dir: Path

    # --- resolution -------------------------------------------------------
    @classmethod
    def load(cls) -> "Config":
        # Priority: real environment > project .env > ~/.config/bbvastl80/credentials
        api_key = os.environ.get("VAST_API_KEY")
        if not api_key:
            api_key = _read_dotenv("VAST_API_KEY")
        if not api_key:
            api_key = _read_credentials_file()
        api_base = os.environ.get("VAST_API_BASE")
        if not api_base:
            api_base = _read_dotenv("VAST_API_BASE")
        return cls(
            api_key=api_key,
            api_base=(api_base or DEFAULT_API_BASE).rstrip("/"),
            l80_bin=os.environ.get("L80_BIN", "l80"),
            betterbench_bin=os.environ.get(
                "L80_BETTERBENCH",
                str(_which("betterbench") or "betterbench"),
            ),
            runs_dir=Path(os.environ.get("BBVASTL80_RUNS_DIR", "./runs")),
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                "No Vast.ai API key found.\n"
                "  1. Create one at https://cloud.vast.ai/manage-keys/\n"
                "  2. export VAST_API_KEY=<key>   (or put it in .env)\n"
                f"     — or write it to {CRED_FILE}"
            )
        return self.api_key

    def check_binaries(self) -> dict[str, str | None]:
        """Locate the external tools we shell out to. Missing => None."""
        return {
            "l80": _which(self.l80_bin),
            "betterbench": _which(self.betterbench_bin)
            if Path(self.betterbench_bin).is_file()
            else _which("betterbench"),
            "ssh": _which("ssh"),
        }


def _read_dotenv(key: str) -> str | None:
    """Read a key from a ``.env`` file in the CWD (minimal parser, no dep)."""
    candidate = Path.cwd() / ".env"
    if not candidate.is_file():
        return None
    for line in candidate.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'") or None
    return None


def _read_credentials_file() -> str | None:
    if not CRED_FILE.exists():
        return None
    text = CRED_FILE.read_text().strip()
    if not text:
        return None
    if text.startswith("{"):  # JSON form: {"api_key": "..."}
        import json

        try:
            return str(json.loads(text).get("api_key", "")).strip() or None
        except json.JSONDecodeError:
            return None
    # plain-text form: first non-comment, non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def _which(name: str) -> str | None:
    return shutil.which(name)
