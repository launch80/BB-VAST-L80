"""Minimal rich-based logging for the CLI.

Kept dependency-light and side-effect-free enough to test: every helper writes
to a single shared :class:`rich.Console`. Tests that care about output inject
their own console via :func:`set_console`.
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

_console = Console()


def set_console(console: Console) -> None:
    global _console
    _console = console


def console() -> Console:
    return _console


def info(msg: str) -> None:
    _console.print(msg)


def detail(msg: str) -> None:
    _console.print(f"[dim]{msg}[/dim]")


def success(msg: str) -> None:
    _console.print(f"[green]✓[/green] {msg}")


def warn(msg: str) -> None:
    _console.print(f"[yellow]⚠[/yellow] {msg}")


def error(msg: str) -> None:
    _console.print(f"[red]✗[/red] {msg}", highlight=False)


def table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    t = Table(title=title, title_justify="left", show_lines=False)
    for h in headers:
        t.add_column(h)
    for row in rows:
        t.add_row(*[str(c) for c in row])
    _console.print(t)
