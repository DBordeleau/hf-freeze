"""Command-line interface for hf-freeze."""

from pathlib import Path

import typer

from hf_freeze import __version__
from hf_freeze.models import DependencyFinding, ScanDiagnostic
from hf_freeze.scan import scan_path

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Freeze Hugging Face Hub dependencies."""


@app.command()
def version() -> None:
    """Show the installed hf-freeze version."""
    typer.echo(f"hf-freeze {__version__}")


@app.command()
def scan(path: Path = typer.Argument(Path("."))) -> None:
    """Discover supported Hugging Face Hub calls in Python source."""
    try:
        result = scan_path(path)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PATH") from error

    entries: list[DependencyFinding | ScanDiagnostic] = [
        *result.findings,
        *result.diagnostics,
    ]
    entries.sort(
        key=lambda item: (item.source.path, item.source.line, item.source.column)
    )
    for entry in entries:
        location = f"{entry.source.path}:{entry.source.line}:{entry.source.column + 1}"
        if isinstance(entry, ScanDiagnostic):
            typer.echo(f"{location}  {entry.message}")
        elif entry.repo_id is None:
            typer.echo(
                f"{location}  {entry.call_kind.value}  unresolved: "
                f"{entry.unresolved_reason}"
            )
        else:
            repo_type = entry.repo_type.value if entry.repo_type else "unknown"
            revision = entry.requested_revision or "<default>"
            typer.echo(
                f"{location}  {entry.call_kind.value}  {repo_type}  "
                f"{entry.repo_id}  revision={revision}"
            )
