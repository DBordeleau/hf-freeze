"""Command-line interface for hf-freeze."""

import typer

from hf_freeze import __version__

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Freeze Hugging Face Hub dependencies."""


@app.command()
def version() -> None:
    """Show the installed hf-freeze version."""
    typer.echo(f"hf-freeze {__version__}")
