"""Command-line interface for PromptLoop."""

import typer
from rich.console import Console

from promptloop import __version__

app = typer.Typer(help="PromptLoop: automatic prompt optimisation.", no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """PromptLoop: automatic prompt optimisation."""


@app.command()
def version() -> None:
    """Print the installed PromptLoop version."""
    console.print(f"promptloop {__version__}")
