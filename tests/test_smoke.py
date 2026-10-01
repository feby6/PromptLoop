from typer.testing import CliRunner

import promptloop
from promptloop.cli import app


def test_package_imports() -> None:
    assert promptloop.__version__


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert promptloop.__version__ in result.output
