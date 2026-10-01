"""Prompt templates used by the optimiser itself.

Templates are `string.Template` files (`$name` placeholders) rather than `str.format`,
because they contain literal JSON braces that would otherwise need escaping.
"""

from importlib.resources import files
from string import Template


def render(name: str, **values: object) -> str:
    """Fill `<name>.txt` with `values`. Missing placeholders raise KeyError."""
    text = (files(__package__) / f"{name}.txt").read_text(encoding="utf-8")
    return Template(text).substitute({k: str(v) for k, v in values.items()})
