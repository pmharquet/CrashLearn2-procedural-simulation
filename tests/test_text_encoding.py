"""Keep UTF-8 interface text intact when moving or editing source files."""

import ast
from pathlib import Path

import crashlearn_sim


def test_application_strings_are_not_mojibake():
    root = Path(crashlearn_sim.__file__).parent
    markers = ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00e2\u02c6", "\ufffd")
    for path in root.rglob("*.py"):
        if "resources" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not any(marker in node.value for marker in markers), (
                    f"Invalid text encoding in {path.name}:{node.lineno}: {node.value!r}"
                )


def test_menu_keeps_minus_infinity_and_accents():
    source = (Path(crashlearn_sim.__file__).parent / "ui/menu_view.py").read_text(encoding="utf-8")
    strings = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"\u2212", "\u221e", "\u2026", "Proc\u00e9dural"} <= strings
