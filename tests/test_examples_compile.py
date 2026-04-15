import ast
from pathlib import Path


def test_examples_compile():
    for path in Path("examples").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
