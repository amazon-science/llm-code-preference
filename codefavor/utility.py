# SPDX-FileCopyrightText: (c) 2024 Amazon.com, Inc. or its affiliates
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import ast

import astor
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


def make_progress():
    return Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )


def remove_comments(code: str):
    # Parse the code into an AST
    tree = ast.parse(code)

    # Remove all comments and docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            node.body = [
                n
                for n in node.body
                if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)
            ]
            node.docstring = None

    # Generate code from the modified AST
    return astor.to_source(tree)
