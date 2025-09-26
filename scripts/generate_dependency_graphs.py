#!/usr/bin/env python3
"""Generate dependency graphs for the project.

Outputs:
- docs/graphs/cmake_graph.dot      : CMake target dependency graph
- docs/graphs/python_imports.dot   : Python package import graph

The script intentionally keeps dependencies minimal and relies on Graphviz DOT
for downstream rendering (e.g., `dot -Tpng file.dot -o file.png`).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
DOC_GRAPH_DIR = ROOT / "docs" / "graphs"
PYTHON_DIR = ROOT / "python"


def _log(msg: str) -> None:
    print(f"[graph] {msg}")


def _run_cmake_graphviz() -> None:
    build_dir = ROOT / "build"
    output = DOC_GRAPH_DIR / "cmake_graph.dot"
    if not build_dir.exists():
        _log("Skipping CMake graph (build/ directory missing). Run `cmake -S . -B build` first.")
        return

    cmd = ["cmake", f"--graphviz={output}", str(build_dir)]
    _log("Running: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
        _log(f"Wrote {output.relative_to(ROOT)}")
    except subprocess.CalledProcessError as exc:
        _log(f"Failed to generate CMake graph: {exc}")


def _module_name(path: Path) -> str:
    rel = path.relative_to(PYTHON_DIR).with_suffix("")
    return ".".join(rel.parts)


def _iter_python_files() -> Iterable[Path]:
    for p in PYTHON_DIR.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def _resolve_import(module: str, level: int, current: str) -> str | None:
    if level == 0:
        return module

    current_parts = current.split(".")
    if level > len(current_parts):
        return module or None
    base = current_parts[: len(current_parts) - level]
    if module:
        base.append(module)
    return ".".join(part for part in base if part)


def _build_python_graph() -> Tuple[Set[str], Set[Tuple[str, str]]]:
    nodes: Set[str] = set()
    edges: Set[Tuple[str, str]] = set()

    for file in _iter_python_files():
        module = _module_name(file)
        nodes.add(module)
        try:
            tree = ast.parse(file.read_text(), filename=str(file))
        except SyntaxError as exc:
            _log(f"Skipping {file}: {exc}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    edges.add((module, target))
            elif isinstance(node, ast.ImportFrom):
                target = _resolve_import(node.module or "", node.level or 0, module)
                if target:
                    edges.add((module, target))

    # Filter to project-local modules when applicable
    project_namespace = _module_name(PYTHON_DIR / "__init__.py" if (PYTHON_DIR / "__init__.py").exists() else PYTHON_DIR)
    filtered_edges: Set[Tuple[str, str]] = set()
    for src, dst in edges:
        if dst.startswith(project_namespace) or dst.split(".")[0] in {"numpy", "matplotlib", "torch", "streamlit"}:
            filtered_edges.add((src, dst))
    return nodes, filtered_edges


def _emit_python_dot(nodes: Set[str], edges: Set[Tuple[str, str]]) -> None:
    output = DOC_GRAPH_DIR / "python_imports.dot"
    lines = ["digraph python_imports {"]
    lines.append("  rankdir=LR;")
    for node in sorted(nodes):
        label = node.split(".")[-1]
        lines.append(f"  \"{node}\" [label=\"{label}\"];\n")
    for src, dst in sorted(edges):
        lines.append(f"  \"{src}\" -> \"{dst}\";\n")
    lines.append("}\n")
    output.write_text("".join(lines))
    _log(f"Wrote {output.relative_to(ROOT)}")


def main() -> int:
    DOC_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    _run_cmake_graphviz()
    nodes, edges = _build_python_graph()
    _emit_python_dot(nodes, edges)
    _log("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
