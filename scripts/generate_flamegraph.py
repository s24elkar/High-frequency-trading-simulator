#!/usr/bin/env python3

"""
Minimal flame graph generator for folded stack data.

Usage:
    python scripts/generate_flamegraph.py --input results/perf/profile.folded --output results/perf/profile.svg
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


FRAME_HEIGHT = 18
FRAME_GAP = 2
HORIZONTAL_PADDING = 2
TEXT_PADDING = 4
FONT_SIZE = 12
DEFAULT_WIDTH = 1200.0


@dataclass
class FlameNode:
    name: str
    value: float = 0.0
    children: Dict[str, "FlameNode"] = field(default_factory=dict)

    def add_sample(self, frames: Iterable[str], weight: float) -> None:
        self.value += weight
        iterator = iter(frames)
        try:
            frame = next(iterator)
        except StopIteration:
            return
        child = self.children.get(frame)
        if child is None:
            child = FlameNode(frame)
            self.children[frame] = child
        child.add_sample(iterator, weight)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children.values())


def parse_folded(path: Path) -> FlameNode:
    root = FlameNode("root")
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                stack_part, weight_part = line.rsplit(" ", 1)
                weight = float(weight_part)
            except ValueError as exc:
                raise ValueError(f"Malformed line in {path}: {raw_line!r}") from exc
            frames = tuple(filter(None, (frame.strip() for frame in stack_part.split(";"))))
            if not frames:
                frames = ("(root)",)
            root.add_sample(frames, weight)
    return root


def color_for(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    r = 180 + digest[0] % 55
    g = 120 + digest[1] % 95
    b = 90 + digest[2] % 120
    return f"#{r:02x}{g:02x}{b:02x}"


def render_svg(root: FlameNode, width: float, height: float) -> str:
    elements: List[str] = []
    elements.append(
        f'<rect x="0" y="0" width="{width:.2f}" height="{height:.2f}" fill="#ffffff"/>'
    )
    scale = (width - 2 * HORIZONTAL_PADDING) / root.value if root.value else 1.0

    def emit_node(node: FlameNode, depth: int, x_offset: float) -> float:
        node_width = max(node.value * scale, 0.5)
        y = height - (depth + 1) * (FRAME_HEIGHT + FRAME_GAP)
        color = color_for(node.name)
        elements.append(
            f'<g class="frame">'
            f'<rect x="{x_offset:.2f}" y="{y:.2f}" width="{node_width:.2f}" height="{FRAME_HEIGHT:.2f}" '
            f'rx="2" ry="2" fill="{color}" stroke="#333333" stroke-width="0.5">'
            f'<title>{node.name} ({node.value:.2f} us)</title></rect>'
        )
        label = node.name
        max_chars = max(int((node_width - 2 * TEXT_PADDING) / (FONT_SIZE * 0.6)), 0)
        if max_chars > 3:
            if len(label) > max_chars:
                label = label[: max_chars - 3] + "..."
            text_x = x_offset + TEXT_PADDING
            text_y = y + FRAME_HEIGHT - 4
            elements.append(
                f'<text x="{text_x:.2f}" y="{text_y:.2f}" font-family="Helvetica,Arial,sans-serif" '
                f'font-size="{FONT_SIZE}" fill="#000000">{label}</text>'
            )
        elements.append("</g>")

        child_x = x_offset
        for child in sorted(node.children.values(), key=lambda c: c.value, reverse=True):
            consumed = emit_node(child, depth + 1, child_x)
            child_x += consumed
        return node_width

    x_cursor = HORIZONTAL_PADDING
    for child in sorted(root.children.values(), key=lambda c: c.value, reverse=True):
        consumed = emit_node(child, 0, x_cursor)
        x_cursor += consumed

    return (
        "<?xml version=\"1.0\" standalone=\"no\"?>\n"
        f'<svg version="1.1" width="{width:.2f}" height="{height:.2f}" '
        f'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )


def build_svg(root: FlameNode, width: float) -> str:
    levels = max(root.depth() - 1, 1)
    height = levels * (FRAME_HEIGHT + FRAME_GAP) + FRAME_GAP
    return render_svg(root, width, height)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a flame graph SVG from folded stacks.")
    parser.add_argument("--input", required=True, type=Path, help="Path to folded stack input")
    parser.add_argument("--output", required=True, type=Path, help="Path for the SVG output")
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH, help="Width of the SVG in px")
    args = parser.parse_args()

    root = parse_folded(args.input)
    svg = build_svg(root, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
