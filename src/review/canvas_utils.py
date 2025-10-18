"""Canvas drawing and geometry utilities for the review GUI.

These helpers are pure or only depend on a tkinter Canvas and can be unit-tested
without the full application wiring.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .constants import (
    BOX_LINE_WIDTH,
    COLOR_BOX,
    COLOR_BOX_SELECTED,
    HANDLE_SIZE,
)
from .models import LabelBox


def normalized_to_canvas(
    box: LabelBox, display_width: int, display_height: int
) -> Tuple[float, float, float, float]:
    """Convert a normalized box to canvas-space rectangle corners (x1, y1, x2, y2)."""
    w = box.width * display_width
    h = box.height * display_height
    x_center = box.x_center * display_width
    y_center = box.y_center * display_height
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return x1, y1, x2, y2


def draw_handles(canvas, idx: int, x1: float, y1: float, x2: float, y2: float) -> None:
    half = HANDLE_SIZE / 2
    corners = {
        "nw": (x1, y1),
        "ne": (x2, y1),
        "sw": (x1, y2),
        "se": (x2, y2),
    }
    for name, (hx, hy) in corners.items():
        canvas.create_rectangle(
            hx - half,
            hy - half,
            hx + half,
            hy + half,
            fill=COLOR_BOX_SELECTED,
            outline="#202020",
            tags=("handle", f"handle-{idx}-{name}"),
        )


def draw_boxes(canvas, boxes: Sequence[LabelBox], selected_indices: Iterable[int], display_width: int, display_height: int) -> None:
    selected = set(int(i) for i in selected_indices)
    canvas.delete("box")
    canvas.delete("handle")
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = normalized_to_canvas(box, display_width, display_height)
        color = COLOR_BOX_SELECTED if idx in selected else COLOR_BOX
        canvas.create_rectangle(
            x1, y1, x2, y2, outline=color, width=BOX_LINE_WIDTH, tags=("box", f"box-{idx}")
        )
        if idx in selected:
            draw_handles(canvas, idx, x1, y1, x2, y2)
    canvas.tag_raise("handle")


def box_corners_norm(box: LabelBox) -> Tuple[float, float, float, float]:
    x1 = box.x_center - box.width / 2
    y1 = box.y_center - box.height / 2
    x2 = box.x_center + box.width / 2
    y2 = box.y_center + box.height / 2
    return x1, y1, x2, y2


def detect_handle(canvas) -> tuple[int, str] | None:
    """Return (box_index, corner) if the current canvas item is a handle.

    Expects handle tags in the form 'handle-{idx}-{corner}'.
    """
    current = canvas.find_withtag("current")
    if not current:
        return None
    tags = canvas.gettags(current[0])
    for tag in tags:
        if tag.startswith("handle-"):
            try:
                _, idx_str, corner = tag.split("-", 2)
                return int(idx_str), corner
            except ValueError:
                return None
    return None
