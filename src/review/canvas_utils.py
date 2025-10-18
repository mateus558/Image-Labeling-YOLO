"""Canvas drawing and geometry utilities for the review GUI.

These helpers are pure or only depend on a tkinter Canvas and can be unit-tested
without the full application wiring.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .constants import (
    BOX_LINE_WIDTH,
    CLASS_COLOR_PALETTE,
    COLOR_BOX,
    COLOR_BOX_SELECTED,
    COLOR_LABEL_BG,
    COLOR_LABEL_TEXT,
    HANDLE_SIZE,
)
from .models import LabelBox


def normalized_to_canvas(
    box: LabelBox, display_width: int, display_height: int, x_offset: int = 0, y_offset: int = 0
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
    return x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset


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
    # Edge handles for horizontal / vertical resize
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    edges = {
        "n": (mid_x, y1),
        "s": (mid_x, y2),
        "w": (x1, mid_y),
        "e": (x2, mid_y),
    }
    for name, (hx, hy) in edges.items():
        canvas.create_rectangle(
            hx - half,
            hy - half,
            hx + half,
            hy + half,
            fill=COLOR_BOX_SELECTED,
            outline="#202020",
            tags=("handle", f"handle-{idx}-{name}"),
        )


def draw_boxes(
    canvas,
    boxes: Sequence[LabelBox],
    selected_indices: Iterable[int],
    display_width: int,
    display_height: int,
    class_names: Sequence[str] | None = None,
    x_offset: int = 0,
    y_offset: int = 0,
) -> None:
    selected = set(int(i) for i in selected_indices)
    canvas.delete("box")
    canvas.delete("handle")
    canvas.delete("label")
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = normalized_to_canvas(box, display_width, display_height, x_offset, y_offset)
        if idx in selected:
            color = COLOR_BOX_SELECTED
        else:
            try:
                palette_color = CLASS_COLOR_PALETTE[box.class_id % len(CLASS_COLOR_PALETTE)]
            except Exception:
                palette_color = COLOR_BOX
            color = palette_color
        canvas.create_rectangle(
            x1, y1, x2, y2, outline=color, width=BOX_LINE_WIDTH, tags=("box", f"box-{idx}")
        )
        # Draw class label above the box (if provided)
        label_text = None
        if class_names is not None and 0 <= box.class_id < len(class_names):
            label_text = class_names[box.class_id]
        else:
            label_text = str(box.class_id)
        # Background rectangle behind text for readability
        # Measure approximate width; tkinter doesn't provide pre-measure easily without font metrics
        # so we pad a bit around the text.
        try:
            text_id = canvas.create_text(
                x1 + 4,
                max(0, y1 - 10),
                anchor="sw",
                text=label_text,
                fill=COLOR_LABEL_TEXT,
                tags=("label", f"label-{idx}"),
            )
            bbox = canvas.bbox(text_id)
            if bbox is not None:
                lx1, ly1, lx2, ly2 = bbox
                pad = 2
                bg = canvas.create_rectangle(
                    lx1 - pad,
                    ly1 - pad,
                    lx2 + pad,
                    ly2 + pad,
                    fill=COLOR_LABEL_BG,
                    outline="",
                    tags=("label", f"label-bg-{idx}"),
                )
                canvas.tag_lower(bg, text_id)
        except Exception:
            # If drawing text fails for any reason, skip gracefully
            pass
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
