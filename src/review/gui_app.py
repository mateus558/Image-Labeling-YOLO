"""Tkinter GUI for reviewing and editing YOLO bounding boxes."""
from __future__ import annotations

from pathlib import Path
from typing import List

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image

from .io import discover_images, read_yolo_labels, write_yolo_labels
from .models import LabelBox
from .constants import (
    CANVAS_MAX_WIDTH,
    CANVAS_MAX_HEIGHT,
    COLOR_PREVIEW,
    BOX_LINE_WIDTH,
    PREVIEW_DASH,
    MIN_BOX_SIZE_NORM,
)
from . import canvas_utils
from .view import LabelReviewView
from .controller import LabelReviewController


class LabelReviewApp:
    def __init__(self, image_dir: Path, label_dir: Path) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        # Defensive: don't raise if default folders are missing; show empty state instead
        try:
            if self.image_dir is None or not self.image_dir.exists() or not self.image_dir.is_dir():
                self.image_paths = []
            else:
                self.image_paths = discover_images(self.image_dir)
        except Exception:
            self.image_paths = []

        self.index = 0
        self.boxes: List[LabelBox] = []
        self.add_mode = False
        self.add_start: tuple[float, float] | None = None
        self.dirty = False
        self.resize_target = None
        self.resize_start_box = None

        # Build view and wire callbacks
        self.view = LabelReviewView()
        self.root = self.view.root
        self.canvas = self.view.canvas
        self.listbox = self.view.listbox
        self.status = self.view.status

        # Build controller and wire events
        self.controller = LabelReviewController(self.image_dir, self.label_dir, self.view)
        self.view.bind_navigation(self.controller.prev_image, self.controller.next_image)
        self.view.bind_editing(self.controller.start_add_box, self.controller.delete_selected_box, self.controller.save_labels)
        self.view.bind_canvas(
            self.controller.on_canvas_press,
            self.controller.on_canvas_drag,
            self.controller.on_canvas_release,
            self.controller.on_select_list,
            on_motion=self.controller.on_canvas_motion,
        )
        # Redraw boxes on canvas resize to follow centering offsets
        self.view.bind_canvas_resize(self.controller.draw_boxes)
        self.view.bind_class_change(self.controller.on_class_change)
        self.view.bind_directory_select(self.controller.select_image_dir, self.controller.select_label_dir)
        self.root.bind("<Escape>", lambda _event: self.controller.cancel_add_or_resize())
        # Quick class hotkeys: number keys 0-9 set current class or apply to selection
        for k in list("0123456789"):
            self.root.bind(k, self._on_digit_class)
            self.root.bind(k.upper(), self._on_digit_class)
        self.root.bind("n", lambda _event: self.next_image())
        self.root.bind("N", lambda _event: self.next_image())
        self.root.bind("p", lambda _event: self.prev_image())
        self.root.bind("P", lambda _event: self.prev_image())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Initial load
        self.controller.load_current()

    # Backward-compat methods retained for external imports; delegate to controller where possible
    def _label_path(self, image_path: Path) -> Path:
        return self.controller._label_path(image_path)

    def _load_labels(self, image_path: Path) -> List[LabelBox]:
        return read_yolo_labels(self._label_path(image_path))

    def _load_image(self, image_path: Path):
        return Image.open(image_path).convert("RGB")

    def _render_image(self, image):
        self.controller.display_width, self.controller.display_height = self.view.render_image_fit(image)
        self.controller.draw_boxes()

    def _draw_boxes(self) -> None:
        self.controller.draw_boxes()

    def _normalized_to_canvas(self, box: LabelBox) -> tuple[float, float, float, float]:
        return canvas_utils.normalized_to_canvas(box, self.controller.display_width, self.controller.display_height)

    def _canvas_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        return self.controller._canvas_to_normalized(x, y)

    def _selected_indices(self) -> List[int]:
        return [int(idx) for idx in self.listbox.curselection()]

    def _on_select_box(self, _event: tk.Event) -> None:
        self.controller.on_select_list(_event)

    def _clear_preview(self) -> None:
        if self._preview_rect is not None:
            self.canvas.delete(self._preview_rect)
            self._preview_rect = None

    def start_add_box(self) -> None:
        self.controller.start_add_box()

    def _on_canvas_press(self, event: tk.Event) -> None:
        self.controller.on_canvas_press(event)

    def _on_canvas_drag(self, event: tk.Event) -> None:
        self.controller.on_canvas_drag(event)

    def _on_canvas_release(self, event: tk.Event) -> None:
        self.controller.on_canvas_release(event)

    def _find_box_at(self, x: float, y: float) -> int | None:
        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self._normalized_to_canvas(box)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return None

    def delete_selected_box(self) -> None:
        self.controller.delete_selected_box()

    def _cancel_add_mode(self) -> None:
        if self.add_mode:
            self.add_mode = False
            self.add_start = None
            self._clear_preview()
            self.status.configure(text="Add cancelled.")
        elif self.resize_target is not None:
            self._cancel_resize()

    def save_labels(self) -> None:
        self.controller.save_labels()

    def _refresh_listbox(self) -> None:
        # Kept for backward-compat calls inside this class; delegate to controller
        self.controller._refresh_list()

    def load_current_image(self) -> None:
        self.controller.load_current()

    def prev_image(self) -> None:
        self.controller.prev_image()

    def next_image(self) -> None:
        self.controller.next_image()

    def run(self) -> None:
        self.root.mainloop()

    def _on_close(self) -> None:
        self.controller.save_labels()
        self.root.destroy()

    def _set_dirty(self, value: bool) -> None:
        # Kept for compatibility; controller owns dirty state
        self.controller._set_dirty(value)

    def _update_title(self) -> None:
        self.controller._update_title()

    def _detect_handle(self) -> tuple[int, str] | None:
        return canvas_utils.detect_handle(self.canvas)

    def _start_resize(self, idx: int, corner: str) -> None:
        self.controller._start_resize(idx, corner)

    def _update_resize(self, canvas_x: float, canvas_y: float) -> None:
        self.controller._update_resize(canvas_x, canvas_y)

    def _finish_resize(self) -> None:
        self.controller._finish_resize()

    def _cancel_resize(self) -> None:
        self.controller._cancel_resize()

    def _clear_resize_state(self) -> None:
        self.controller._clear_resize_state()

    def _update_listbox_entry(self, idx: int) -> None:
        self.controller._update_listbox_entry(idx)

    def _box_corners_norm(self, box: LabelBox) -> tuple[float, float, float, float]:
        return canvas_utils.box_corners_norm(box)

    def _format_box_summary(self, idx: int, box: LabelBox) -> str:
        return f"#{idx+1}: x={box.x_center:.2f} y={box.y_center:.2f} w={box.width:.2f} h={box.height:.2f}"

    def _draw_handles(self, idx: int, x1: float, y1: float, x2: float, y2: float) -> None:
        canvas_utils.draw_handles(self.canvas, idx, x1, y1, x2, y2)

    # Local key handler to route digit keys for class changes
    def _on_digit_class(self, event: tk.Event) -> None:
        try:
            digit = int(event.keysym)  # keysym is '0'..'9'
        except Exception:
            return
        self.controller.on_class_change(digit)
