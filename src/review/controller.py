"""Controller for the label review GUI.

This module orchestrates I/O, view updates, and label editing logic. It is a
future target for moving logic out of gui_app.LabelReviewApp incrementally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Protocol, Sequence

from PIL import Image

import tkinter as tk

from .io import discover_images, read_yolo_labels, write_yolo_labels, read_class_names_from_yaml
from .models import LabelBox
from .constants import MIN_BOX_SIZE_NORM
from . import canvas_utils


class View(Protocol):
    canvas: Any
    def render_image_fit(self, image: Image.Image) -> tuple[int, int]: ...
    def set_status(self, text: str) -> None: ...
    def set_title(self, text: str) -> None: ...
    def set_list_items(self, items: Sequence[str]) -> None: ...
    def get_selected_indices(self) -> List[int]: ...
    def set_selection(self, indices: Sequence[int]) -> None: ...
    def set_class_names(self, class_names: Sequence[str]) -> None: ...
    def set_current_class_id(self, class_id: int) -> None: ...
    def set_directory_display(self, image_dir: Path, label_dir: Path) -> None: ...
    def prompt_directory(self, title: str, initialdir: Path | None = None) -> Path | None: ...
    def start_preview_rect(self, x: float, y: float) -> None: ...
    def update_preview_rect(self, x0: float, y0: float, x1: float, y1: float) -> None: ...
    def clear_preview_rect(self) -> None: ...
    def update_crosshair(self, x: float, y: float) -> None: ...
    def clear_crosshair(self) -> None: ...
    def show_canvas_message(self, lines: Sequence[str] | str) -> None: ...


class LabelReviewController:
    def __init__(self, image_dir: Path, label_dir: Path, view: View) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.view = view
        self.image_paths = discover_images(self.image_dir)
        self.index = 0
        self.boxes: List[LabelBox] = []
        self.display_width = 1
        self.display_height = 1
        self.add_mode = False
        self.add_start: tuple[float, float] | None = None
        self.dirty = False
        self.resize_target: tuple[int, str] | None = None
        self.resize_start_box: LabelBox | None = None
        # Class handling
        self.class_names: List[str] = []
        self.current_class_id: int = 0
        self._reload_classes()
        self.view.set_directory_display(self.image_dir, self.label_dir)

    def _label_path(self, image_path: Path) -> Path:
        return self.label_dir / f"{image_path.stem}.txt"

    def load_current(self) -> None:
        if not self.image_paths:
            # Empty state: clear canvas/list and show helpful status
            try:
                self.view.canvas.delete("all")
            except Exception:
                pass
            self.view.show_canvas_message(
                [
                    "No images loaded.",
                    "Use the Folders panel to pick your image and label directories.",
                    "After an image loads, press Add Box (A) and click two corners to add a label.",
                ]
            )
            self.boxes = []
            self._update_title()
            self._refresh_list()
            self.view.set_status("No images found. Use the Folders panel to select your image and label directories.")
            return
        current = self.image_paths[self.index]
        image = Image.open(current).convert("RGB")
        self.boxes = read_yolo_labels(self._label_path(current))
        self.display_width, self.display_height = self.view.render_image_fit(image)
        self._update_title()
        self._refresh_list()
        self.view.set_status(
            "Arrows/N/P to navigate. A add, D delete, S save. 0-9 sets class. Esc cancels."
        )
        # Keep class selector synced with single selection
        self._sync_class_selector_with_selection()

    # View helpers
    def _update_title(self) -> None:
        dirty_mark = " *" if getattr(self, "dirty", False) else ""
        if not self.image_paths:
            self.view.set_title(f"Label Review - 0/0{dirty_mark}")
            return
        current = self.image_paths[self.index]
        self.view.set_title(
            f"Label Review - {current.name} ({self.index + 1}/{len(self.image_paths)}){dirty_mark}"
        )

    def _refresh_list(self) -> None:
        items = [self._format_box_summary(i, b) for i, b in enumerate(self.boxes)]
        self.view.set_list_items(items)
        self.draw_boxes()

    def draw_boxes(self) -> None:
        canvas_utils.draw_boxes(
            self.view.canvas,
            self.boxes,
            self.view.get_selected_indices(),
            self.display_width,
            self.display_height,
            self.class_names,
            getattr(self.view, "image_x_offset", 0),
            getattr(self.view, "image_y_offset", 0),
        )

    # Formatting
    def _format_box_summary(self, idx: int, box: LabelBox) -> str:
        cname = self._class_name(box.class_id)
        return (
            f"#{idx+1} [{box.class_id}:{cname}] x={box.x_center:.2f} y={box.y_center:.2f} "
            f"w={box.width:.2f} h={box.height:.2f}"
        )

    # Navigation
    def prev_image(self) -> None:
        if not self.image_paths:
            self.view.set_status("No images to navigate.")
            return
        self.save_labels()
        self.index = (self.index - 1) % len(self.image_paths)
        self.load_current()

    def next_image(self) -> None:
        if not self.image_paths:
            self.view.set_status("No images to navigate.")
            return
        self.save_labels()
        self.index = (self.index + 1) % len(self.image_paths)
        self.load_current()

    # Selection
    def on_select_list(self, _event: tk.Event) -> None:
        self.draw_boxes()
        self._sync_class_selector_with_selection()

    # Add/delete/save
    def start_add_box(self) -> None:
        if not self.image_paths:
            self.view.set_status("No image loaded. Select an image directory first.")
            return
        self.add_mode = True
        self.add_start = None
        self.view.clear_preview_rect()
        self.view.set_selection([])
        self.view.set_status("Add mode: click the first corner, then click the opposite corner. (Esc to cancel)")

    def cancel_add_or_resize(self) -> None:
        # Cancel any in-progress add/resize interaction
        if self.add_mode:
            self.add_mode = False
            self.add_start = None
            self.view.clear_preview_rect()
            self.view.set_status("Add cancelled.")
            return
        if self.resize_target is not None:
            self._cancel_resize()
            return

    def delete_selected_box(self) -> None:
        if self.resize_target is not None:
            self._cancel_resize()
        selected = self.view.get_selected_indices()
        if not selected:
            # Keep behavior consistent: use status rather than dialog here (dialogs can block in headless)
            self.view.set_status("No selection to delete.")
            return
        for idx in sorted(selected, reverse=True):
            del self.boxes[idx]
        self._refresh_list()
        if selected:
            self._set_dirty(True)
        noun = "box" if len(selected) == 1 else "boxes"
        self.view.set_status(f"Deleted {len(selected)} {noun}.")

    def save_labels(self) -> None:
        if not self.image_paths:
            self.view.set_status("Nothing to save.")
            return
        label_path = self._label_path(self.image_paths[self.index])
        if self.boxes:
            write_yolo_labels(label_path, self.boxes)
        elif label_path.exists():
            label_path.unlink()
        self._set_dirty(False)
        self.view.set_status(f"Saved {label_path.name}.")

    def select_image_dir(self) -> None:
        new_dir = self.view.prompt_directory("Select image folder", self.image_dir)
        if new_dir is None:
            return
        suggested_labels = self._suggest_label_dir(new_dir)
        self._switch_to_dataset(new_dir, suggested_labels)

    def select_label_dir(self) -> None:
        new_dir = self.view.prompt_directory("Select label folder", self.label_dir)
        if new_dir is None:
            return
        if self.dirty:
            self.save_labels()
        self.label_dir = new_dir
        self.view.set_directory_display(self.image_dir, self.label_dir)
        self._reload_classes()
        self._set_dirty(False)
        self.load_current()

    # Canvas events
    def on_canvas_press(self, event: tk.Event) -> None:
        if self.add_mode:
            if self.add_start is None:
                self.add_start = (event.x, event.y)
                self.view.clear_preview_rect()
                self.view.start_preview_rect(event.x, event.y)
                self.view.set_status("First corner set. Click another point to finish.")
                return
            # Second click finishes the box (two-click workflow)
            if self._commit_add_box(event.x, event.y):
                return
            # If commit failed (box too small), treat this click as a new start
            self.add_start = (event.x, event.y)
            self.view.clear_preview_rect()
            self.view.start_preview_rect(event.x, event.y)
            self.view.set_status("First corner set. Click another point to finish.")
            return
        handle = canvas_utils.detect_handle(self.view.canvas)
        if handle is not None:
            idx, corner = handle
            self._start_resize(idx, corner)
            return
        idx = self._find_box_at(event.x, event.y)
        if idx is not None:
            try:
                state = int(event.state)
            except (TypeError, ValueError):
                state = 0
            ctrl = bool(state & 0x0004)
            shift = bool(state & 0x0001)
            if shift:
                current = self.view.get_selected_indices()
                anchor = current[0] if current else idx
                start = min(anchor, idx)
                end = max(anchor, idx)
                self.view.set_selection(list(range(start, end + 1)))
            elif ctrl:
                current = set(self.view.get_selected_indices())
                if idx in current:
                    current.discard(idx)
                else:
                    current.add(idx)
                self.view.set_selection(sorted(current))
            else:
                self.view.set_selection([idx])
            self.draw_boxes()
        else:
            # Clicked empty canvas while not in add mode; remind user how to draw
            if self.image_paths:
                self.view.set_status("Press Add Box or hit A before drawing a new box.")

    def on_canvas_drag(self, event: tk.Event) -> None:
        if self.resize_target is not None:
            self._update_resize(event.x, event.y)
            return
        if not self.add_mode or self.add_start is None:
            return
        self.view.update_preview_rect(self.add_start[0], self.add_start[1], event.x, event.y)

    def on_canvas_release(self, event: tk.Event) -> None:
        if self.resize_target is not None:
            self._finish_resize()
            return
        if self.add_mode:
            return
        if self.add_start is None:
            return
        self._commit_add_box(event.x, event.y)

    def on_canvas_motion(self, event: tk.Event) -> None:
        # Update crosshair lines to follow cursor
        try:
            self.view.update_crosshair(event.x, event.y)
        except Exception:
            pass
        if self.add_mode and self.add_start is not None:
            self.view.update_preview_rect(self.add_start[0], self.add_start[1], event.x, event.y)

    # Geometry/select helpers
    def _canvas_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        # Adjust for centering offsets if present
        x_adj = x - getattr(self.view, "image_x_offset", 0)
        y_adj = y - getattr(self.view, "image_y_offset", 0)
        return x_adj / max(1, self.display_width), y_adj / max(1, self.display_height)

    def _find_box_at(self, x: float, y: float) -> int | None:
        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = canvas_utils.normalized_to_canvas(
                box,
                self.display_width,
                self.display_height,
                getattr(self.view, "image_x_offset", 0),
                getattr(self.view, "image_y_offset", 0),
            )
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return None

    def _commit_add_box(self, end_x: float, end_y: float) -> bool:
        if self.add_start is None:
            return False
        start_x, start_y = self.add_start
        self.view.clear_preview_rect()
        start_norm = self._canvas_to_normalized(start_x, start_y)
        end_norm = self._canvas_to_normalized(end_x, end_y)
        x0n = min(max(start_norm[0], 0.0), 1.0)
        y0n = min(max(start_norm[1], 0.0), 1.0)
        x1n = min(max(end_norm[0], 0.0), 1.0)
        y1n = min(max(end_norm[1], 0.0), 1.0)
        width = abs(x1n - x0n)
        height = abs(y1n - y0n)
        if width < MIN_BOX_SIZE_NORM or height < MIN_BOX_SIZE_NORM:
            self.view.set_status("Box too small; pick points farther apart.")
            self.add_start = None
            return False
        x_center = (x0n + x1n) / 2
        y_center = (y0n + y1n) / 2
        new_box = LabelBox(
            class_id=self.current_class_id,
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
        ).clamp()
        self.boxes.append(new_box)
        self._refresh_list()
        self.add_mode = False
        self.add_start = None
        self._set_dirty(True)
        self.view.set_status("Box added. Click Add Box to draw another.")
        return True

    # Resize workflow
    def _start_resize(self, idx: int, corner: str) -> None:
        if idx < 0 or idx >= len(self.boxes):
            return
        self.resize_target = (idx, corner)
        box = self.boxes[idx]
        self.resize_start_box = LabelBox(
            class_id=box.class_id,
            x_center=box.x_center,
            y_center=box.y_center,
            width=box.width,
            height=box.height,
        )
        self.view.set_status(f"Resizing box #{idx + 1}; drag to adjust, release to commit.")

    def _update_resize(self, canvas_x: float, canvas_y: float) -> None:
        if self.resize_target is None or self.resize_start_box is None:
            return
        idx, corner = self.resize_target
        if idx < 0 or idx >= len(self.boxes):
            return
        start_box = self.resize_start_box
        x1_init, y1_init, x2_init, y2_init = canvas_utils.box_corners_norm(start_box)
        nx, ny = self._canvas_to_normalized(canvas_x, canvas_y)
        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)
        # Corner handles
        if corner == "nw":
            new_x1, new_y1, new_x2, new_y2 = nx, ny, x2_init, y2_init
        elif corner == "ne":
            new_x1, new_y1, new_x2, new_y2 = x1_init, ny, nx, y2_init
        elif corner == "sw":
            new_x1, new_y1, new_x2, new_y2 = nx, y1_init, x2_init, ny
        elif corner == "se":
            new_x1, new_y1, new_x2, new_y2 = x1_init, y1_init, nx, ny
        # Edge handles: north/south adjust only y; west/east adjust only x
        elif corner == "n":
            new_x1, new_x2 = x1_init, x2_init
            new_y1, new_y2 = ny, y2_init
        elif corner == "s":
            new_x1, new_x2 = x1_init, x2_init
            new_y1, new_y2 = y1_init, ny
        elif corner == "w":
            new_x1, new_x2 = nx, x2_init
            new_y1, new_y2 = y1_init, y2_init
        elif corner == "e":
            new_x1, new_x2 = x1_init, nx
            new_y1, new_y2 = y1_init, y2_init
        else:
            return
        new_x1, new_x2 = sorted((new_x1, new_x2))
        new_y1, new_y2 = sorted((new_y1, new_y2))
        min_size = MIN_BOX_SIZE_NORM
        if new_x2 - new_x1 < min_size:
            if corner in ("nw", "sw"):
                new_x1 = max(0.0, new_x2 - min_size)
            else:
                new_x2 = min(1.0, new_x1 + min_size)
        if new_y2 - new_y1 < min_size:
            if corner in ("nw", "ne"):
                new_y1 = max(0.0, new_y2 - min_size)
            else:
                new_y2 = min(1.0, new_y1 + min_size)
        new_x1 = min(max(new_x1, 0.0), 1.0)
        new_x2 = min(max(new_x2, 0.0), 1.0)
        new_y1 = min(max(new_y1, 0.0), 1.0)
        new_y2 = min(max(new_y2, 0.0), 1.0)
        target_box = self.boxes[idx]
        target_box.x_center = (new_x1 + new_x2) / 2
        target_box.y_center = (new_y1 + new_y2) / 2
        target_box.width = new_x2 - new_x1
        target_box.height = new_y2 - new_y1
        target_box.clamp()
        self.draw_boxes()
        self._update_listbox_entry(idx)
        self.view.set_status(
            f"Resizing box #{idx + 1}: w={target_box.width:.2f} h={target_box.height:.2f}. Release to commit."
        )

    def _finish_resize(self) -> None:
        if self.resize_target is None:
            return
        idx, _corner = self.resize_target
        self._set_dirty(True)
        self.view.set_status(f"Resize applied to box #{idx + 1}.")
        self._clear_resize_state()
        self.draw_boxes()

    def _cancel_resize(self) -> None:
        if self.resize_target is None or self.resize_start_box is None:
            return
        idx, _ = self.resize_target
        if 0 <= idx < len(self.boxes):
            original = self.resize_start_box
            self.boxes[idx] = LabelBox(
                class_id=original.class_id,
                x_center=original.x_center,
                y_center=original.y_center,
                width=original.width,
                height=original.height,
            )
            self._update_listbox_entry(idx)
        self._clear_resize_state()
        self.draw_boxes()
        self.view.set_status("Resize cancelled.")

    def _clear_resize_state(self) -> None:
        self.resize_target = None
        self.resize_start_box = None

    def _update_listbox_entry(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.boxes):
            return
        selected = self.view.get_selected_indices()
        items = [self._format_box_summary(i, b) for i, b in enumerate(self.boxes)]
        self.view.set_list_items(items)
        self.view.set_selection(selected)
        self._sync_class_selector_with_selection()

    def _set_dirty(self, value: bool) -> None:
        self.dirty = value
        self._update_title()

    def _reload_classes(self) -> None:
        dataset_root = self._infer_dataset_root()
        self.class_names = read_class_names_from_yaml(dataset_root)
        if not self.class_names:
            self.class_names = ["class0"]
        self.view.set_class_names(self.class_names)
        if not (0 <= self.current_class_id < len(self.class_names)):
            self.current_class_id = 0
        self.view.set_current_class_id(self.current_class_id)

    def _switch_to_dataset(self, image_dir: Path, label_dir: Path) -> None:
        if image_dir == self.image_dir and label_dir == self.label_dir:
            self.view.set_status("Already using selected folders.")
            return
        try:
            new_images = discover_images(image_dir)
        except FileNotFoundError:
            self.view.set_status(f"Image folder missing: {image_dir}")
            return
        if not new_images:
            # Switch to the selected folders but show empty state
            if self.dirty:
                self.save_labels()
            self.image_dir = image_dir
            self.label_dir = label_dir
            self.image_paths = []
            self.index = 0
            self.boxes = []
            self.add_mode = False
            self.add_start = None
            self._clear_resize_state()
            self.view.clear_preview_rect()
            self.view.set_selection([])
            self._reload_classes()
            self.view.set_directory_display(self.image_dir, self.label_dir)
            self._set_dirty(False)
            self.load_current()
            return
        if self.dirty:
            self.save_labels()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_paths = new_images
        self.index = 0
        self.boxes = []
        self.add_mode = False
        self.add_start = None
        self._clear_resize_state()
        self.view.clear_preview_rect()
        self.view.set_selection([])
        self._reload_classes()
        self.view.set_directory_display(self.image_dir, self.label_dir)
        self._set_dirty(False)
        self.load_current()

    def _suggest_label_dir(self, image_dir: Path) -> Path:
        if image_dir.parent.name == "images":
            dataset_root = image_dir.parent.parent
            return dataset_root / "labels" / image_dir.name
        return self.label_dir

    # Class helpers and wiring
    def _infer_dataset_root(self) -> Path:
        # Try images/.../train -> dataset root two levels up
        if self.image_dir.name in {"train", "val", "test"} and self.image_dir.parent.name == "images":
            return self.image_dir.parents[1]
        if self.label_dir.name in {"train", "val", "test"} and self.label_dir.parent.name == "labels":
            return self.label_dir.parents[1]
        # Fallback to common parent
        return self.image_dir.parent

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)

    def on_class_change(self, class_id: int) -> None:
        # Update current class selection and optionally apply to selected boxes
        self.current_class_id = class_id
        selected = self.view.get_selected_indices()
        if selected:
            for idx in selected:
                if 0 <= idx < len(self.boxes):
                    self.boxes[idx].class_id = class_id
            self._set_dirty(True)
            self._refresh_list()
        else:
            # No selection; just update status
            self.view.set_status(f"Current class set to {class_id}:{self._class_name(class_id)}")

    def _sync_class_selector_with_selection(self) -> None:
        selected = self.view.get_selected_indices()
        if len(selected) == 1:
            cid = self.boxes[selected[0]].class_id
            self.view.set_current_class_id(cid)
        else:
            self.view.set_current_class_id(self.current_class_id)
