"""Tkinter GUI for reviewing and editing YOLO bounding boxes."""
from __future__ import annotations

from pathlib import Path
from typing import List

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

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


class LabelReviewApp:
    def __init__(self, image_dir: Path, label_dir: Path) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_paths = discover_images(self.image_dir)
        if not self.image_paths:
            raise SystemExit(f"No images found in {image_dir}")

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

        self.view.bind_navigation(self.prev_image, self.next_image)
        self.view.bind_editing(self.start_add_box, self.delete_selected_box, self.save_labels)
        self.view.bind_canvas(
            self._on_canvas_press, self._on_canvas_drag, self._on_canvas_release, self._on_select_box
        )
        self.root.bind("<Escape>", lambda _event: self._cancel_add_mode())
        self.root.bind("n", lambda _event: self.next_image())
        self.root.bind("N", lambda _event: self.next_image())
        self.root.bind("p", lambda _event: self.prev_image())
        self.root.bind("P", lambda _event: self.prev_image())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._photo = None
        self.display_width = 1
        self.display_height = 1
        self._preview_rect = None

        self.load_current_image()

    def _label_path(self, image_path: Path) -> Path:
        return self.label_dir / f"{image_path.stem}.txt"

    def _load_labels(self, image_path: Path) -> List[LabelBox]:
        return read_yolo_labels(self._label_path(image_path))

    def _load_image(self, image_path: Path):
        return Image.open(image_path).convert("RGB")

    def _render_image(self, image):
        max_width = CANVAS_MAX_WIDTH
        max_height = CANVAS_MAX_HEIGHT
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        display_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        try:
            resample = Image.Resampling.BILINEAR  # Pillow>=9
        except AttributeError:
            resample = getattr(Image, "BILINEAR", 2)  # Pillow<9 fallback constant value
        resized = image.resize(display_size, resample)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.configure(width=display_size[0], height=display_size[1])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.display_width, self.display_height = display_size
        self._draw_boxes()

    def _draw_boxes(self) -> None:
        canvas_utils.draw_boxes(
            self.canvas,
            self.boxes,
            self._selected_indices(),
            self.display_width,
            self.display_height,
        )

    def _normalized_to_canvas(self, box: LabelBox) -> tuple[float, float, float, float]:
        return canvas_utils.normalized_to_canvas(box, self.display_width, self.display_height)

    def _canvas_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        return x / max(1, self.display_width), y / max(1, self.display_height)

    def _selected_indices(self) -> List[int]:
        return [int(idx) for idx in self.listbox.curselection()]

    def _on_select_box(self, _event: tk.Event) -> None:
        self._draw_boxes()

    def _clear_preview(self) -> None:
        if self._preview_rect is not None:
            self.canvas.delete(self._preview_rect)
            self._preview_rect = None

    def start_add_box(self) -> None:
        self.add_mode = True
        self.add_start = None
        self._clear_preview()
        self.listbox.selection_clear(0, tk.END)
        self.status.configure(
            text="Add mode: click and drag to draw the box, release to finish. (Esc to cancel)"
        )

    def _on_canvas_press(self, event: tk.Event) -> None:
        if self.add_mode:
            self.add_start = (event.x, event.y)
            self._clear_preview()
            self._preview_rect = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline=COLOR_PREVIEW, width=BOX_LINE_WIDTH, dash=PREVIEW_DASH
            )
            self.status.configure(text="Drag to size the box, release to confirm.")
            return

        handle = self._detect_handle()
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
                current = self._selected_indices()
                anchor = current[0] if current else idx
                start = min(anchor, idx)
                end = max(anchor, idx)
                self.listbox.selection_clear(0, tk.END)
                for i in range(start, end + 1):
                    self.listbox.selection_set(i)
            elif ctrl:
                if idx in self._selected_indices():
                    self.listbox.selection_clear(idx)
                else:
                    self.listbox.selection_set(idx)
            else:
                self.listbox.selection_clear(0, tk.END)
                self.listbox.selection_set(idx)
            self._draw_boxes()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self.resize_target is not None:
            self._update_resize(event.x, event.y)
            return
        if not self.add_mode or self.add_start is None or self._preview_rect is None:
            return
        self.canvas.coords(self._preview_rect, self.add_start[0], self.add_start[1], event.x, event.y)

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self.resize_target is not None:
            self._finish_resize()
            return
        if not self.add_mode or self.add_start is None:
            return

        start_x, start_y = self.add_start
        end_x, end_y = event.x, event.y
        self.add_start = None

        self._clear_preview()

        start_norm = self._canvas_to_normalized(start_x, start_y)
        end_norm = self._canvas_to_normalized(end_x, end_y)
        x0n = min(max(start_norm[0], 0.0), 1.0)
        y0n = min(max(start_norm[1], 0.0), 1.0)
        x1n = min(max(end_norm[0], 0.0), 1.0)
        y1n = min(max(end_norm[1], 0.0), 1.0)
        width = abs(x1n - x0n)
        height = abs(y1n - y0n)
        if width < MIN_BOX_SIZE_NORM or height < MIN_BOX_SIZE_NORM:
            messagebox.showinfo("Too small", "Draw a larger box.")
            self.status.configure(text="Add mode cancelled; box too small.")
            return

        x_center = (x0n + x1n) / 2
        y_center = (y0n + y1n) / 2
        new_box = LabelBox(class_id=0, x_center=x_center, y_center=y_center, width=width, height=height).clamp()
        self.boxes.append(new_box)
        self._refresh_listbox()
        self.add_mode = False
        self._set_dirty(True)
        self.status.configure(text="Box added. Click Add Box to draw another.")

    def _find_box_at(self, x: float, y: float) -> int | None:
        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self._normalized_to_canvas(box)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return None

    def delete_selected_box(self) -> None:
        if self.resize_target is not None:
            self._cancel_resize()
        selected = self._selected_indices()
        if not selected:
            messagebox.showinfo("No selection", "Select at least one box to delete.")
            return
        for idx in sorted(selected, reverse=True):
            del self.boxes[idx]
        self._refresh_listbox()
        count = len(selected)
        noun = "box" if count == 1 else "boxes"
        if count:
            self._set_dirty(True)
        self.status.configure(text=f"Deleted {count} {noun}.")

    def _cancel_add_mode(self) -> None:
        if self.add_mode:
            self.add_mode = False
            self.add_start = None
            self._clear_preview()
            self.status.configure(text="Add cancelled.")
        elif self.resize_target is not None:
            self._cancel_resize()

    def save_labels(self) -> None:
        label_path = self._label_path(self.image_paths[self.index])
        if self.boxes:
            write_yolo_labels(label_path, self.boxes)
        elif label_path.exists():
            label_path.unlink()
        self._set_dirty(False)
        self.status.configure(text=f"Saved {label_path.name}.")

    def _refresh_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        for idx, box in enumerate(self.boxes):
            self.listbox.insert(tk.END, self._format_box_summary(idx, box))
        self._draw_boxes()

    def load_current_image(self) -> None:
        self.current_image = self.image_paths[self.index]
        image = self._load_image(self.current_image)
        self.boxes = self._load_labels(self.current_image)
        self.add_mode = False
        self.add_start = None
        self.resize_target = None
        self.resize_start_box = None
        self._clear_preview()
        self._update_title()
        self._render_image(image)
        self._refresh_listbox()
        self.status.configure(text="Use arrow keys or N/P to navigate. A to add, D to delete, S to save. Esc cancels add mode.")

    def prev_image(self) -> None:
        self.save_labels()
        self.index = (self.index - 1) % len(self.image_paths)
        self.load_current_image()

    def next_image(self) -> None:
        self.save_labels()
        self.index = (self.index + 1) % len(self.image_paths)
        self.load_current_image()

    def run(self) -> None:
        self.root.mainloop()

    def _on_close(self) -> None:
        self.save_labels()
        self.root.destroy()

    def _set_dirty(self, value: bool) -> None:
        self.dirty = value
        self._update_title()

    def _update_title(self) -> None:
        dirty_mark = " *" if getattr(self, "dirty", False) else ""
        self.root.title(
            f"Label Review - {self.image_paths[self.index].name} ({self.index + 1}/{len(self.image_paths)}){dirty_mark}"
        )

    def _detect_handle(self) -> tuple[int, str] | None:
        current = self.canvas.find_withtag("current")
        if not current:
            return None
        tags = self.canvas.gettags(current[0])
        for tag in tags:
            if tag.startswith("handle-"):
                try:
                    _, idx_str, corner = tag.split("-", 2)
                    return int(idx_str), corner
                except ValueError:
                    return None
        return None

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
        self.status.configure(text=f"Resizing box #{idx + 1}; drag to adjust, release to commit.")

    def _update_resize(self, canvas_x: float, canvas_y: float) -> None:
        if self.resize_target is None or self.resize_start_box is None:
            return
        idx, corner = self.resize_target
        if idx < 0 or idx >= len(self.boxes):
            return

        start_box = self.resize_start_box
        x1_init, y1_init, x2_init, y2_init = self._box_corners_norm(start_box)
        nx, ny = self._canvas_to_normalized(canvas_x, canvas_y)
        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)

        if corner == "nw":
            new_x1, new_y1, new_x2, new_y2 = nx, ny, x2_init, y2_init
        elif corner == "ne":
            new_x1, new_y1, new_x2, new_y2 = x1_init, ny, nx, y2_init
        elif corner == "sw":
            new_x1, new_y1, new_x2, new_y2 = nx, y1_init, x2_init, ny
        else:  # "se"
            new_x1, new_y1, new_x2, new_y2 = x1_init, y1_init, nx, ny

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

        self._draw_boxes()
        self._update_listbox_entry(idx)
        self.status.configure(
            text=(
                f"Resizing box #{idx + 1}: w={target_box.width:.2f} h={target_box.height:.2f}. Release to commit."
            )
        )

    def _finish_resize(self) -> None:
        if self.resize_target is None:
            return
        idx, _corner = self.resize_target
        self._set_dirty(True)
        self.status.configure(text=f"Resize applied to box #{idx + 1}.")
        self._clear_resize_state()
        self._draw_boxes()

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
        self._draw_boxes()
        self.status.configure(text="Resize cancelled.")

    def _clear_resize_state(self) -> None:
        self.resize_target = None
        self.resize_start_box = None

    def _update_listbox_entry(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.boxes):
            return
        selected = self._selected_indices()
        summary = self._format_box_summary(idx, self.boxes[idx])
        self.listbox.delete(idx)
        self.listbox.insert(idx, summary)
        for sel in selected:
            if sel < self.listbox.size():
                self.listbox.selection_set(sel)

    def _box_corners_norm(self, box: LabelBox) -> tuple[float, float, float, float]:
        return canvas_utils.box_corners_norm(box)

    def _format_box_summary(self, idx: int, box: LabelBox) -> str:
        return f"#{idx+1}: x={box.x_center:.2f} y={box.y_center:.2f} w={box.width:.2f} h={box.height:.2f}"

    def _draw_handles(self, idx: int, x1: float, y1: float, x2: float, y2: float) -> None:
        canvas_utils.draw_handles(self.canvas, idx, x1, y1, x2, y2)
