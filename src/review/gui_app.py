"""Tkinter GUI for reviewing and editing YOLO bounding boxes."""
from __future__ import annotations

from pathlib import Path
from typing import List

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from .io import discover_images, read_yolo_labels, write_yolo_labels
from .models import LabelBox

# UI constants
CANVAS_MAX_WIDTH = 960
CANVAS_MAX_HEIGHT = 720
COLOR_BOX = "#40c9ff"
COLOR_BOX_SELECTED = "#ffcc00"
COLOR_PREVIEW = "#ffcc00"
BOX_LINE_WIDTH = 2
PREVIEW_DASH = (4, 2)
MIN_BOX_SIZE_NORM = 1e-3


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

        self.root = tk.Tk()
        self.root.title("Label Review")
        self.root.geometry("1200x800")

        self.canvas = tk.Canvas(self.root, width=CANVAS_MAX_WIDTH, height=CANVAS_MAX_HEIGHT, background="#202020")
        self.canvas.grid(row=0, column=0, rowspan=6, sticky="nsew", padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        controls = ttk.Frame(self.root)
        controls.grid(row=0, column=1, sticky="ns", padx=12, pady=12)

        self.listbox = tk.Listbox(controls, height=20, width=40, selectmode=tk.EXTENDED)
        self.listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.listbox.bind("<<ListboxSelect>>", self._on_select_box)

        btn_prev = ttk.Button(controls, text="Prev", command=self.prev_image)
        btn_next = ttk.Button(controls, text="Next", command=self.next_image)
        btn_prev.grid(row=1, column=0, pady=4, sticky="ew")
        btn_next.grid(row=1, column=1, pady=4, sticky="ew")

        btn_add = ttk.Button(controls, text="Add Box", command=self.start_add_box)
        btn_del = ttk.Button(controls, text="Delete Selected", command=self.delete_selected_box)
        btn_add.grid(row=2, column=0, pady=4, sticky="ew")
        btn_del.grid(row=2, column=1, pady=4, sticky="ew")

        btn_save = ttk.Button(controls, text="Save", command=self.save_labels)
        btn_save.grid(row=3, column=0, columnspan=2, pady=12, sticky="ew")

        self.status = ttk.Label(controls, text="Click Add Box, then click and drag on the image.")
        self.status.grid(row=4, column=0, columnspan=2, sticky="ew", pady=6)

        controls.rowconfigure(0, weight=1)
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.root.bind("<Left>", lambda _event: self.prev_image())
        self.root.bind("<Right>", lambda _event: self.next_image())
        self.root.bind("<Escape>", lambda _event: self._cancel_add_mode())
        # Extra keybindings
        self.root.bind("a", lambda _event: self.start_add_box())
        self.root.bind("A", lambda _event: self.start_add_box())
        self.root.bind("d", lambda _event: self.delete_selected_box())
        self.root.bind("D", lambda _event: self.delete_selected_box())
        self.root.bind("s", lambda _event: self.save_labels())
        self.root.bind("S", lambda _event: self.save_labels())
        self.root.bind("n", lambda _event: self.next_image())
        self.root.bind("N", lambda _event: self.next_image())
        self.root.bind("p", lambda _event: self.prev_image())
        self.root.bind("P", lambda _event: self.prev_image())
        self.root.bind("<Return>", lambda _event: self.next_image())
        self.root.bind("<BackSpace>", lambda _event: self.prev_image())
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
        self.canvas.delete("box")
        selected = set(self._selected_indices())
        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self._normalized_to_canvas(box)
            color = COLOR_BOX_SELECTED if idx in selected else COLOR_BOX
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=color, width=BOX_LINE_WIDTH, tags=("box", f"box-{idx}")
            )

    def _normalized_to_canvas(self, box: LabelBox) -> tuple[float, float, float, float]:
        w = box.width * self.display_width
        h = box.height * self.display_height
        x_center = box.x_center * self.display_width
        y_center = box.y_center * self.display_height
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return x1, y1, x2, y2

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
        else:
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
        if not self.add_mode or self.add_start is None or self._preview_rect is None:
            return
        self.canvas.coords(self._preview_rect, self.add_start[0], self.add_start[1], event.x, event.y)

    def _on_canvas_release(self, event: tk.Event) -> None:
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
            summary = f"#{idx+1}: x={box.x_center:.2f} y={box.y_center:.2f} w={box.width:.2f} h={box.height:.2f}"
            self.listbox.insert(tk.END, summary)
        self._draw_boxes()

    def load_current_image(self) -> None:
        self.current_image = self.image_paths[self.index]
        image = self._load_image(self.current_image)
        self.boxes = self._load_labels(self.current_image)
        self.add_mode = False
        self.add_start = None
        self._clear_preview()
        self._update_title()
        self._render_image(image)
        self._refresh_listbox()
        self.status.configure(
            text="Use arrow keys or N/P to navigate. A to add, D to delete, S to save. Esc cancels add mode."
        )

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
