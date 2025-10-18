"""View layer for the label review GUI.

This module defines the Tkinter widgets and binds callbacks provided by a controller.
The goal is to keep UI wiring separate from business logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from .constants import BOX_LINE_WIDTH, CANVAS_MAX_HEIGHT, CANVAS_MAX_WIDTH, COLOR_PREVIEW, PREVIEW_DASH


class LabelReviewView:
    def __init__(self, root: Optional[tk.Tk] = None) -> None:
        self.root = root or tk.Tk()
        self.root.title("Label Review")
        self.root.geometry("1200x800")

        self.canvas = tk.Canvas(self.root, width=CANVAS_MAX_WIDTH, height=CANVAS_MAX_HEIGHT, background="#202020")
        self.canvas.grid(row=0, column=0, rowspan=6, sticky="nsew", padx=8, pady=8)

        controls = ttk.Frame(self.root)
        controls.grid(row=0, column=1, sticky="ns", padx=12, pady=12)

        dataset_frame = ttk.LabelFrame(controls, text="Folders")
        dataset_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        dataset_frame.columnconfigure(1, weight=1)

        ttk.Label(dataset_frame, text="Images").grid(row=0, column=0, sticky="w", padx=(4, 4), pady=2)
        self.image_dir_var = tk.StringVar(value="")
        self.image_dir_entry = ttk.Entry(dataset_frame, textvariable=self.image_dir_var, state="readonly")
        self.image_dir_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4), pady=2)
        self.btn_browse_images = ttk.Button(dataset_frame, text="Browse...")
        self.btn_browse_images.grid(row=0, column=2, sticky="ew", padx=(0, 4), pady=2)

        ttk.Label(dataset_frame, text="Labels").grid(row=1, column=0, sticky="w", padx=(4, 4), pady=2)
        self.label_dir_var = tk.StringVar(value="")
        self.label_dir_entry = ttk.Entry(dataset_frame, textvariable=self.label_dir_var, state="readonly")
        self.label_dir_entry.grid(row=1, column=1, sticky="ew", padx=(0, 4), pady=2)
        self.btn_browse_labels = ttk.Button(dataset_frame, text="Browse...")
        self.btn_browse_labels.grid(row=1, column=2, sticky="ew", padx=(0, 4), pady=2)

        self.listbox = tk.Listbox(controls, height=20, width=40, selectmode=tk.EXTENDED)
        self.listbox.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.btn_prev = ttk.Button(controls, text="Prev")
        self.btn_next = ttk.Button(controls, text="Next")
        self.btn_prev.grid(row=2, column=0, pady=4, sticky="ew")
        self.btn_next.grid(row=2, column=1, pady=4, sticky="ew")

        self.btn_add = ttk.Button(controls, text="Add Box")
        self.btn_del = ttk.Button(controls, text="Delete Selected")
        self.btn_add.grid(row=3, column=0, pady=4, sticky="ew")
        self.btn_del.grid(row=3, column=1, pady=4, sticky="ew")

        # Class selector
        ttk.Label(controls, text="Class").grid(row=4, column=0, sticky="w")
        self.class_var = tk.StringVar(value="0")
        self.class_combo = ttk.Combobox(controls, textvariable=self.class_var, state="readonly", width=32)
        self.class_combo.grid(row=4, column=1, sticky="ew")

        self.btn_save = ttk.Button(controls, text="Save")
        self.btn_save.grid(row=5, column=0, columnspan=2, pady=12, sticky="ew")

        self.status = ttk.Label(controls, text="")
        self.status.grid(row=6, column=0, columnspan=2, sticky="ew", pady=6)

        controls.rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self._photo = None
        self._preview_rect = None

    # Wiring helpers
    def bind_navigation(self, on_prev: Callable[[], None], on_next: Callable[[], None]) -> None:
        self.btn_prev.configure(command=on_prev)
        self.btn_next.configure(command=on_next)
        self.root.bind("<Left>", lambda _e: on_prev())
        self.root.bind("<Right>", lambda _e: on_next())
        self.root.bind("<Return>", lambda _e: on_next())
        self.root.bind("<BackSpace>", lambda _e: on_prev())

    def bind_editing(self, on_add: Callable[[], None], on_delete: Callable[[], None], on_save: Callable[[], None]) -> None:
        self.btn_add.configure(command=on_add)
        self.btn_del.configure(command=on_delete)
        self.btn_save.configure(command=on_save)
        self.root.bind("a", lambda _e: on_add())
        self.root.bind("A", lambda _e: on_add())
        self.root.bind("d", lambda _e: on_delete())
        self.root.bind("D", lambda _e: on_delete())
        self.root.bind("s", lambda _e: on_save())
        self.root.bind("S", lambda _e: on_save())

    def bind_directory_select(self, on_image: Callable[[], None], on_label: Callable[[], None]) -> None:
        self.btn_browse_images.configure(command=on_image)
        self.btn_browse_labels.configure(command=on_label)

    def bind_canvas(self, on_press, on_drag, on_release, on_select_list) -> None:
        self.canvas.bind("<ButtonPress-1>", on_press)
        self.canvas.bind("<B1-Motion>", on_drag)
        self.canvas.bind("<ButtonRelease-1>", on_release)
        self.listbox.bind("<<ListboxSelect>>", on_select_list)

    def bind_class_change(self, on_change: Callable[[int], None]) -> None:
        def _on_sel(_event=None):
            try:
                idx = int(self.class_var.get().split(" ")[0])
            except Exception:
                # Fallback: try direct int
                try:
                    idx = int(self.class_var.get())
                except Exception:
                    return
            on_change(idx)
        self.class_combo.bind("<<ComboboxSelected>>", _on_sel)

    # Render helpers
    def render_image_fit(self, image: Image.Image) -> tuple[int, int]:
        scale = min(CANVAS_MAX_WIDTH / image.width, CANVAS_MAX_HEIGHT / image.height, 1.0)
        display_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = getattr(Image, "BILINEAR", 2)
        resized = image.resize(display_size, resample)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.configure(width=display_size[0], height=display_size[1])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        return display_size

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def set_title(self, text: str) -> None:
        self.root.title(text)

    def set_directory_display(self, image_dir: Path, label_dir: Path) -> None:
        self.image_dir_var.set(str(image_dir))
        self.label_dir_var.set(str(label_dir))

    def prompt_directory(self, title: str, initialdir: Path | None = None) -> Path | None:
        initial = str(initialdir) if initialdir else ""
        result = filedialog.askdirectory(parent=self.root, title=title, initialdir=initial, mustexist=True)
        if not result:
            return None
        return Path(result)

    def set_list_items(self, items: Sequence[str]) -> None:
        self.listbox.delete(0, tk.END)
        for it in items:
            self.listbox.insert(tk.END, it)

    def get_selected_indices(self) -> List[int]:
        return [int(i) for i in self.listbox.curselection()]

    def set_selection(self, indices: Iterable[int]) -> None:
        self.listbox.selection_clear(0, tk.END)
        for i in indices:
            if i < self.listbox.size():
                self.listbox.selection_set(i)

    def mainloop(self) -> None:
        self.root.mainloop()

    # Class UI helpers
    def set_class_names(self, class_names: Sequence[str]) -> None:
        # Show as "id name" for clarity
        display = [f"{i} {name}" for i, name in enumerate(class_names)]
        self.class_combo["values"] = display
        # Keep current selection in range
        if display:
            if self.class_var.get() not in display:
                self.class_var.set(display[0])

    def set_current_class_id(self, class_id: int) -> None:
        values = list(self.class_combo["values"]) or []
        if 0 <= class_id < len(values):
            self.class_var.set(values[class_id])
        else:
            # Fallback to numeric id
            self.class_var.set(str(class_id))

    # Preview rectangle helpers (for add mode)
    def start_preview_rect(self, x: float, y: float) -> None:
        if self._preview_rect is not None:
            self.canvas.delete(self._preview_rect)
            self._preview_rect = None
        self._preview_rect = self.canvas.create_rectangle(
            x, y, x, y, outline=COLOR_PREVIEW, width=BOX_LINE_WIDTH, dash=PREVIEW_DASH
        )

    def update_preview_rect(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if self._preview_rect is not None:
            self.canvas.coords(self._preview_rect, x0, y0, x1, y1)

    def clear_preview_rect(self) -> None:
        if self._preview_rect is not None:
            self.canvas.delete(self._preview_rect)
            self._preview_rect = None
