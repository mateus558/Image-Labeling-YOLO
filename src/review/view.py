"""View layer for the label review GUI.

This module defines the Tkinter widgets and binds callbacks provided by a controller.
The goal is to keep UI wiring separate from business logic.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import tkinter as tk
from tkinter import ttk

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

        self.listbox = tk.Listbox(controls, height=20, width=40, selectmode=tk.EXTENDED)
        self.listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")

        self.btn_prev = ttk.Button(controls, text="Prev")
        self.btn_next = ttk.Button(controls, text="Next")
        self.btn_prev.grid(row=1, column=0, pady=4, sticky="ew")
        self.btn_next.grid(row=1, column=1, pady=4, sticky="ew")

        self.btn_add = ttk.Button(controls, text="Add Box")
        self.btn_del = ttk.Button(controls, text="Delete Selected")
        self.btn_add.grid(row=2, column=0, pady=4, sticky="ew")
        self.btn_del.grid(row=2, column=1, pady=4, sticky="ew")

        self.btn_save = ttk.Button(controls, text="Save")
        self.btn_save.grid(row=3, column=0, columnspan=2, pady=12, sticky="ew")

        self.status = ttk.Label(controls, text="")
        self.status.grid(row=4, column=0, columnspan=2, sticky="ew", pady=6)

        controls.rowconfigure(0, weight=1)
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

    def bind_canvas(self, on_press, on_drag, on_release, on_select_list) -> None:
        self.canvas.bind("<ButtonPress-1>", on_press)
        self.canvas.bind("<B1-Motion>", on_drag)
        self.canvas.bind("<ButtonRelease-1>", on_release)
        self.listbox.bind("<<ListboxSelect>>", on_select_list)

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
