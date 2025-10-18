"""Controller for the label review GUI.

This module orchestrates I/O, view updates, and label editing logic. It is a
future target for moving logic out of gui_app.LabelReviewApp incrementally.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Protocol

from PIL import Image

from .io import discover_images, read_yolo_labels, write_yolo_labels
from .models import LabelBox


class View(Protocol):
    def render_image_fit(self, image: Image.Image) -> tuple[int, int]: ...
    def set_status(self, text: str) -> None: ...
    def set_title(self, text: str) -> None: ...
    def set_list_items(self, items: List[str]) -> None: ...
    def get_selected_indices(self) -> List[int]: ...


class LabelReviewController:
    def __init__(self, image_dir: Path, label_dir: Path, view: View) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.view = view
        self.image_paths = discover_images(self.image_dir)
        if not self.image_paths:
            raise SystemExit(f"No images found in {image_dir}")
        self.index = 0
        self.boxes: List[LabelBox] = []
        self.display_width = 1
        self.display_height = 1

    def _label_path(self, image_path: Path) -> Path:
        return self.label_dir / f"{image_path.stem}.txt"

    def load_current(self) -> None:
        current = self.image_paths[self.index]
        image = Image.open(current).convert("RGB")
        self.boxes = read_yolo_labels(self._label_path(current))
        self.display_width, self.display_height = self.view.render_image_fit(image)
        self.view.set_title(
            f"Label Review - {current.name} ({self.index + 1}/{len(self.image_paths)})"
        )
        self.view.set_list_items([
            f"#{i+1}: x={b.x_center:.2f} y={b.y_center:.2f} w={b.width:.2f} h={b.height:.2f}"
            for i, b in enumerate(self.boxes)
        ])
        self.view.set_status("Use arrow keys to navigate. A to add, D to delete, S to save.")
