from __future__ import annotations

from pathlib import Path
from typing import List

from .models import LabelBox


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_images(image_dir: Path) -> List[Path]:
    return sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )


def read_yolo_labels(label_path: Path) -> List[LabelBox]:
    if not label_path.exists():
        return []
    boxes: List[LabelBox] = []
    with label_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, w, h = parts
            try:
                boxes.append(
                    LabelBox(
                        class_id=int(class_id),
                        x_center=float(xc),
                        y_center=float(yc),
                        width=float(w),
                        height=float(h),
                    ).clamp()
                )
            except ValueError:
                continue
    return boxes


def write_yolo_labels(label_path: Path, boxes: List[LabelBox]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as fp:
        for box in boxes:
            fp.write(box.as_line() + "\n")
