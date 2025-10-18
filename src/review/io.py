from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from .models import LabelBox


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_images(image_dir: Path) -> List[Path]:
    # If the folder doesn't exist or isn't a directory, return an empty list
    if not image_dir or not image_dir.exists() or not image_dir.is_dir():
        return []
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_EXTS)


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


def read_class_names_from_yaml(dataset_root: Path) -> List[str]:
    """Very small YAML reader to extract YOLO 'names' from data.yaml without PyYAML.

    Looks under `dataset_root / "data.yaml"` for:
    - `names:` followed by `- name` lines
    - If not found, returns ["class0", ..., f"class{nc-1}"] if `nc: N` is present
    - Else returns ["class0"]
    """
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        return ["class0"]
    names: List[str] = []
    nc: int | None = None
    with yaml_path.open("r", encoding="utf-8") as fp:
        lines = fp.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("nc:"):
            try:
                nc = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
        if line.startswith("names:"):
            i += 1
            # Collect indented list items starting with '-'
            while i < len(lines):
                raw = lines[i]
                if raw.startswith(" ") or raw.startswith("\t"):
                    stripped = raw.strip()
                    if stripped.startswith("-"):
                        name = stripped[1:].strip()
                        # Strip quotes if any
                        if (name.startswith("'") and name.endswith("'")) or (
                            name.startswith('"') and name.endswith('"')
                        ):
                            name = name[1:-1]
                        names.append(name)
                        i += 1
                        continue
                break
            break
        i += 1
    if names:
        return names
    if nc is not None and nc > 0:
        return [f"class{i}" for i in range(nc)]
    return ["class0"]
