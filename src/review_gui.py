"""CLI entrypoint for the review GUI."""
from __future__ import annotations

import argparse
from pathlib import Path

from review.gui_app import LabelReviewApp


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults: prefer .../images/train and .../labels/train; fall back to roots
    images_root = root / "yolo_dataset" / "images"
    labels_root = root / "yolo_dataset" / "labels"
    train_images = images_root / "train"
    train_labels = labels_root / "train"
    default_images = train_images if train_images.exists() else images_root
    default_labels = train_labels if train_labels.exists() else labels_root
    parser.add_argument("--image-dir", type=Path, default=default_images, help="Directory containing YOLO images.")
    parser.add_argument("--label-dir", type=Path, default=default_labels, help="Directory containing YOLO label files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = LabelReviewApp(args.image_dir, args.label_dir)
    app.run()


if __name__ == "__main__":
    main()
