"""
Train YOLOv8 on the labeled dataset under yolo_dataset/.

Features:
- Creates a train/val split from yolo_dataset/images/train into text file lists.
- Generates a minimal Ultralytics data.yaml with class names.
- Optionally touches missing label files as empty .txt to avoid loader warnings.
- Kicks off training via ultralytics.YOLO with common flags.

Usage (examples):
  python src/train_yolo.py --model yolov8n.pt --epochs 50 --val-frac 0.1
  python src/train_yolo.py --dataset-dir yolo_dataset --imgsz 640 --batch 16

Dataset layout expected:
  yolo_dataset/
    images/train/*.jpg|png|jpeg
    labels/train/*.txt    # may be empty files for negatives

If yolo_dataset/classes.txt exists, it's used as class names (one per line).
Otherwise defaults to a single class: ["display"].
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class TrainConfig:
    dataset_dir: Path
    model: str
    epochs: int
    imgsz: int
    batch: int
    val_frac: float
    seed: int
    device: str | None
    cache: bool
    touch_missing_labels: bool
    workers: int
    offline: bool


def find_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    imgs.sort()
    if not imgs:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return imgs


def label_for_image(img_path: Path, dataset_dir: Path) -> Path:
    # Replace /images/ with /labels/ and extension with .txt
    try:
        rel = img_path.relative_to(dataset_dir)
    except ValueError:
        # If not under dataset_dir, try to replace parts heuristically
        rel = Path("images") / img_path.name
    parts = list(rel.parts)
    if len(parts) >= 2 and parts[0] == "images":
        parts[0] = "labels"
    lbl_rel = Path(*parts).with_suffix(".txt")
    return dataset_dir / lbl_rel


def ensure_label_files(images: List[Path], dataset_dir: Path) -> None:
    for img in images:
        lbl = label_for_image(img, dataset_dir)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        if not lbl.exists():
            # Create empty label file for negatives or unlabeled samples
            lbl.touch()


def make_splits(images: List[Path], val_frac: float, seed: int) -> Tuple[List[Path], List[Path]]:
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be between 0 and 1 (exclusive)")
    rng = random.Random(seed)
    imgs = images.copy()
    rng.shuffle(imgs)
    n_val = max(1 if len(imgs) > 1 else 0, int(round(len(imgs) * val_frac)))
    val = imgs[:n_val]
    train = imgs[n_val:]
    if not train:
        # If split consumed all but none training, adjust
        train, val = imgs, []
    return train, val


def write_split_files(train_imgs: List[Path], val_imgs: List[Path], splits_dir: Path) -> Tuple[Path, Path]:
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_txt = splits_dir / "train.txt"
    val_txt = splits_dir / "val.txt"
    train_txt.write_text("\n".join(str(p.resolve()) for p in train_imgs) + "\n")
    val_txt.write_text("\n".join(str(p.resolve()) for p in val_imgs) + "\n")
    return train_txt, val_txt


def read_class_names(dataset_dir: Path) -> List[str]:
    classes_txt = dataset_dir / "classes.txt"
    if classes_txt.exists():
        names = [ln.strip() for ln in classes_txt.read_text().splitlines() if ln.strip()]
        if names:
            return names
    # Fallback single-class default
    return ["display"]


def write_data_yaml(dataset_dir: Path, names: List[str], train_txt: Path, val_txt: Path) -> Path:
    data_yaml = dataset_dir / "data.yaml"
    content = [
        f"nc: {len(names)}",
        "names:",
    ]
    content += [f"  - {n}" for n in names]
    content += [
        f"train: {train_txt.resolve()}",
        f"val: {val_txt.resolve()}",
    ]
    data_yaml.write_text("\n".join(content) + "\n")
    return data_yaml


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train YOLOv8 on yolo_dataset labels")
    ap.add_argument("--dataset-dir", type=Path, default=Path("yolo_dataset"), help="Dataset root directory")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Model to train (weights or YAML)")
    ap.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    ap.add_argument("--imgsz", type=int, default=640, help="Training image size")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction for split")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split")
    ap.add_argument("--device", type=str, default=None, help="Device id, e.g. '0', 'cpu', or '0,1' (auto if omitted)")
    ap.add_argument("--no-cache", action="store_true", help="Disable dataloader caching")
    ap.add_argument("--no-touch-missing-labels", action="store_true", help="Do not create empty .txt for missing labels")
    ap.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    ap.add_argument("--offline", action="store_true", help="Avoid remote downloads; use architecture YAML instead of .pt")
    args = ap.parse_args()
    return TrainConfig(
        dataset_dir=args.dataset_dir,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        val_frac=args.val_frac,
        seed=args.seed,
        device=args.device,
        cache=not args.no_cache,
        touch_missing_labels=not args.no_touch_missing_labels,
        workers=args.workers,
        offline=args.offline,
    )


def main() -> int:
    cfg = parse_args()

    images_dir = cfg.dataset_dir / "images" / "train"
    labels_dir = cfg.dataset_dir / "labels" / "train"
    if not labels_dir.exists():
        labels_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(images_dir)
    if cfg.touch_missing_labels:
        ensure_label_files(images, cfg.dataset_dir)

    train_imgs, val_imgs = make_splits(images, cfg.val_frac, cfg.seed)

    splits_dir = cfg.dataset_dir / "splits"
    train_txt, val_txt = write_split_files(train_imgs, val_imgs, splits_dir)

    names = read_class_names(cfg.dataset_dir)
    data_yaml = write_data_yaml(cfg.dataset_dir, names, train_txt, val_txt)

    # Delay import to allow quick --help without ultralytics installed
    try:
        # Preferred public import
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except Exception:
        try:
            from ultralytics.yolo.engine.model import YOLO  # type: ignore
        except Exception as e:
            print("Ultralytics is required. Install with: pip install ultralytics", file=sys.stderr)
            print(f"Import error: {e}", file=sys.stderr)
            return 2

    # Resolve model considering offline mode and download failures
    requested_model = cfg.model
    use_yaml = False
    if cfg.offline and requested_model.endswith(".pt") and "yolov8" in requested_model:
        requested_model = requested_model.replace(".pt", ".yaml")
        use_yaml = True

    try:
        model = YOLO(requested_model)
    except Exception as e:
        msg = str(e)
        download_related = any(s in msg for s in [
            "CERTIFICATE_VERIFY_FAILED",
            "Curl return value",
            "Download failure",
            "attempt_download_asset",
        ])
        yaml_candidate = None
        if requested_model.endswith(".pt") and "yolov8" in requested_model:
            yaml_candidate = requested_model.replace(".pt", ".yaml")
        if download_related and yaml_candidate and not use_yaml:
            print("Encountered download error loading weights. Retrying with architecture YAML (training from scratch):", file=sys.stderr)
            print(f"  {requested_model} -> {yaml_candidate}", file=sys.stderr)
            model = YOLO(yaml_candidate)
        else:
            raise

    run_name = f"{Path(cfg.model).stem}-sz{cfg.imgsz}-e{cfg.epochs}-{int(time.time())}"
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        name=run_name,
        project=str(Path("runs") / "train"),
        device=cfg.device if cfg.device else None,
        cache=cfg.cache,
        workers=cfg.workers,
        exist_ok=True,
    )

    print("Starting training with config:", train_kwargs)
    results = model.train(**{k: v for k, v in train_kwargs.items() if v is not None})
    # results is a ultralytics.engine.results.Results object; best weights saved under results.save_dir / 'weights/best.pt'
    try:
        save_dir = Path(getattr(results, "save_dir", Path("runs/train/exp")))
        best = save_dir / "weights" / "best.pt"
        print(f"Training finished. Best weights: {best}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
