"""Convert Grounding DINO JSON outputs into YOLO label files."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from tqdm import tqdm

CLASS_ID = 0  # single-class dataset for machine displays


@dataclass
class Detection:
    bbox: np.ndarray  # xyxy in relative coordinates (0-1 range)
    score: float
    phrase: str

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return max(self.width * self.height, 0.0)


@dataclass
class ImageRecord:
    json_path: Path
    image_path: Path
    image_width: int
    image_height: int
    detections: List[Detection]

    @property
    def area(self) -> float:
        return float(self.image_width * self.image_height)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=root / "outputs" / "auto_labels",
        help="Directory containing JSON detection files.",
    )
    parser.add_argument(
        "--pos-image-dir",
        type=Path,
        default=root / "seed_images" / "pos",
        help="Directory containing positive seed images.",
    )
    parser.add_argument(
        "--neg-image-dir",
        type=Path,
        default=root / "seed_images" / "neg",
        help="Directory containing negative seed images.",
    )
    parser.add_argument(
        "--yolo-image-dir",
        type=Path,
        default=root / "yolo_dataset" / "images" / "train",
        help="Destination directory for YOLO training images.",
    )
    parser.add_argument(
        "--yolo-label-dir",
        type=Path,
        default=root / "yolo_dataset" / "labels" / "train",
        help="Destination directory for YOLO training labels.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.30,
        help="Minimum detection confidence score to keep.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.01,
        help="Minimum relative area (percentage of image, 0-1) to keep.",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.6,
        help="IoU threshold to merge overlapping boxes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing YOLO labels and copied images.",
    )
    return parser.parse_args()


def load_image_record(json_path: Path, pos_dir: Path) -> ImageRecord:
    with json_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    image_name = payload.get("image_name") or f"{json_path.stem}{json_path.suffix}"
    image_path = Path(payload.get("image_path", pos_dir / image_name))
    if not image_path.exists():
        fallback = pos_dir / image_name
        if fallback.exists():
            image_path = fallback
        else:
            raise FileNotFoundError(f"Cannot locate image referenced by {json_path}")

    size_dict = payload.get("image_size", {})
    width = int(size_dict.get("width", 0))
    height = int(size_dict.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {json_path}")

    detections = []
    for det in payload.get("detections", []):
        bbox = np.asarray(det.get("bbox", []), dtype=float)
        if bbox.shape != (4,):  # skip malformed entries
            continue
        bbox = np.clip(bbox, 0.0, 1.0)
        detections.append(
            Detection(
                bbox=bbox,
                score=float(det.get("score", 0.0)),
                phrase=str(det.get("phrase", "")).strip(),
            )
        )

    return ImageRecord(json_path=json_path, image_path=image_path, image_width=width, image_height=height, detections=detections)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max((xa2 - xa1) * (ya2 - ya1), 0.0)
    area_b = max((xb2 - xb1) * (yb2 - yb1), 0.0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def filter_detections(detections: Iterable[Detection], min_score: float, min_area: float) -> List[Detection]:
    return [det for det in detections if det.score >= min_score and det.area >= min_area]


def merge_detections(detections: Sequence[Detection], threshold: float) -> List[Detection]:
    if not detections:
        return []

    clusters: List[List[int]] = []
    unvisited = set(range(len(detections)))

    while unvisited:
        idx = unvisited.pop()
        cluster = {idx}
        queue = [idx]
        while queue:
            current = queue.pop()
            for other in list(unvisited):
                if iou(detections[current].bbox, detections[other].bbox) > threshold:
                    unvisited.remove(other)
                    cluster.add(other)
                    queue.append(other)
        clusters.append(sorted(cluster))

    merged: List[Detection] = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged.append(detections[cluster[0]])
            continue
        boxes = np.stack([detections[i].bbox for i in cluster], axis=0)
        scores = np.array([detections[i].score for i in cluster], dtype=float)
        weights = scores / np.clip(scores.sum(), a_min=1e-6, a_max=None)
        bbox = np.sum(boxes * weights[:, None], axis=0)
        best_idx = cluster[int(scores.argmax())]
        merged.append(Detection(bbox=bbox, score=float(scores.max()), phrase=detections[best_idx].phrase))

    return merged


def write_yolo_label(label_path: Path, detections: Sequence[Detection]) -> None:
    if not detections:
        if label_path.exists():
            label_path.unlink()
        return

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as fp:
        for det in detections:
            x1, y1, x2, y2 = det.bbox.tolist()
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            width = max(x2 - x1, 0.0)
            height = max(y2 - y1, 0.0)
            fp.write(
                f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )


def copy_image(image_path: Path, destination_dir: Path, overwrite: bool) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / image_path.name
    if destination_path.exists() and not overwrite:
        return destination_path
    shutil.copy2(image_path, destination_path)
    return destination_path


def process_positive_images(
    records: Sequence[ImageRecord],
    yolo_image_dir: Path,
    yolo_label_dir: Path,
    min_score: float,
    min_area: float,
    merge_iou_threshold: float,
    overwrite: bool,
) -> None:
    for record in tqdm(records, desc="JSON->YOLO"):
        filtered = filter_detections(record.detections, min_score=min_score, min_area=min_area)
        merged = merge_detections(filtered, threshold=merge_iou_threshold)
        copy_image(record.image_path, yolo_image_dir, overwrite=overwrite)
        label_path = yolo_label_dir / f"{record.image_path.stem}.txt"
        write_yolo_label(label_path, merged)


def process_negative_images(neg_dir: Path, yolo_image_dir: Path, yolo_label_dir: Path, overwrite: bool) -> None:
    neg_images = sorted(
        path
        for path in neg_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    for image_path in tqdm(neg_images, desc="Copy negatives"):
        copy_image(image_path, yolo_image_dir, overwrite=overwrite)
        label_path = yolo_label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            label_path.unlink()


def main() -> None:
    args = parse_args()
    args.yolo_image_dir.mkdir(parents=True, exist_ok=True)
    args.yolo_label_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(path for path in args.json_dir.glob("*.json") if path.is_file())
    records = [load_image_record(path, args.pos_image_dir) for path in json_files]

    process_positive_images(
        records=records,
        yolo_image_dir=args.yolo_image_dir,
        yolo_label_dir=args.yolo_label_dir,
        min_score=args.min_score,
        min_area=args.min_area,
        merge_iou_threshold=args.merge_iou,
        overwrite=args.overwrite,
    )

    process_negative_images(
        neg_dir=args.neg_image_dir,
        yolo_image_dir=args.yolo_image_dir,
        yolo_label_dir=args.yolo_label_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
