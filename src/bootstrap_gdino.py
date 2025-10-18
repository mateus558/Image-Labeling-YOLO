"""Bootstrap detections for exercise machine displays using Grounding DINO.

This script scans positive seed images, runs an open-vocabulary detector with a
predefined prompt, saves raw detections to JSON, and writes visualizations to
help human reviewers triage the results quickly.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.error import URLError

import torch
from PIL import Image  # type: ignore[import-not-found]
from tqdm import tqdm  # type: ignore[import-not-found]

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:
    np = None  # type: ignore[assignment]

try:
    from groundingdino.util.inference import annotate, load_image, load_model, predict
except ImportError as exc:
    raise SystemExit(
        "groundingdino package is required. Install it with `pip install groundingdino-py`."
    ) from exc

LOGGER = logging.getLogger("bootstrap_gdino")

DEFAULT_PROMPT = (
    "exercise machine display, treadmill console, rowing machine monitor, "
    "bike computer, elliptical console"
)
DEFAULT_BOX_THR = 0.25
DEFAULT_TEXT_THR = 0.25

# ✅ Public mirror (no token required)
DEFAULT_CHECKPOINT_URL = (
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/"
    "groundingdino_swint_ogc.pth"
)
DEFAULT_CONFIG_NAME = "GroundingDINO_SwinT_OGC.py"


def resolve_project_root() -> Path:
    """Return the repository root (one level above this file)."""
    return Path(__file__).resolve().parents[1]


def default_paths() -> Dict[str, Path]:
    """Compute default input/output directories relative to project root."""
    root = resolve_project_root()
    return {
        "seed_dir": root / "seed_images" / "pos",
        "viz_dir": root / "outputs" / "viz",
        "json_dir": root / "outputs" / "auto_labels",
    }


def ensure_checkpoint(path: Path, hf_token: str | None = None) -> Path:
    """Download the default Grounding DINO checkpoint if it is missing."""
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading Grounding DINO checkpoint to %s", path)
    try:
        torch.hub.download_url_to_file(DEFAULT_CHECKPOINT_URL, str(path))
        LOGGER.info("Downloaded checkpoint directly from public mirror.")
        return path
    except URLError as url_err:
        LOGGER.warning("Direct download failed (%s). Trying Hugging Face hub...", url_err.reason or url_err)
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import (
                GatedRepoError,
                HfHubHTTPError,
                RepositoryNotFoundError,
            )
        except ImportError as import_err:
            raise RuntimeError(
                "Automatic download failed and huggingface_hub is not installed. "
                "Install it or manually download groundingdino_swint_ogc.pth "
                "from https://huggingface.co/ShilongLiu/GroundingDINO "
                "and supply --checkpoint-path."
            ) from import_err

        try:
            # ✅ Try the public repo first
            cached_path = hf_hub_download(
                repo_id="ShilongLiu/GroundingDINO",
                filename="groundingdino_swint_ogc.pth",
                token=hf_token,
            )
        except RepositoryNotFoundError:
            # fallback to the official gated repo
            cached_path = hf_hub_download(
                repo_id="IDEA-Research/GroundingDINO",
                filename="groundingdino_swint_ogc.pth",
                token=hf_token,
            )
        except GatedRepoError as gated_err:
            raise RuntimeError(
                "Access to the Grounding DINO weights is gated. Visit "
                "https://huggingface.co/IDEA-Research/GroundingDINO, accept the license, "
                "and retry with a valid token."
            ) from gated_err
        except HfHubHTTPError as http_err:
            raise RuntimeError(
                f"Hugging Face responded with HTTP {http_err.response.status_code if http_err.response else 'error'}. "
                "Retry after logging in via `huggingface-cli login` or manually download the checkpoint."
            ) from http_err
        except Exception as hub_err:
            raise RuntimeError(
                "Failed to download Grounding DINO weights. Provide a valid Hugging Face token via "
                "--hf-token or HF_TOKEN, or download manually and use --checkpoint-path."
            ) from hub_err

        shutil.copyfile(cached_path, path)
        LOGGER.info("Copied checkpoint from Hugging Face cache: %s", cached_path)
        return path


def resolve_model_paths(
    config_path: Path | None,
    checkpoint_path: Path | None,
    hf_token: str | None = None,
) -> Tuple[str, str]:
    """Resolve model config and checkpoint paths, downloading weights as needed."""
    if config_path is None:
        import groundingdino

        package_root = Path(groundingdino.__file__).resolve().parent
        config_path = package_root / "config" / DEFAULT_CONFIG_NAME

    if checkpoint_path is None:
        cache_dir = Path(
            os.environ.get("GROUNDING_DINO_CACHE", Path.home() / ".cache" / "groundingdino")
        )
        checkpoint_path = cache_dir / "groundingdino_swint_ogc.pth"
        ensure_checkpoint(checkpoint_path, hf_token=hf_token)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    return str(config_path), str(checkpoint_path)


def parse_args() -> argparse.Namespace:
    paths = default_paths()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, default=paths["seed_dir"], help="Directory of positive seed images.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Comma-separated prompt passed to Grounding DINO.",
    )
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THR)
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THR)
    parser.add_argument("--viz-dir", type=Path, default=paths["viz_dir"])
    parser.add_argument("--json-dir", type=Path, default=paths["json_dir"])
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.bmp")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def find_images(seed_dir: Path, extensions: Iterable[str]) -> List[Path]:
    images: List[Path] = []
    for extension in extensions:
        images.extend(sorted(seed_dir.rglob(f"*{extension}")))
    return sorted({path.resolve() for path in images})


def _image_size(image_source: Any) -> Tuple[int, int]:
    """Return width, height regardless of image container type."""
    if isinstance(image_source, Image.Image):
        return image_source.size
    if np is not None and isinstance(image_source, np.ndarray):
        height, width = image_source.shape[0], image_source.shape[1]
        return int(width), int(height)
    if hasattr(image_source, "shape") and len(image_source.shape) >= 2:
        height, width = image_source.shape[0], image_source.shape[1]
        return int(width), int(height)
    raise TypeError("Unsupported image type returned by load_image")


def prepare_detection_payload(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: List[str],
    image_size: Tuple[int, int],
    image_path: Path,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Dict[str, object]:
    width, height = image_size
    boxes = boxes.detach().cpu()
    logits = logits.detach().cpu()
    detections = []
    for box, score, phrase in zip(boxes, logits.sigmoid(), phrases):
        x1, y1, x2, y2 = map(float, box.tolist())
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": "xyxy_rel",
                "score": float(score.item()),
                "phrase": phrase,
            }
        )

    return {
        "image_path": str(image_path.resolve()),
        "image_name": image_path.name,
        "image_size": {"width": width, "height": height},
        "prompt": prompt,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "detections": detections,
    }


def run_inference(args: argparse.Namespace) -> None:
    LOGGER.info("Loading Grounding DINO model on %s", args.device)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    config_path, checkpoint_path = resolve_model_paths(
        args.config_path, args.checkpoint_path, hf_token=hf_token
    )
    model = load_model(config_path, checkpoint_path, device=args.device)

    args.viz_dir.mkdir(parents=True, exist_ok=True)
    args.json_dir.mkdir(parents=True, exist_ok=True)

    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in args.extensions.split(",")]
    images = find_images(args.seed_dir, extensions)
    if args.max_images is not None:
        images = images[: args.max_images]

    if not images:
        LOGGER.warning("No images found under %s", args.seed_dir)
        return

    LOGGER.info("Running Grounding DINO on %d images", len(images))

    for image_path in tqdm(images, desc="GroundingDINO"):
        json_path = args.json_dir / f"{image_path.stem}.json"
        viz_path = args.viz_dir / f"{image_path.stem}.jpg"

        if json_path.exists() and viz_path.exists() and not args.overwrite:
            continue

        image_source, image = load_image(str(image_path))
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        width, height = _image_size(image_source)
        payload = prepare_detection_payload(
            boxes=boxes,
            logits=logits,
            phrases=phrases,
            image_size=(width, height),
            image_path=image_path,
            prompt=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

        annotated_frame = annotate(
            image_source=image_source.copy() if isinstance(image_source, Image.Image) else image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases,
        )
        if isinstance(annotated_frame, Image.Image):
            annotated_frame.save(viz_path)
        else:
            Image.fromarray(annotated_frame).save(viz_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
