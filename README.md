# Gym Display Bootstrap

Semi-automatic pipeline for bootstrapping bounding-box annotations of exercise machine displays. The workflow combines Grounding DINO for open-vocabulary detection with utilities that convert detections into YOLOv8-compatible datasets and a lightweight GUI for manual review.

## Repository Layout

```
seed_images/
  pos/                # images that should contain exercise-machine displays
  neg/                # negative images without displays
src/
  bootstrap_gdino.py  # run Grounding DINO and dump raw detections + visualizations
  json_to_yolo.py     # convert Grounding DINO JSON into YOLO label files
  review_gui.py       # Tkinter GUI to inspect and edit YOLO labels
outputs/
  viz/                # side-by-side visualizations for quick QA
  auto_labels/        # raw JSON detection dumps from Grounding DINO
yolo_dataset/
  images/train/       # final training images copied here
  labels/train/       # YOLO-format label files
```

## Prerequisites

- Python 3.11
- PyTorch 2.4 with CUDA 12.x support (falls back to CPU if CUDA unavailable)
- GroundingDINO (`groundingdino-py`), Ultralytics ≥ 8.2, OpenCV, tqdm, NumPy, Pillow, Matplotlib
- Optional for the reviewer GUI: Tkinter (bundled with most CPython builds) or PyQt5 if you prefer to migrate the view later.

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

1. **Seed your images**
   - Place positive samples that contain displays under `seed_images/pos`.
   - Place negative samples (no displays) under `seed_images/neg`.

2. **Run Grounding DINO bootstrap**

   ```bash
   python src/bootstrap_gdino.py \
     --seed-dir seed_images/pos \
     --prompt "exercise machine display, treadmill console, rowing machine monitor, bike computer, elliptical console"
   ```

   - Raw detections land in `outputs/auto_labels/<image>.json`.
   - Visualizations with boxes are saved under `outputs/viz/<image>.jpg`.
   - If checkpoint downloads are blocked (corporate SSL, gated repo), pass a Hugging Face token with `--hf-token $HF_TOKEN` or set `HF_TOKEN`/`HUGGINGFACE_TOKEN`. Alternatively download `groundingdino_swint_ogc.pth` manually and point `--checkpoint-path` at it.

3. **Convert JSON to YOLO labels**

   ```bash
   python src/json_to_yolo.py
   ```

   - Filters out weak detections (`score < 0.30` or <1% area).
   - Merges overlapping proposals (`IoU > 0.6`).
   - Copies all images (positives and negatives) to `yolo_dataset/images/train/` and writes labels to `yolo_dataset/labels/train/`.

4. **Review & correct** (optional but recommended)

   ```bash
   python src/review_gui.py
   ```

   - Arrow keys navigate images; **Add Box** lets you draw new boxes with two clicks.
   - **Delete Selected** removes highlighted boxes.
   - **Save** writes the corresponding YOLO label file.

5. **Train YOLOv8**
   - Create a standard Ultralytics dataset YAML that points at `yolo_dataset/images/train` and `yolo_dataset/labels/train`.
   - Kick off training with `yolo detect train ...`.

## Tips

- Use the `--max-images` flag in `bootstrap_gdino.py` for quick smoke tests before full runs.
- The default Grounding DINO checkpoint downloads automatically to `~/.cache/groundingdino/`; override with `--checkpoint-path` if you maintain your own weights.
- Re-run `json_to_yolo.py` whenever you tweak JSON thresholds; it overwrites labels and re-copies images when `--overwrite` is provided.
- Visualizations under `outputs/viz` are great for batch QA or for sharing quick progress snapshots with stakeholders.
