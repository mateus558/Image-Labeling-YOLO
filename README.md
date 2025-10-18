# YOLO Labeling Toolkit

Semi-automatic pipeline for bootstrapping and reviewing bounding-box annotations for arbitrary objects. It pairs Grounding DINO for open‑vocabulary proposal generation with utilities that convert detections into YOLOv8‑compatible datasets and a lightweight Tkinter GUI for manual review.

## Repository Layout

```
seed_images/
  pos/                # images that should contain your target objects
  neg/                # negative images without the target objects
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
  - Place positive samples that contain your target objects under `seed_images/pos`.
  - Place negative samples (no target objects) under `seed_images/neg`.

2. **Run Grounding DINO bootstrap**

   ```bash
   python src/bootstrap_gdino.py \
     --seed-dir seed_images/pos \
     --prompt "screen, monitor, gauge, dial, panel"  # customize to your objects
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

  Launch the GUI:

  ```bash
  python src/review_gui.py
  ```

   - Arrow keys or N/P to navigate.
   - Add Box to draw; drag handles to resize.
   - Class dropdown or 0–9 keys to set/apply class labels.
   - Delete Selected removes highlighted boxes; Save writes YOLO files.

5. **Train YOLOv8**
     - Use the provided helper script to auto-split, write `data.yaml`, and launch Ultralytics training:

     ```bash
     python src/train_yolo.py \
       --model yolov8n.pt \
       --epochs 50 \
       --imgsz 640 \
       --batch 16 \
       --val-frac 0.1
     ```

     - Assumes images live in `yolo_dataset/images/train` and labels in `yolo_dataset/labels/train`.
  - Optionally create `yolo_dataset/classes.txt` with one class name per line; if missing, the code defaults to a single class (currently `display`). Create this file to customize your class names for your use case.
     - Best weights will be saved under `runs/train/<run-name>/weights/best.pt`.

    Offline/proxy environments:

    - If pretrained weights fail to download (SSL/proxy), add `--offline` to train from the architecture YAML instead of `.pt` weights:

    ```bash
    python src/train_yolo.py --offline
    ```
    - Alternatively, manually download a weights file (e.g., `yolov8n.pt`) and pass its local path to `--model`.

## Tips

- Use the `--max-images` flag in `bootstrap_gdino.py` for quick smoke tests before full runs.
- The default Grounding DINO checkpoint downloads automatically to `~/.cache/groundingdino/`; override with `--checkpoint-path` if you maintain your own weights.
- Re-run `json_to_yolo.py` whenever you tweak JSON thresholds; it overwrites labels and re-copies images when `--overwrite` is provided.
- Visualizations under `outputs/viz` are great for batch QA or for sharing quick progress snapshots with stakeholders.

## Adapting to your use case

- Prompts: Change `--prompt` in `bootstrap_gdino.py` to match your objects (comma-separated list of synonyms helps).
- Classes: Create `yolo_dataset/classes.txt` (one class per line) to control label names and `nc` in `data.yaml`.
- Thresholds: Tweak confidence/IoU parameters in `json_to_yolo.py` to filter or merge detections for your data.

## Development

- Install dev extras and run tests:

```bash
pip install -e .[dev]
pytest -q
```

CI runs basic tests on GitHub Actions for pull requests.
