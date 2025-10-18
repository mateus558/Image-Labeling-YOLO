# Label Review GUI

Lightweight Tkinter application for inspecting and editing YOLO-format bounding boxes. Drop in your dataset, launch the GUI, and iterate on annotations without touching heavyweight training or bootstrapping pipelines.

![GUI demo](example.gif)

Figure: quick demo of the reviewer GUI — click Add Box (A), then click two corners to create a bounding box.

## Repository Layout

```
src/
  review_gui.py       # CLI entry point for the GUI
  review/             # MVC-style modules backing the UI (controller, view, models, utils)
yolo_dataset/
  images/             # images the GUI will load (train/val subfolders optional)
  labels/             # YOLO .txt files, one per image (matching directory structure)
tests/
  *.py                # fast unit tests for core helpers
```

## Prerequisites

- Python 3.11 (3.9+ also works)
- Tkinter (ships with most CPython installers)
- Pillow for image loading and resizing

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. **Organize your dataset**
  - Place images under `yolo_dataset/images/`. Subdirectories are supported; the GUI will recurse one level deep when you pick a folder.
  - Store YOLO `.txt` label files under `yolo_dataset/labels/`, mirroring the image folder structure (e.g., `images/train/img001.jpg` ↔ `labels/train/img001.txt`). Missing label files are treated as unlabeled images.
  - Optionally add `yolo_dataset/data.yaml` to supply human-friendly class names for the dropdown. The helper loaders understand the common Ultralytics format (`names:` list or `nc:` count).

2. **Install dependencies** (once per environment)

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

3. **Launch the reviewer**

  ```bash
  python src/review_gui.py
  ```

  - Use the folder pickers on the right to point at your image and label directories if the defaults do not exist.
  - Arrow keys, `N`, or `P` navigate images; `A` adds a box, `D` deletes selections, `S` saves.
  - Click and drag handles to resize; press digits `0-9` to assign classes quickly.

## Tips

- Press `Esc` to cancel an in-progress add or resize gesture if you mis-click.
- The status bar surfaces all keyboard shortcuts; glance there whenever you forget the workflow.
- Saving happens per image. The app auto-saves when you move between images but you can hit `S` any time for peace of mind.
- Add `yolo_dataset/data.yaml` with `names:` to see friendly class labels instead of numeric IDs.

## Adapting to your use case

- Drop any existing YOLO project into `yolo_dataset/` and it will load instantly.
- If you keep separate `train/`, `val/`, or `test/` folders, switch among them via the folder picker without moving files around.
- Using a different labeling scheme? Edit `data.yaml` (or create one) with your preferred class labels and the GUI will pick them up on next load.

## Development

- Install dev extras and run tests:

```bash
pip install -e .[dev]
pytest -q
```

CI runs basic tests on GitHub Actions for pull requests.
