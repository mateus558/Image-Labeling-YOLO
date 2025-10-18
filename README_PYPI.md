# image-labeling-yolo

Lightweight Tkinter GUI for reviewing and editing YOLO-format bounding boxes. Point it at your dataset and quickly add, edit, and save labels.

## Features

- Simple, fast GUI for YOLO labels (class id, x_center, y_center, width, height)
- Keyboard-friendly: A add box, D delete, S save, digits 0–9 to set class
- Reads class names from `yolo_dataset/data.yaml` (`names:` list) when present
- Works with images under `yolo_dataset/images/` and matching labels under `yolo_dataset/labels/`

## Install

```bash
pip install image-labeling-yolo
```

Python 3.9+ is supported. Tkinter must be available in your Python installation (most CPython builds include it by default).

## Quick start

1) Prepare folders (minimal structure):

```
yolo_dataset/
  images/
    img001.jpg
    img002.jpg
  labels/
    img001.txt   # optional; created on save if missing
    img002.txt
  data.yaml      # optional; provides class names via `names:`
```

2) Launch the GUI:

```bash
label-review
```

If you prefer, you can also run:

```bash
python -m review_gui
```

3) Use the folder pickers (right panel) to select your `images/` and `labels/` directories if the defaults are not detected.

## Usage notes

- Navigation: Left/Right arrows or N/P
- Add a box: Press A, then click two opposite corners; drag handles to resize
- Delete selection: D
- Save: S (auto-saves when you navigate)
- Set class: Digits 0–9 (or choose from the dropdown)
- Class names: Provide a `yolo_dataset/data.yaml` with a `names:` list to show readable labels; otherwise numeric IDs are shown

## Troubleshooting

- Tkinter missing: On some Linux distributions you may need to install Tk packages (e.g., `sudo apt-get install python3-tk`), or use a Python build that includes Tkinter.
- Headless servers: Run the GUI on a machine with a display or use X forwarding/remote desktop options.

## License

MIT

## Links

- Source and issues: https://github.com/mateus558/image-labeling-yolo