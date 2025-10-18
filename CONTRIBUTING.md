Contributing
===========

Thanks for your interest in improving this project! Here’s how to get started.

Setup
-----
- Use Python 3.11.
- Create a virtual environment and install dev deps:

```
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Running Tests
-------------
```
pytest -q
```

Local Usage
-----------
- Prepare a dataset under `yolo_dataset/images/train` and `yolo_dataset/labels/train`.
- Run the review GUI:

```
label-review
```

Code Style
----------
- Keep changes minimal and focused.
- Prefer small, pure helpers (like those in `src/review/canvas_utils.py`).
- Use type hints where practical.

Pull Requests
-------------
- Describe the change and rationale.
- Include before/after screenshots for UI changes.
- Add or update tests when touching testable logic.
