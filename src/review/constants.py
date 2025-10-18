"""Shared constants for the review GUI and drawing utilities."""
from __future__ import annotations

# UI constants
CANVAS_MAX_WIDTH = 960
CANVAS_MAX_HEIGHT = 720

# Colors
COLOR_BOX = "#40c9ff"
COLOR_BOX_SELECTED = "#ffcc00"
COLOR_PREVIEW = "#ffcc00"
COLOR_LABEL_BG = "#202020"
COLOR_LABEL_TEXT = "#ffffff"
# Palette used to color boxes per class (cycled if more classes than colors)
CLASS_COLOR_PALETTE = [
	"#40c9ff",  # light blue
	"#66ff8a",  # mint
	"#ffb347",  # orange
	"#ff6b81",  # pink
	"#a29bfe",  # lavender
	"#ffd54f",  # amber
	"#7bed9f",  # green
	"#70a1ff",  # sky blue
	"#ff9551",  # coral
	"#1dd1a1",  # teal
]

# Rendering
BOX_LINE_WIDTH = 2
PREVIEW_DASH = (4, 2)

# Behavior
MIN_BOX_SIZE_NORM = 1e-3
HANDLE_SIZE = 8
