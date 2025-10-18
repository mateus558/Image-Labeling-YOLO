from review.models import LabelBox
from review.canvas_utils import normalized_to_canvas, box_corners_norm


def test_normalized_to_canvas_basic() -> None:
    box = LabelBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.4)
    x1, y1, x2, y2 = normalized_to_canvas(box, 1000, 500)
    # width in px: 0.2 * 1000 = 200; height: 0.4 * 500 = 200
    # centered at (500, 250)
    assert (x2 - x1) == 200
    assert (y2 - y1) == 200
    assert x1 == 400 and x2 == 600
    assert y1 == 150 and y2 == 350


def test_box_corners_norm_round_trip() -> None:
    box = LabelBox(class_id=0, x_center=0.25, y_center=0.25, width=0.2, height=0.1)
    x1, y1, x2, y2 = box_corners_norm(box)
    # Recompute center/size
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    assert abs(xc - box.x_center) < 1e-6
    assert abs(yc - box.y_center) < 1e-6
    assert abs(w - box.width) < 1e-6
    assert abs(h - box.height) < 1e-6

