from pathlib import Path

from review.io import read_yolo_labels, write_yolo_labels, read_class_names_from_yaml
from review.models import LabelBox


def test_write_and_read_yolo_labels(tmp_path: Path) -> None:
    boxes = [
        LabelBox(class_id=0, x_center=0.5, y_center=0.5, width=0.25, height=0.1),
        LabelBox(class_id=1, x_center=0.1, y_center=0.9, width=0.2, height=0.2),
    ]
    label_path = tmp_path / "sample.txt"
    write_yolo_labels(label_path, boxes)
    assert label_path.exists()
    loaded = read_yolo_labels(label_path)
    assert len(loaded) == 2
    assert loaded[0].class_id == 0
    assert abs(loaded[0].x_center - 0.5) < 1e-6
    assert loaded[1].class_id == 1


def test_read_class_names_from_yaml_names_list(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        """
names:
  - cat
  - dog
train: dummy
        """.strip()
    )
    names = read_class_names_from_yaml(tmp_path)
    assert names == ["cat", "dog"]


def test_read_class_names_from_yaml_fallback_nc(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("nc: 3\n")
    names = read_class_names_from_yaml(tmp_path)
    assert names == ["class0", "class1", "class2"]

