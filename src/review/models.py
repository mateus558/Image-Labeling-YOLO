from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabelBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def clamp(self) -> "LabelBox":
        self.x_center = min(max(self.x_center, 0.0), 1.0)
        self.y_center = min(max(self.y_center, 0.0), 1.0)
        self.width = min(max(self.width, 0.0), 1.0)
        self.height = min(max(self.height, 0.0), 1.0)
        return self

    def as_line(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
