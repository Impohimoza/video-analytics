from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


@dataclass
class Detection:
    absolute_box: Tuple[int, int, int, int]
    relative_box: Tuple[float, float, float, float]
    score: float
    label_as_int: int
    label_as_str: str


@dataclass
class ImageDetection:
    """Класс для изображений и Bounding boxes"""
    img: np.ndarray
    detections: List[Detection]