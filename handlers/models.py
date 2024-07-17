from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


@dataclass
class Bowl:
    absolute_box: Tuple[int, int, int, int]
    relative_box: Tuple[float, float, float, float]
    crop: np.ndarray
    score: float
    label_as_int: int
    label_as_str: str
    embedding: np.ndarray = None
    type: str = None
    etalon_score: float = 0.0


@dataclass
class ImageDetection:
    """Класс для изображений и Bounding boxes"""
    img: np.ndarray
    detections: List[Bowl] = None


@dataclass
class EtalonVector:
    """Класс для эталона и его вектора"""
    label: str
    vector: np.ndarray
