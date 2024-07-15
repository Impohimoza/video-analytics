from dataclasses import dataclass
from typing import Any, Tuple, List

import numpy as np
import cv2

from .handler import Handler


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


class PostProcessor(Handler):
    """Обработчик, который отвечает за парсинг “сырых“ результатов модели детекции"""
    def __init__(self, confidence: float = 0.5) -> None:
        self.confidence: float = confidence

    def handle(self, detection: Any) -> ImageDetection:
        filtered_results = detection.pred[0][detection.pred[0][:, -1] == 45]

        detections: List[Detection] = []
        img: np.ndarray = cv2.cvtColor(detection.ims[0], cv2.COLOR_RGB2BGR)
        for det in filtered_results:
            x1, y1, x2, y2, conf, clss = det
            score: float = conf.item()
            if score < self.confidence:
                continue
            absolute_box: Tuple[int, int, int, int] = (
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            )
            relative_box: Tuple[float, float, float, float] = (
                x1.item() / img.shape[1],
                y1.item() / img.shape[0],
                x2.item() / img.shape[1],
                y2.item() / img.shape[0]
            )
            label_as_int: int = int(clss.item())
            label_as_str: str = 'blow'

            detections.append(Detection(
                absolute_box,
                relative_box,
                score,
                label_as_int,
                label_as_str
            ))
        img_detection = ImageDetection(img, detections)
        return img_detection
