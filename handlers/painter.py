import numpy as np
import cv2

from .handler import Handler
from .post_processor import ImageDetection


class Painter(Handler):
    """Обработчик для нанесения на изображения найденные детекции"""
    def handle(self, detections: ImageDetection) -> np.ndarray:
        img: np.ndarray = detections.img
        for detection in detections.detections:
            x1, y1, x2, y2 = detection.absolute_box
            label: str = detection.label_as_str
            score: float = detection.score

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f'{label} {score:.2%}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        return img
