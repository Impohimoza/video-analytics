from pathlib import Path

import cv2
import numpy as np

from .handler import Handler
from .models import ImageDetection


class ImageReader(Handler):
    """Обработчик предназначен для чтения изображений"""
    def handle(self, image_path: str) -> np.ndarray:
        if not Path(image_path).is_file():
            raise AttributeError(f"Unable to open image file {image_path}")
        img: np.ndarray = cv2.imread(r'{}'.format(image_path))
        img_detect: ImageDetection = ImageDetection(img)
        return img_detect
