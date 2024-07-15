import cv2
import numpy as np

from .handler import Handler


class ImageDirectoryReader(Handler):
    def handle(self, image_path: str) -> np.ndarray:
        img: np.ndarray = cv2.imread(r'{}'.format(image_path))
        if img is None:
            raise AttributeError(f"Unable to open image file {image_path}")
        return img
