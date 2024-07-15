from typing import Tuple

import cv2
import numpy as np

from .handler import Handler


class PreProcessor(Handler):
    """Обработчик изображений"""
    def __init__(self, size: Tuple[int, int] = (600, 600)) -> None:
        self.__size = size

    def handle(self, img: np.ndarray, *args) -> np.ndarray:
        # frame: np.ndarray = cv2.dnn.blobFromImage(
        #     img,
        #     scalefactor=1,
        #     size=self.__size,
        #     swapRB=True,
        #     crop=False,
        #     mean=(104, 117, 123)
        # )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, self.__size)
        return img
