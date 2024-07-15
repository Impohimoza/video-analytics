from typing import Tuple

import cv2
import numpy as np

from .handler import Handler


class PreProcessor(Handler):
    def __init__(self, size: Tuple[int, int] = (300, 300)) -> None:
        self.__size = size

    def handle(self, img: np.ndarray) -> np.ndarray:
        frame: np.ndarray = cv2.dnn.blobFromImage(
            img,
            scalefactor=1,
            size=self.__size,
            swapRB=True,
            crop=False
        )
        return frame
