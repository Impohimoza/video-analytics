from typing import Tuple

import cv2
import numpy as np

from .handler import Handler


class PreProcessor(Handler):
    """Обработчик изображений"""
    def __init__(
        self,
        size: Tuple[int, int] = (600, 600),
        scalefactor: float = 1,
        swapRB: bool = True,
        ddepth: int = cv2.CV_32F
    ) -> None:
        self.__size = size
        self.__scalefactor = scalefactor
        self.__swapRB = swapRB
        self.__ddepth = ddepth

    def handle(self, img: np.ndarray, *args) -> np.ndarray:
        frame: np.ndarray = cv2.dnn.blobFromImage(
            img,
            scalefactor=self.__scalefactor,
            size=self.__size,
            swapRB=self.__swapRB,
            crop=False,
            ddepth=self.__ddepth
        )

        return frame[0]
