from typing import Tuple

import cv2
import numpy as np
from torchvision import transforms

from .handler import Handler
from .models import ImageDetection


class PreProcessor(Handler):
    """Обработчик изображений"""
    def __init__(
        self,
        size: Tuple[int, int] = (600, 600),
        scalefactor: float = 1,
        swapRB: bool = True,
        ddepth: int = cv2.CV_32F,
        mean: Tuple[int, ...] = 0
    ) -> None:
        self.__size = size
        self.__scalefactor = scalefactor
        self.__swapRB = swapRB
        self.__ddepth = ddepth
        self.__mean = mean

    @staticmethod
    def preprocessing(
        img: np.ndarray,
        size: Tuple[int, int],
        scalefactor: float,
        swapRB: bool,
        ddepth: int,
        mean: Tuple[int, ...]
    ) -> np.ndarray:
        frame: np.ndarray = cv2.dnn.blobFromImage(
                img,
                scalefactor=scalefactor,
                size=size,
                swapRB=swapRB,
                mean=mean,
                crop=False,
                ddepth=ddepth
            )
        return frame[0]

    def handle(self, img_detect: ImageDetection, *args) -> np.ndarray:
        if img_detect.detections is None:
            frame: np.ndarray = self.preprocessing(
                img_detect.img,
                self.__size,
                self.__scalefactor,
                self.__swapRB,
                self.__ddepth,
                self.__mean
            )
            return frame
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            for detection in img_detect.detections:
                frame: np.ndarray = self.preprocessing(
                    detection.crop,
                    self.__size,
                    self.__scalefactor,
                    self.__swapRB,
                    self.__ddepth,
                    self.__mean
                )
                frame = np.transpose(frame, (1, 2, 0))
                detection.crop = transform(frame).unsqueeze(0)
            return img_detect
