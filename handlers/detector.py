from typing import Any

import numpy as np
import torch

from .handler import Handler


class Detector(Handler):
    """Обработчик для инференса модели детекции YOLOv5"""
    def __init__(self, model_name: str = 'yolov5s'):
        self.model_name: str = model_name
        self.model: Any = None

    def on_start(self, *args, **kwargs) -> None:
        """Метод для инициализации компонента"""
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            self.model_name,
            pretrained=True
        )

    def handle(self, frame: np.ndarray) -> Any:
        """Метод для обработки инференса модели детекции"""
        if self.model is None:
            raise Exception("The component is not initialized")
        results: Any = self.model(frame)

        return results

    def on_exit(self, *args, **kwargs) -> None:
        """Метод для освобождения инициализированных ресурсов"""
        del self.model
