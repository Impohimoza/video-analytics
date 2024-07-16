from typing import Any

import torch
from torchvision import transforms
import numpy as np

from .handler import Handler
from .models import Net, ImageDetection


class SimilarityModelHandler(Handler):
    """Обработчик отвечает за инференс модели подобия"""
    def __init__(self) -> None:
        self.model: Net = None

    def on_start(self) -> Any:
        """Метод для инициализации компонента"""
        self.model: Net = Net()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(
            r"models\trained_model.pth",
            map_location=device
        ))

    def handle(self, img_detect: ImageDetection) -> Any:
        """Метод для нахождения embeddings для всех блюд на изображении"""
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        for detection in img_detect.detections:
            x1, y1, x2, y2 = detection.absolute_box
            img: np.ndarray = img_detect.img[y1:y2, x1:x2]

            img_tensor: torch.Tensor = transform(img).unsqueeze(0)
            detection.vector_img = self.model(img_tensor)
        return img_detect

    def on_exit(self) -> None:
        """Метод для освобождения инициализированных ресурсов"""
        del self.model
