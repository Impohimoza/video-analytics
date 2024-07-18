from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models

from .handler import Handler
from .models import ImageDetection


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x


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
        for detection in img_detect.detections:
            result: torch.Tensor = self.model(detection.crop)
            detection.embedding = result.cpu().detach().numpy()
        return img_detect

    def on_exit(self) -> None:
        """Метод для освобождения инициализированных ресурсов"""
        del self.model
