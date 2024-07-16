from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch import Tensor


@dataclass
class Detection:
    absolute_box: Tuple[int, int, int, int]
    relative_box: Tuple[float, float, float, float]
    score: float
    label_as_int: int
    label_as_str: str
    vector_img: Tensor = None
    type: str = None
    etalon_score: float = 0.0


@dataclass
class ImageDetection:
    """Класс для изображений и Bounding boxes"""
    img: np.ndarray
    detections: List[Detection]


@dataclass
class EtalonVector:
    """Класс для эталона и его вектора"""
    label: str
    vector: Tensor


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
