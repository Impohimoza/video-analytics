from typing import Any

import torch
from torchvision import transforms

from .handler import Handler
from .models import Net, ImageDetection


class SimilarityModelHandler(Handler):
    def __init__(self) -> None:
        self.model = None

    def on_start(self) -> Any:
        self.model = Net()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(
            r"models\trained_model.pth",
            map_location=device
            ))

    def handle(self, img_detect: ImageDetection) -> Any:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(img_detect.img).unsqueeze(0)
        img_detect.vector_img = self.model(img_tensor)
        return img_detect

    def on_exit(self) -> None:
        del self.model
