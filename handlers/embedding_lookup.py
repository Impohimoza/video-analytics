from typing import List
from pathlib import Path
import os

import cv2
import torch
from torchvision import transforms
import numpy as np

from .handler import Handler
from .models import EtalonVector, Net, ImageDetection


class EmbeddingLookupHandler(Handler):
    """Обработчик для классификации и
    оценивания схожести с эталонной сервировкой"""
    def __init__(self, etalon_dir: str) -> None:
        if not Path(etalon_dir).is_dir:
            raise AttributeError(f'Unable to open dir {etalon_dir}')
        self.__etalon_dir: str = etalon_dir
        self.__etalon_vectors: List[EtalonVector] = []

    def on_start(self) -> None:
        """Метод для инициализации компонента
        загрузки векторов эталонных блюд"""
        net = Net()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.load_state_dict(torch.load(
            r"models\trained_model.pth",
            map_location=device
        ))
        transform = transforms.Compose([
                transforms.ToTensor()
        ])

        for name in os.listdir(self.__etalon_dir):
            img: np.ndarray = cv2.imread(os.path.join(self.__etalon_dir, name))
            img_tensor: torch.Tensor = transform(img).unsqueeze(0)
            vector: torch.Tensor = net(img_tensor)
            self.__etalon_vectors.append(EtalonVector(name, vector))

    @staticmethod
    def cosine_similarity(embedding1, embedding2) -> float:
        """Метод для нахождения Cosine similarity"""
        cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        cos_sim = cos_sim.mean().item() * 100
        return cos_sim

    def handle(self, img_detect: ImageDetection) -> ImageDetection:
        """Метод для классификации и оценки схожести с эталоном"""
        for detection in img_detect.detections:
            for etalon_vec in self.__etalon_vectors:
                score: float = self.cosine_similarity(
                    detection.vector_img,
                    etalon_vec.vector
                )
                if score > detection.etalon_score:
                    detection.type = etalon_vec.label
                    detection.etalon_score = score
        return img_detect

    def on_exit(self, *args, **kwargs) -> None:
        """Метод для освобождения инициализированных ресурсов"""
        self.__etalon_vectors = []
