from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .handler import Handler
from .models import EtalonVector, ImageDetection


class EmbeddingLookupHandler(Handler):
    """Обработчик для классификации и
    оценивания схожести с эталонной сервировкой"""
    def __init__(self, embedding_path: str, type_path: str) -> None:
        if not Path(embedding_path).is_dir or not Path(type_path):
            raise AttributeError(f'Unable to open path {embedding_path}')
        self.__embedding_path: str = embedding_path
        self.__type_path: str = type_path
        self.__etalon_vectors: List[EtalonVector] = []

    def on_start(self) -> None:
        """Метод для загрузки векторов эталонных блюд"""
        etalon_embeddings: np.ndarray = np.load(self.__embedding_path)
        types: pd.DataFrame = pd.read_csv(self.__type_path)

        for i, type in enumerate(types['Type']):
            etalon_vector = EtalonVector(type, etalon_embeddings[i])
            self.__etalon_vectors.append(etalon_vector)

    def handle(self, img_detect: ImageDetection) -> ImageDetection:
        """Метод для классификации и оценки схожести с эталоном"""
        for detection in img_detect.detections:
            for etalon_vec in self.__etalon_vectors:
                score: np.ndarray = cosine_similarity(
                    detection.embedding,
                    etalon_vec.vector
                )
                score = score.item() * 100
                if score > detection.etalon_score:
                    detection.type = etalon_vec.label
                    detection.etalon_score = score
        return img_detect

    def on_exit(self, *args, **kwargs) -> None:
        """Метод для освобождения инициализированных ресурсов"""
        self.__etalon_vectors = []
