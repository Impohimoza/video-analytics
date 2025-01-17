import numpy as np
import cv2

from .handler import Handler


class Visualizer(Handler):
    """Обработчик для визуализации изображения """
    def handle(self, img: np.ndarray) -> np.ndarray:
        cv2.imshow('Inference', img)
        # Логика показа изображения 10 секунд или до нажатия 'q'
        start_time = cv2.getTickCount()
        while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 10:
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        cv2.destroyAllWindows()
