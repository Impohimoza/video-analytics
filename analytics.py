from typing import Any, List

import numpy as np

from handler import Handler


class Analytics:
    """Этот класс управляет всеми обработчиками
    и реализует общую логику анализа кадров."""
    def __init__(self, handlers: List[Handler]):
        self.__handlers: List[Handler] = handlers

    def add_handler(self, handler: Handler) -> None:
        """Метод для добавления обработчиков"""
        self.__handlers.append(handler)

    def process_frame(self, frame: np.ndarray) -> Any:
        """Метод для обработки кадров"""
        data: Any = frame
        for handler in self.__handlers:
            data = handler.handle(data)
        return data

    def on_start(self) -> None:
        """Метод для инициализации всех обработчиков"""
        for handler in self.__handlers:
            handler.on_start()

    def on_exit(self) -> None:
        """Метод для освобождения инициализированных ресурсов"""
        for handler in self.__handlers:
            handler.on_exit()
