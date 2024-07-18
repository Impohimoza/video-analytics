# video-analytics

## Описание

Этот проект предоставляет пример реализации видеоаналитики для оценки соответствия блюда эталонной сервировке. Обработчики реализованы в виде последовательных компонентов, каждый из которых выполняет свою строго определенную функцию.

## Установка

1. Клонируйте репозиторий:
    ```
    git clone https://github.com/Impohimoza/video-analytics.git
    cd video-analytics
    ```

2. Создайте и активируйте виртуальное окружение:
    ```
    python -m venv venv
    source venv/bin/activate
    pip3 install -U pip
    ```

3. Установите зависимости:
    ```sh
    pip install -r requirements.txt
    ```

## Использование
1. Запуск скрипта `etalon_embedding.py` для нахождения эмбеддингов эталонных блюд и их названия
    ```sh
    python etalon_embedding.py path/to/etalon_dir
    ```

2. Запустите `main.py`:
    ```sh
    python main.py path/to/your/image.jpg path/to/etalon_embedding path/to/etalon_type --confidence 0.5
    # Если не указать --confidence, то значение будет 0.5
    ```

