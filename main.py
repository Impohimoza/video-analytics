import argparse
from typing import Any

from handlers.image_reader import ImageDirectoryReader
from handlers.image_preprocessor import PreProcessor
from analytics import Analytics


def main(img: str):
    handlers = [ImageDirectoryReader(), PreProcessor()]
    analytic = Analytics(handlers)
    result = analytic.process_frame(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video-analytics')
    parser.add_argument('image_path', type=str, help='Path to the image')

    args = parser.parse_args()
    main(args.image_path)
