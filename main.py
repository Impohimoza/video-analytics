import argparse

from handlers.image_reader import ImageDirectoryReader
from analytics import Analytics


def main(img: str):
    handlers = [ImageDirectoryReader(),]
    analytic = Analytics(handlers)
    print(type(analytic.process_frame(img)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video-analytics')
    parser.add_argument('image_path', type=str, help='Path to the image')

    args = parser.parse_args()
    main(args.image_path)
