import argparse
from typing import List

from handlers.handler import Handler
from handlers.image_reader import ImageReader
from handlers.image_preprocessor import PreProcessor
from handlers.detector import Detector
from handlers.post_processor import PostProcessor
from handlers.painter import Painter
from handlers.visualizer import Visualizer
from handlers.similarity_model import SimilarityModelHandler
from handlers.embedding_lookup import EmbeddingLookupHandler
from analytics import Analytics


def main(img: str, etalon_embedding: str, etalon_type: str, confidence: float):
    handlers: List[Handler] = [
        ImageReader(),
        PreProcessor(),
        Detector(),
        PostProcessor(confidence=confidence),
        PreProcessor(scalefactor=1/255.0, size=(128, 128), swapRB=True),
        SimilarityModelHandler(),
        EmbeddingLookupHandler(etalon_embedding, etalon_type),
        Painter(),
        Visualizer()
        ]
    analytic: Analytics = Analytics(handlers)
    analytic.on_start()
    analytic.process_frame(img)
    analytic.on_exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video-analytics')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument(
        'etalon_embedding',
        type=str,
        help='Path to the etalon embeddings'
    )
    parser.add_argument(
        'etalon_type',
        type=str,
        help='Path to the etalon type'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detection'
    )

    args = parser.parse_args()
    main(
        args.image_path,
        args.etalon_embedding,
        args.etalon_type,
        args.confidence
    )
