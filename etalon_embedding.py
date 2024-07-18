from typing import List
from pathlib import Path
import os
import argparse

import numpy as np
import pandas as pd

from handlers.handler import Handler
from handlers.image_reader import ImageReader
from handlers.image_preprocessor import PreProcessor
from handlers.detector import Detector
from handlers.post_processor import PostProcessor
from handlers.similarity_model import SimilarityModelHandler
from handlers.models import ImageDetection
from analytics import Analytics


def main(etalon_dir: str):
    if not Path(etalon_dir).is_dir:
        raise AttributeError(f'Unable to open dir {etalon_dir}')

    handlers: List[Handler] = [
        ImageReader(),
        PreProcessor(),
        Detector(),
        PostProcessor(confidence=0.1),
        PreProcessor(scalefactor=1/255.0, size=(128, 128), swapRB=True),
        SimilarityModelHandler()
    ]

    analytic: Analytics = Analytics(handlers)
    analytic.on_start()

    embeddings: List[np.array] = []
    etalon_type: dict[str, List] = {'Type': []}

    for name in os.listdir(etalon_dir):
        file_path: str = os.path.join(etalon_dir, name)
        # получение embedding
        result: ImageDetection = analytic.process_frame(file_path)
        for detection in result.detections:
            embeddings.append(detection.embedding)
            etalon_type['Type'].append(name[0:-4])

    df = pd.DataFrame(etalon_type)
    csv_file_path = 'etalon_type.csv'
    df.to_csv(csv_file_path, index=True)
    array = np.array(embeddings)
    np.save('etalon', array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video-analytics')
    parser.add_argument(
        'etalon_dir',
        type=str,
        help='Path to the etalon img dir'
    )

    args = parser.parse_args()
    main(args.etalon_dir)
