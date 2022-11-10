"""
This script provides a local test routine so you can verify the algorithm works before pushing it to evaluation.

It runs your detector on several local images and verify whether they have obvious issues, e.g:
    - Fail to start
    - Wrong output format

It also prints out the runtime for the algorithms for your references.


The participants are expected to implement a face forgery detector class. The sample detector illustrates the interface.
Do not modify other part of the evaluation toolkit otherwise the evaluation will fail.

Author: Yuanjun Xiong, Zhengkui Guo, Yuanhan Zhang
Contact: celebaspoof@gmail.com

CelebA-Spoof 
"""

from tsn_predict import TSNPredictor as CelebASpoofDetector
import time
import sys
import logging
from video_dataset import VideoFrameDataset
import os
import numpy as np
from client import get_image, verify_output
from torchvision import transforms as T
import torch.utils.data as data
import csv

# Folder in which all videos lie in a specific structure
root = os.path.join(os.getcwd(), 'zaloai/train/videos')
# A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)
annotation_file = os.path.join(
    root.replace('videos', ''), 'val_annotations.txt')

logging.basicConfig(level=logging.INFO)
val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    # T.Normalize(
    #     mean=(0.5, 0.5, 0.5),
    #     std=(0.5, 0.5, 0.5)
    # ),

])

dataset = VideoFrameDataset(
    root_path=root,
    annotationfile_path=annotation_file,
    num_segments=4,
    frames_per_segment=1,
    imagefile_template='img_{:05d}.jpg',
    transform=val_transform,
    test_mode=True
)
val_loader = data.DataLoader(
    dataset, batch_size=4, num_workers=4, shuffle=False)


def run_test(detector_class, image_iter):
    """
    In this function we create the detector instance. And evaluate the wall time for performing CelebASpoofDetector.
    """

    # initialize the detector
    logging.info("Initializing detector.")
    try:
        detector = detector_class()
    except:
        # send errors to the eval frontend
        raise
    logging.info("Detector initialized.")

    # run the images one-by-one and get runtime
    eval_cnt = 0
    result = {}

    logging.info("Starting runtime evaluation")
    for i, sample in enumerate(dataset):
        time_before = time.time()
        prob = detector.predict(sample[0])
        output_probs = 0 if float(prob[:, 1]) < 0.7 else 1
        video_id = sample[2].rsplit('/', 1)[1]+'.mp4'
        result[video_id] = output_probs
        # break
        # if i == 5:
        #     break

    print(result)
    with open('predict.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(('fname', 'liveness_score'))
        for key, value in result.items():
            writer.writerow([key, value])

    logging.info("""
    ================================================================================
    All images finished, showing verification info below:
    ================================================================================
    """)

    # verify the algorithm output
    # verify_output(output_probs)


if __name__ == '__main__':
    celebA_spoof_image_iter = get_image()
    run_test(CelebASpoofDetector, celebA_spoof_image_iter)
