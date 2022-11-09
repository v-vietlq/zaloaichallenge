import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import argparse
from queue import Queue
import threading


# extract frame from video

def video2rgb(video_filename, out_dir):
    file_template = 'img_{:05d}.jpg'

    reader = cv2.VideoCapture(video_filename)

    sucess, frame = reader.read()

    count = 0
    frame_no = 0

    while sucess:
        if count % 4 == 0:
            out_filepath = os.path.join(
                out_dir, file_template.format(frame_no))
            cv2.imwrite(out_filepath, frame)
            frame_no += 1
        sucess, frame = reader.read()
        count += 1


def process_videofile(video_filename, video_path, rgb_out_path, file_extention: str = '.mp4'):
    filepath = os.path.join(video_path, video_filename)
    video_filename = video_filename.replace(file_extention, '')
    out_dir = os.path.join(rgb_out_path, video_filename)
    os.makedirs(out_dir, exist_ok=True)
    video2rgb(filepath, out_dir)


def thread_job(queue, video_path, rgb_out_path, file_extesion='.webm'):
    while not queue.empty():
        video_filename = queue.get()
        process_videofile(video_filename, video_path,
                          rgb_out_path, file_extention=file_extesion)
        queue.task_done()


if __name__ == "__main__":
    video_path = '/home/vietlq4/Downloads/public_test/public/videos'
    rgb_outpath = 'public_test/videos'
    file_extension = '.mp4'

    video_filenames = os.listdir(video_path)

    queue = Queue()
    [queue.put(video_filename) for video_filename in video_filenames]

    NUM_THREADS = 9
    for i in range(NUM_THREADS):
        worker = threading.Thread(target=thread_job, args=(
            queue, video_path, rgb_outpath, file_extension))
        worker.start()

    print('waiting for all videos to be completed.', queue.qsize(), 'videos')
    print('This can take an hour or two depending on dataset size')
    queue.join()
    print('all done')
