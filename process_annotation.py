import numpy as np
import os
import pandas as pd

path = '/home/vietlq4/zaloaichallenge/zaloai/train/videos'
annotation_path = '/home/vietlq4/zaloaichallenge/zaloai/train/label.csv'
annotation_out_file = '/home/vietlq4/zaloaichallenge/zaloai/train/annotations.txt'

video_filenames = os.listdir(path)
label_df = None
label2id = {}
if len(annotation_path) != 0:
    label_df = pd.read_csv(annotation_path)

    for idx, row in label_df.iterrows():
        label2id[row['fname'].replace('.mp4', '')] = int(row['liveness_score'])

with open(annotation_out_file, 'w') as f:

    for video_filename in sorted(video_filenames, key= lambda x: int(x)):
        start_frame = 0
        video_path = os.path.join(path, video_filename)
        try:
            num_frames = len(os.listdir(video_path))
            if num_frames == 0:
                print(f'{video_filename}- no frames')
                continue
        except FileNotFoundError as e:
            print(e)
            continue
            
        if len(annotation_path) == 0:
            annotation_string = "{} {} {}\n".format(
                video_filename, start_frame, num_frames - 1)
        else:
            annotation_string = "{} {} {} {}\n".format(
                video_filename, start_frame, num_frames - 1, label2id[video_filename])

        f.write(annotation_string)
