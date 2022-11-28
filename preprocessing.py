import cv2
import os
import numpy as np
import torch


def preprocess_video(data_path, video_name, device):
    cap = cv2.VideoCapture(os.path.join(data_path, video_name))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_per_segment = 4
    idx_fetch = np.linspace(0, frame_count - 1, frame_per_segment, dtype=int)
    tensor_batch = torch.zeros(len(idx_fetch), 224, 224, 3)
    for i, idx in enumerate(idx_fetch):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor_batch[i] = torch.from_numpy(image).float() / 255.0
        # cv2.imwrite(f'{idx}.jpg', image)
    tensor_batch = tensor_batch.permute(0, 3, 1, 2).to(device)

    return tensor_batch
