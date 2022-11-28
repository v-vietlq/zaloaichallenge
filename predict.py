import torch
import cv2
import numpy as np
from models import DeepPixBis
import os
import time
from preprocessing import preprocess_video
import torch.nn.functional as F
import csv


def load_model(net, path):
    if path is not None and path.endswith(".ckpt"):
        print(path)
        state_dict = torch.load(path, map_location='cpu')

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        compatible_state_dict = {}
        for k, v in state_dict.items():
            k = k[4:]
            compatible_state_dict[k] = v

        net.load_state_dict(compatible_state_dict)

    return net


def write_predict_file(file_name='./result/jupyter_submission.csv', result=None, mode='time'):
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        if mode == 'time':
            writer.writerow(('fname', 'time'))
        else:
            writer.writerow(('fname', 'liveness_score'))
        for key, value in result.items():
            writer.writerow([key, value])


def run_test(data_path='./data', model=None):
    all_predicted_time = {}
    all_result = {}
    test_cases = sorted(os.listdir(data_path),
                        key=lambda x: int(x.replace('.mp4', '')))
    for file_name in test_cases:
        t1 = time.time()
        input_ = preprocess_video(data_path, file_name, device)
        input_ = input_.unsqueeze(0)
        out_map, out = model(input_)
        out, out_map = out.detach(), out_map.detach()
        out = F.softmax(out, dim=1).detach().cpu().numpy()
        out = float(out[:, 1])
        out_map = np.mean(out_map.numpy())
        final_out = (out + out_map) / 2
        t2 = time.time()
        predicted_time = int(t2*1000 - t1*1000)
        all_result[file_name] = final_out
        all_predicted_time[file_name] = predicted_time

    try:
        if not os.path.exists(os.path.dirname('./result/submission.csv')):
            os.makedirs(os.path.dirname('./result/submission.csv'))
    except OSError as err:
        print(err)

    write_predict_file(file_name='./result/submission.csv',
                       result=all_result, mode='predict')


if __name__ == '__main__':
    model = DeepPixBis(encoder_name='resnet18', num_classes=2, phase='test')
    model = load_model(model, './weights/best-epoch=109-val_acc=0.99.ckpt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    run_test(data_path='./data', model=model)
