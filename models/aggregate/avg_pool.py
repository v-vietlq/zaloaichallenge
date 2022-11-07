import torch
import torch.nn as nn
import torch.nn.functional as F


class FastAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class Aggregate(nn.Module):
    def __init__(self, sampled_frames=None, nvids=None, args=None):
        super(Aggregate, self).__init__()
        self.clip_length = sampled_frames
        self.nvids = nvids
        self.args = args

    def forward(self, x):
        nvids = x.shape[0] // self.clip_length
        x = x.view((-1, self.clip_length) + x.size()[1:])
        o = x.mean(dim=1)
        return o
