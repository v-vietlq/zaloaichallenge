import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from aggregate.avg_pool import FastAvgPool2d, Aggregate

backbone_filters = {
    # Resnet
    "resnet18": [64, 64, 128, 256, 512],
    "resnet34": [64, 64, 128, 256, 512],
    "resnet50": [64, 256, 512, 1024, 2048],
    "resnet101": [64, 256, 512, 1024, 2048],
    "resnet152": [64, 256, 512, 1024, 2048],

    # MobileNetV2
    "mobilenet_v2": [16, 24, 32, 96, 1280]
}


class DeepPixBis(nn.Module):
    def __init__(self, encoder_name='resnet18', num_classes=2, pretrained=True, phase='train'):
        super(DeepPixBis, self).__init__()
        feature_extractor = getattr(
            torchvision.models, encoder_name)(pretrained=True)
        if encoder_name.startswith('resnet'):
            self.encoders = [
                nn.Sequential(feature_extractor.conv1,
                              feature_extractor.bn1, feature_extractor.relu),
                nn.Sequential(feature_extractor.maxpool,
                              feature_extractor.layer1),
                feature_extractor.layer2,
                feature_extractor.layer3,
                feature_extractor.layer4
            ]
        else:
            raise NotImplementedError('backbone should be resnet')

        self.encoders = nn.ModuleList(self.encoders)

        self.feat_in = backbone_filters[encoder_name]


        self.dec = nn.Conv2d(
            self.feat_in[-2], 1, kernel_size=1, stride=1, padding=0)

        self.fc = nn.Linear(self.feat_in[-1], num_classes)

        self.global_pool_layer = FastAvgPool2d(flatten=True)
        self.phase = phase

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()

        x = x.view(batch_size*time_steps, channels, height, width)

        features = []

        for module in self.encoders:
            x = module(x)
            features += [x]

        x0, x1, x2, x3, feat = features
        print(x0.shape, x1.shape, x2.shape, x3.shape)

        o = self.global_pool_layer(feat)
        o = o.view((-1, time_steps) + o.size()[1:])
        o = o.mean(dim=1)

        out = self.fc(o)

        out_map = self.dec(x3)
        # out_map = self.global_pool_layer(out_map)
        out_map = out_map.view(batch_size, time_steps, 14, 14)
        out_map = out_map.mean(dim=1)
        out_map = F.sigmoid(out_map)

        if self.phase == 'train':
            return out_map, out, o
        else:
            return out_map, out


if __name__ == '__main__':
    x = torch.rand(4, 4, 3, 224, 224)
    m = DeepPixBis()
    y = m(x)
    print(y[0].shape, y[1].shape)
