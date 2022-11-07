import torch
import torch.nn as nn
import torchvision
from aggregate.avg_pool import FastAvgPool2d, Aggregate
import torch.nn.functional as F

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


class fasmodel(nn.Module):
    def __init__(self, encoder_name='resnet18', num_classes=2):
        super(fasmodel, self).__init__()

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

        self.global_pool_layer = FastAvgPool2d(flatten=True)

        self.feat_in = backbone_filters[encoder_name]
        
        self.aggregate = Aggregate()

        self.fc = nn.Linear(self.feat_in[-1], num_classes)
        

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        x = x.view(batch_size * time_steps, channels, height, width)
        features = []
        for module in self.encoders:
            x = module(x)
            features += [x]
        x0, x1, x2, x3, fea = features

        o = self.global_pool_layer(fea)
        
        o = o.view((-1, time_steps) + o.size()[1:])
        o = o.mean(dim=1)
        

        o = self.fc(o)

        return o

if __name__ == '__main__':
    x = torch.rand(4,16,3,224,224)
    m = fasmodel()
    y = m(x)
    print(y.shape)
    