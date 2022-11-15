import torch
import torch.nn as nn
import torchvision
from aggregate.avg_pool import FastAvgPool2d, Aggregate
from aggregate.transformer_agg import TAggregate
import torch.nn.functional as F
import timm
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
    def __init__(self, encoder_name='vit_small_patch16_224_dino', num_classes=2):
        super(fasmodel, self).__init__()

        self.feature_extractor = timm.create_model(model_name= encoder_name, num_classes = num_classes, pretrained=True)
  

        self.global_pool_layer = FastAvgPool2d(flatten=True)

        # self.feat_in = backbone_filters[encoder_name]
        
        # self.aggregate = Aggregate()

        # self.fc = nn.Linear(self.feat_in[-1], num_classes)
        
        # self.agg =TAggregate(clip_length=4, embed_dim=self.feat_in[-1])
        # self.dec = nn.Conv1d(384, 1, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        x = x.view(batch_size * time_steps, channels, height, width)
        
        o = self.feature_extractor.forward_features(x)
        
   
        # fea = self.feature_extractor.forward_head(fea)

        o = o.view((-1, time_steps) + o.size()[1:])
        o = o.mean(dim=1)
        
        o = self.feature_extractor.forward_head(o)

        # o = self.fc(o)
        

        # o = self.global_pool_layer(fea)
        
        # o = self.agg(o)
        
        

        # o = self.fc(o)

        # o = F.sigmoid(o)

        return o

if __name__ == '__main__':
    x = torch.rand(4,4,3,224,224)
    m = fasmodel()
    y = m(x)
    print(y.shape)
    