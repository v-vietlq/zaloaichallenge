from PIL import Image
import sys
import numpy as np
import torchvision
import torch

# from models import AENet

from models import fasmodel, DeepPixBis


from detector import CelebASpoofDetector


def pretrain(model, state_dict):
    own_state = model.state_dict()

    for name, param in state_dict.items():
        realname = name.replace('module.', '')
        if realname in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[realname].copy_(param)
            except:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(realname, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")


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


class TSNPredictor(CelebASpoofDetector):

    def __init__(self):
        self.num_class = 2
        # self.net = AENet(num_classes=self.num_class)
        self.net = DeepPixBis(encoder_name='resnet18',
                              num_classes=self.num_class)

        self.net = load_model(
            self.net, './weights/deeppixel.ckpt')
        # checkpoint = torch.load('./ckpt_iter.pth.tar',
        #                         map_location=torch.device('cpu'))

        # pretrain(self.net, checkpoint['state_dict'])

        self.new_width = self.new_height = 224

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.new_width, self.new_height)),
            torchvision.transforms.ToTensor(),
        ])

        self.net.to(torch.device('cpu'))
        self.net.eval()

    def preprocess_data(self, image):
        processed_data = Image.fromarray(image)
        processed_data = self.transform(processed_data)
        return processed_data

    def eval_image(self, image):
        # data = torch.stack(image, dim=0)
        channel = 3
        image = image.unsqueeze(0)
        # input_var = data.view(-1, channel, data.size(2), data.size(3))
        with torch.no_grad():
            out_map, rst = self.net(image)
            rst = rst.detach()
            out_map = out_map.detach()
        return rst.reshape(-1, self.num_class), out_map.reshape(14, 14)

    def predict(self, images):
        # real_data = []
        # for image in images:
        #     # data = self.preprocess_data(image)
        #     real_data.append(image)
        rst, out_map = self.eval_image(images)
        rst = torch.nn.functional.softmax(
            rst, dim=1).cpu().numpy().copy()
        probability = np.array(rst)
        return probability, np.mean(np.array(out_map))
