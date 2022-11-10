import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torchvision
import torch.nn as nn
from typing import Optional, Sequence
from torch import Tensor


class AdMSoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30., m=0.4):
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(self.in_features, self.out_features, bias=False)

    def forward(self, x, labels):
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)

        numerator = self.s * \
            (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0)
                         for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + \
            torch.sum(torch.exp(self.s*excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.weight = torch.Tensor([alpha, 1-alpha]).cuda()
        self.nllLoss = nn.NLLLoss(weight=self.weight)
        self.gamma = gamma

    def forward(self, input, target):
        softmax = F.softmax(input, dim=1)
        log_logits = torch.log(softmax)
        fix_weights = (1 - softmax) ** self.gamma
        logits = fix_weights * log_logits
        return self.nllLoss(logits, target)


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets *
                       torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                  self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                              self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                self.loss *= self.asymmetric_w
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss


if __name__ == '__main__':

    in_features = 512
    out_features = 10  # Number of classes

    # Default values recommended by [1]
    criterion = AdMSoftmaxLoss(in_features, out_features, s=30.0, m=0.4)

    # Forward method works similarly to nn.CrossEntropyLoss
    # x of shape (batch_size, in_features), labels of shape (batch_size,)
    # labels should indicate class of each sample, and should be an int, l satisying 0 <= l < out_dim
    input = torch.rand(4, 512, requires_grad=True)
    target = torch.randint(0, 1, (4, 1)).random_(2).type(torch.LongTensor)
    loss = criterion(x, target)
    loss.backward()
