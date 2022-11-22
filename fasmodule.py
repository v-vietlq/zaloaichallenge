import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
from pytorch_lightning.core.lightning import LightningModule
import timm
from utils.loss import *
from models import *
from torchmetrics import Accuracy
from typing import Union
import muar
from muar.utils import tensor2pil
from muar.augmentations import BatchRandAugment, MuAugment


def calc_acc(pred, target):
    equal = torch.mean(pred.eq(target).type(torch.FloatTensor))
    return equal.item()


class FasModule(LightningModule):
    def __init__(self,
                 n_tfms: int = 3,
                 magn: int = 3,
                 n_compositions: int = 4,
                 n_selected: int = 2,
                 main_opt=None,
                 val_opt=None) -> None:
        super(FasModule, self).__init__()
        if val_opt is None:
            self.test_opt = main_opt
            self.save_hyperparameters(vars(main_opt))
            self.net = fasmodel(main_opt.backbone, num_classes=2)
            return

        self.train_opt = main_opt
        self.save_hyperparameters()

        if self.train_opt.model == 'deeppixel':
            self.loss_pixel = nn.BCELoss()
            self.contrastive_loss = ContrastiveLoss(margin=1.0)
            self.net = DeepPixBis(main_opt.backbone, num_classes=2)
        else:
            self.net = fasmodel(main_opt.backbone, num_classes=2)
        self.out_weights = [1]

        self.criterion = []
        for loss_name in self.train_opt.loss:
            if loss_name == "focal":
                self.criterion += [(loss_name, FocalLoss())]

        self.val_acc = Accuracy()
        self.n_tfms = n_tfms
        self.magn = magn
        self.n_compositions = n_compositions
        self.n_selected = n_selected

    def forward_one(self, x):
        return self.net(x)

    def forward(self, x, y):
        return self.net(x), self.net(y)

    def training_step(self, batch, batch_idx):

        video1, video2, simarity_label = batch
        image, label, path, = video1
        image1, label1, path1 = video2

        mean, std = image.mean((0, 2, 3)), image.std((0, 2, 3))
        rand_augment = BatchRandAugment(self.n_tfms, self.magn, mean, std)
        self.mu_transform = MuAugment(
            rand_augment, self.n_compositions, self.n_selected)
        self.mu_transform.setup(self)
        image, label = self.mu_transform(image, label)
        image1, label1 = self.mu_transform(image1, label1)

        total_loss = 0

        if self.train_opt.model == 'deeppixel':

            # gen label for deeppixel
            map_label = torch.abs(torch.sub(label, 0.01)).view(-1, 1)

            ones = torch.ones(image.size(0), 196).cuda()
            label_map = ones * map_label.expand_as(ones)
            label_map = label_map.view(image.size(0), 14, 14)

            # forward model
            o, o1 = self(image, image1)
            out_map, outputs, out_feat = o
            _, _, out1_feat = o1

            loss_pixel = self.loss_pixel(out_map, label_map)

            out_feat = F.normalize(out_feat, dim=1)
            out1_feat = F.normalize(out1_feat, dim=1)

            loss_contrastive = self.contrastive_loss(
                out_feat, out1_feat, simarity_label)

            total_loss += loss_pixel + loss_contrastive

            self.log('loss_pixel', loss_pixel, on_step=False,
                     on_epoch=True, logger=True)

            self.log('loss_contrastive', loss_contrastive, on_step=False,
                     on_epoch=True, logger=True)

        if len(self.out_weights) == 1:
            outputs = [outputs]

        for loss_name, criteria in self.criterion:
            loss = 0
            for output, weight in zip(outputs, self.out_weights):
                loss = loss + weight*criteria(output, label)

            total_loss += loss

            self.log('loss_' + loss_name, loss, on_step=False,
                     on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        video1, video2, simarity_label = batch
        image, label, path, = video1
        image1, label1, path1 = video2
        # image, label, path = batch
        with torch.no_grad():
            if self.train_opt.model == 'deeppixel':
                out_map, outputs, _ = self.forward_one(image)

            else:
                outputs = self(image)

        if len(self.out_weights) == 1:
            # pred = F.sigmoid(outputs)
            pred = F.softmax(outputs, dim=1).type(torch.FloatTensor)
        label = label.type(torch.LongTensor)

        self.val_acc.update(pred.cpu(), label.cpu())

    def validation_epoch_end(self, outputs):
        val_acc = self.val_acc.compute()

        self.log('val_acc', val_acc, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_acc.reset()
        return val_acc

    def configure_optimizers(self):
        # Create optimizer
        optimizer = None
        if self.train_opt.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.net.parameters(
            ), lr=self.train_opt.lr, momentum=self.train_opt.momentum, weight_decay=self.train_opt.weight_decay)
        elif self.train_opt.optimizer == "adam":
            optimizer = torch.optim.Adam(self.net.parameters(
            ), lr=self.train_opt.lr, weight_decay=self.train_opt.weight_decay)
        elif self.train_opt.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.net.parameters(), lr=self.train_opt.lr)

        # Create learning rate scheduler
        scheduler = None
        if self.train_opt.lr_policy == "exp":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.train_opt.lr_gamma)
        elif self.train_opt.lr_policy == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.train_opt.lr_step, gamma=self.train_opt.lr_gamma)
        elif self.train_opt.lr_policy == "multi_step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.train_opt.lr_milestones, gamma=self.train_opt.lr_gamma)
        elif self.train_opt.lr_policy == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.train_opt.lr, steps_per_epoch=len(self.trainer._data_connector._train_dataloader_source.dataloader()), epochs=self.train_opt.max_epoch,
                                                            pct_start=0.2)
        return [optimizer], [{'scheduler': scheduler, 'name': 'lr'}]
