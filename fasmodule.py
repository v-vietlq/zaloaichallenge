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
from models.models import *
from torchmetrics import Accuracy


def calc_acc(pred, target):
    equal = torch.mean(pred.eq(target).type(torch.FloatTensor))
    return equal.item()


class FasModule(LightningModule):
    def __init__(self, main_opt, val_opt=None) -> None:
        super(FasModule, self).__init__()
        if val_opt is None:
            self.test_opt = main_opt
            self.save_hyperparameters(vars(main_opt))
            self.net = fasmodel(main_opt.backbone, num_classes=2)
            return

        self.train_opt = main_opt
        self.save_hyperparameters(vars(main_opt))

        self.net = fasmodel(main_opt.backbone, num_classes=2)
        self.out_weights = [1]

        self.criterion = []
        for loss_name in self.train_opt.loss:
            if loss_name == "focal":
                self.criterion += [(loss_name, nn.CrossEntropyLoss())]

        self.val_acc = Accuracy()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        image, label, path = batch

        outputs = self(image)
        if len(self.out_weights) == 1:
            outputs = [outputs]

        total_loss = 0
        for loss_name, criteria in self.criterion:
            loss = 0
            for output, weight in zip(outputs, self.out_weights):
                loss = loss + weight*criteria(output, label)

            total_loss += loss

            self.log('loss_' + loss_name, loss, on_step=False,
                     on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        image, label, path = batch
        with torch.no_grad():
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
