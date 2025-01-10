from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import lightning as L
from torchmetrics.classification import Accuracy

class MNISTModule(L.LightningModule):
    def __init__(self):
        super(MNISTModule, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            # fully connected layer, output 10 classes
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return logits, loss

    def validation_step(self, *args, **kwargs):
        return self.training_step(*args, **kwargs)


class LogisticRegression(L.LightningModule):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor):
        return self.model(x.view(-1, 28 * 28))
    
    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return logits, loss

    def validation_step(self, *args, **kwargs):
        return self.training_step(*args, **kwargs)


class CIFAR10Module(L.LightningModule):
    def __init__(self):
        super(CIFAR10Module, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return logits, loss
    
    def validation_step(self, *args, **kwargs):
        return self.training_step(*args, **kwargs)


class LitResnet(L.LightningModule):
    def __init__(self, model_name, n_class=10, lr=0.1, weight_decay=0.0, optimizer="sgd", min_lr=1e-3, max_epochs=500):
        super(LitResnet, self).__init__()
        self.model = models.__dict__[model_name](num_classes=n_class, norm_layer=partial(nn.BatchNorm2d, track_running_stats=False)) 
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.accuracy_train = Accuracy(num_classes=10, task="multiclass")
        self.accuracy_val = Accuracy(num_classes=10, task="multiclass")
        self.optimizer = optimizer
        self.max_epochs = max_epochs

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.accuracy_train(logits.argmax(-1), y)
        self.log('train_acc_step', self.accuracy_train, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.accuracy_val(logits.argmax(-1), y)
        self.log('val_acc_step', self.accuracy_val, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)


class ImageNetModule(L.LightningModule):
    def __init__(self, model_name, blurpool=True):
        super(ImageNetModule, self).__init__()
        if model_name.startswith('resnet'):
            self.model = models.__dict__[model_name](num_classes=1000, norm_layer=partial(nn.BatchNorm2d, track_running_stats=False))
        else:
            self.model = models.__dict__[model_name](num_classes=1000)
        self.model = models.__dict__[model_name](num_classes=1000)
        if blurpool: 
            apply_blurpool(self.model)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.accuracy_train = Accuracy(num_classes=1000, task="multiclass")

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.accuracy_train(logits.argmax(-1), y)
        self.log('train_acc_step', self.accuracy_train, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, *args, **kwargs):
        return self.training_step(*args, **kwargs)


class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                            groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)


def apply_blurpool(model: torch.nn.Module):
    for (name, child) in model.named_children():
        if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
            setattr(model, name, BlurPoolConv2d(child))
        else: apply_blurpool(child)