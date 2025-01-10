import math
from pathlib import Path
from typing import List
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import torchvision.transforms.v2 as transforms


def prepare_data(fabric, hparams, master_process):
    if hparams.dataset == "mnist":
        transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
        if master_process:            
            train_set = MNIST(root="data", train=True, transform=transform, download=fabric.is_global_zero) 
            val_set = MNIST(root="data", train=False, transform=transform, download=fabric.is_global_zero)
        fabric.barrier()
        if fabric.global_rank > 0:
            train_set = MNIST(root="data", train=True, transform=transform, download=False) 
            val_set = MNIST(root="data", train=False, transform=transform, download=False)
    elif hparams.dataset == "cifar10" or hparams.dataset == "cifar100":
        if hparams.model == "resnet18" or hparams.model == "resnet50":
            transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32, scale=True),
                        transforms.Normalize(
                            [x / 255.0 for x in [125.3, 123.0, 113.9]], [x / 255.0 for x in [63.0, 62.1, 66.7]],
                        ),
                    ])
            transform_val = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                            [x / 255.0 for x in [125.3, 123.0, 113.9]], [x / 255.0 for x in [63.0, 62.1, 66.7]],
                        ),
            ])
        elif hparams.model == "cnn":
            transform_train = transform_val = transforms.Compose([
                        transforms.ToImage(),
                        transforms.ToDtype(torch.float32, scale=True),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
        elif hparams.model == "vit":
            transform_val = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
                ]
            )
            # For training, we add some augmentation. Networks are too powerful and would overfit.
            transform_train = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
                ]
            )
        if master_process:
            if hparams.dataset == "cifar10":
                train_set = CIFAR10(root='data', train=True, download=True, transform=transform_train)
                val_set = CIFAR10(root="data", train=False, download=True, transform=transform_val)
            elif hparams.dataset == "cifar100":
                train_set = CIFAR100(root='data', train=True, download=True, transform=transform_train)
                val_set = CIFAR100(root="data", train=False, download=True, transform=transform_val)
        fabric.barrier()
        if fabric.global_rank > 0:
            if hparams.dataset == "cifar10":
                train_set = CIFAR10(root='data', train=True, download=False, transform=transform_train)
                val_set = CIFAR10(root="data", train=False, download=False, transform=transform_val)
            elif hparams.dataset == "cifar100":
                train_set = CIFAR100(root='data', train=True, download=False, transform=transform_train)
                val_set = CIFAR100(root="data", train=False, download=False, transform=transform_val)
    elif hparams.dataset == "imagenet":
        train_set, val_set = None, None
        
    micro_batch_size = min(math.ceil(hparams.batch_size / (fabric.world_size * hparams.gradient_accumulation_steps)), 
        hparams.max_test_micro_batch_size if hparams.test else hparams.max_micro_batch_size)
    global_batch_size = hparams.global_batch_sizes[0] if hparams.stagewise_ramp_up else hparams.batch_size
    gradient_accumulation_steps = global_batch_size // (fabric.world_size * micro_batch_size)
    
    assert gradient_accumulation_steps > 0, "global_batch_size must be at least world_size * micro_batch_size"
    assert hparams.gradient_accumulation_steps == gradient_accumulation_steps, "gradient_accumulation_steps must be equal to global_batch_size / (world_size * micro_batch_size)"
    assert micro_batch_size >= 2, "micro batch size must be at least 2"

    # set up dataloaders (for distributed training)
    if hparams.dataset == "imagenet": 
        train_dataloader = get_imagenet_loader(data_pth=Path(hparams.data_dir) / "train_500_0.50_90.ffcv", batch_size=micro_batch_size,
            num_workers=hparams.num_workers, drop_last=True, device=fabric.device, train=True, seed=hparams.seed,
            distributed=True, res=224, in_memory=True, batch_ahead=hparams.batch_ahead
        )        
        val_dataloader = get_imagenet_loader(data_pth=Path(hparams.data_dir) / "val_500_0.50_90.ffcv", batch_size=micro_batch_size,
            num_workers=hparams.num_workers, drop_last=False, device=fabric.device, train=False, seed=hparams.seed,
            distributed=True, res=224, in_memory=True, batch_ahead=hparams.batch_ahead
        )
    else:
        train_dataloader = DataLoader(train_set, pin_memory=torch.cuda.is_available(), shuffle=hparams.shuffle, num_workers=hparams.num_workers, batch_size=micro_batch_size)
        val_dataloader = DataLoader(val_set, pin_memory=torch.cuda.is_available(), shuffle=False, num_workers=hparams.num_workers, batch_size=micro_batch_size)
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

        return train_set, val_set, train_dataloader, val_dataloader


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

def get_imagenet_loader(
    data_pth, batch_size, num_workers, drop_last, device, train, seed,
    distributed=True, res=224, in_memory=True, batch_ahead=4):
    from ffcv.pipeline.operation import Operation
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
    from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
    from ffcv.fields.basics import IntDecoder
    
    if train:
        decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(device, non_blocking=True)
        ]

        order = OrderOption.RANDOM

        loader = Loader(data_pth,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=drop_last,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed,
                        seed=seed,
                        batches_ahead=batch_ahead)
    else:
        cropper = CenterCropRGBImageDecoder((res, res), ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(device,
            non_blocking=True)
        ]

        loader = Loader(data_pth,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed,
                        seed=seed,
                        batches_ahead=batch_ahead)
    return loader
