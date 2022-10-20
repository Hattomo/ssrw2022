# -*- coding: utf-8 -*-

import random

import torch
import torchvision.transforms as transforms
import torchaudio
import torchvision
# import torchvi

class DataTransform:

    def train_img_transform(self, x, opts):
        x = x.to(opts.device)
        trans = torch.nn.Sequential(
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.AugMix(),
            # transforms.Resize((224, 224)),
            # transforms.Grayscale(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0], std=[1]),
        ).to(torch.device(opts.device))
        return trans(x)

    def base_img_transform(self, x, opts):
        x = x.to(opts.device)
        trans = torch.nn.Sequential(
            # transforms.Resize((224, 224)),
            # transforms.Grayscale(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0], std=[1]),
        ).to(torch.device(opts.device))
        return trans(x)

    
