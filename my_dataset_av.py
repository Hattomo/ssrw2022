# -*- coding: utf-8 -*-

import glob
import re
import random
import os
import time

import pandas as pd
from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
import torchvision.transforms as transforms
import time

class ROHANDataset(data.Dataset):

    def __init__(self, image_path, csv_path, labels, data_size, opts, mode, transform=None) -> None:
        super().__init__()
        self.images = []
        self.start_point, self.end_point = data_size
        self.labels = labels
        self.transform = transform
        self.opts = opts
        self.dlib = []
        self.mode = mode

        self.images = sorted(glob.glob(image_path))[self.start_point:self.end_point]
        dlibs = sorted(glob.glob(csv_path))[self.start_point:self.end_point]

        for dlib in tqdm(dlibs):
            self.dlib.append(torch.tensor(pd.read_csv(dlib, header=None, encoding='utf-8').values)[:, :-1])

    def get_time_rand_mask(self, high) -> zip:
        rand = random.randint(0, 3)
        start = torch.randint(size=(rand,), high=high)
        mask = torch.randint(size=(rand,), high=25)
        return zip(start, mask)

    def get_video_rand_mask(self, high) -> zip:
        rand = random.randint(0, 3)
        start_h = torch.randint(size=(rand,), high=high)
        start_w = torch.randint(size=(rand,), high=high)
        mask_h = torch.randint(size=(rand,), high=25)
        mask_w = torch.randint(size=(rand,), high=25)
        return zip(start_h, start_w, mask_h, mask_w)

    def specagu(self, video: torch.tensor, dlib: torch.tensor):
        time_mask = self.get_time_rand_mask(video.size(0))
        feature_mask = self.get_video_rand_mask(video.size(2))
        for start, band in time_mask:
            video[start:start + band, :, :, :] = 0
            dlib[start:start + band, :] = 0
        for start_h, start_w, mask_h, mask_w in feature_mask:
            video[:, :, start_h:start_h + mask_h, start_w:start_w + mask_w] = 0
        return video, dlib

    def __getitem__(self, index: int):
        video = torch.load(self.images[index])
        # video = torchvision.io.read_video(self.images[index], pts_unit='sec', output_format="TCHW")[0]
        video = self.transform(video, self.opts)
        dlib = torch.unsqueeze(self.dlib[index], 0)
        dlib = transforms.Normalize(mean=[0], std=[1])(dlib)
        dlib = torch.squeeze(dlib, 0)
        if self.mode:
            video, dlib = self.specagu(video, dlib)
        return video, torch.tensor(self.labels[index]), video.size()[0], len(self.labels[index]), dlib

    def __len__(self) -> int:
        return len(self.labels)

class MyCollator(object):

    def __init__(self, phones: list):
        self.phones = phones

    def __call__(self, batch):
        inputs, targets, input_lengths, target_lengths, dlib = list(zip(*batch))
        dlib = rnn.pad_sequence(dlib, batch_first=True)
        inputs = rnn.pad_sequence(inputs, batch_first=True)
        targets = rnn.pad_sequence(targets, batch_first=True, padding_value=self.phones.index('_'))
        return inputs, targets, torch.tensor(input_lengths), torch.tensor(target_lengths), dlib
