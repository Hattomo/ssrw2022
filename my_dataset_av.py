# -*- coding: utf-8 -*-

import glob
import re
import os
import time

import pandas as pd
from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
import torchvision
import time

class ROHANDataset(data.Dataset):

    def __init__(self, image_path, csv_path, labels, data_size, opts, transform=None) -> None:
        super().__init__()
        self.images = []
        self.start_point, self.end_point = data_size
        self.labels = labels
        self.transform = transform
        self.opts = opts
        self.dlib = []

        self.images = sorted(glob.glob(image_path))[self.start_point:self.end_point]
        dlibs = sorted(glob.glob(csv_path))[self.start_point:self.end_point]

        for dlib in tqdm(dlibs):
            self.dlib.append(torch.tensor(pd.read_csv(dlib, header=None, encoding='utf-8').values)[:, :-1])

    def __getitem__(self, index: int):
        video = torch.load(self.images[index])
        # video = torchvision.io.read_video(self.images[index], pts_unit='sec', output_format="TCHW")[0]
        video = self.transform(video, self.opts)
        return video, torch.tensor(self.labels[index]), video.size()[0], len(self.labels[index]), self.dlib[index]

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
