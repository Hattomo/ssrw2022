# -*- coding: utf-8 -*-

import glob
import re
import os
import time
import random

import pandas as pd
from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn

class KDataset(data.Dataset):

    def __init__(self, labels, opts, phones, transform=None) -> None:
        super().__init__()
        self.phones = phones
        self.labels = labels
        self.opts = opts
        self.inputs_length = []
        self.mask_ratio = 0.20

        for sentence in self.labels:
            self.inputs_length.append(len(sentence))

    def random_replace(self, tgt: list) -> list:
        mask_num = int(len(tgt) * self.mask_ratio)
        mask_indexs = torch.randint(0, len(tgt), (mask_num,))
        for i in mask_indexs:
            mask_phone = random.randint(0, len(self.phones)-1)
            if not (mask_phone == self.phones.index("_") or mask_phone == self.phones.index("mask")):
                tgt[i] = mask_phone
        return tgt

    def __getitem__(self, index: int):
        label = self.labels[index]
        tgt = self.random_replace(label)
        return torch.tensor(label), torch.tensor(tgt)

    def __len__(self) -> int:
        return len(self.labels)

class MyCollator(object):

    def __init__(self, phones: list):
        self.phones = phones

    def __call__(self, batch):
        inputs, targets = list(zip(*batch))
        inputs = rnn.pad_sequence(inputs, batch_first=True, padding_value=self.phones.index("_"))
        targets = rnn.pad_sequence(targets, batch_first=True, padding_value=self.phones.index("_"))
        return inputs, targets
