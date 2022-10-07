# -*- coding: utf-8 -*-

import csv
import sys

import torch
import torch.nn.functional as F
import pandas as pd

import glob
from tqdm import tqdm

def get_phones_csv(label_path: str) -> list:
    label_path = glob.glob(label_path)
    phones = []
    for path in tqdm(label_path):
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            phones.append(line.rstrip("\n").split(" ")[-1])
    phones = list(set(phones))
    phones.append("_")
    return phones

def get_label_csv(label_path: str, phones: list) -> list:
    label_path = sorted(glob.glob(label_path))
    labels = []
    for path in tqdm(label_path):
        line_list = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line_list.append(phones.index(line.rstrip("\n").split(" ")[-1]))
        labels.append(line_list)
    return labels
