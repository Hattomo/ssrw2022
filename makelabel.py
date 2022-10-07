# -*- coding: utf-8 -*-

import json

import torch
import torch.nn.functional as F
import pandas as pd

import glob
from tqdm import tqdm

def get_phones_csv(label_path: str, start:int, end:int) -> list:
    label_path = sorted(glob.glob(label_path))[start:end]
    phones = []
    for path in tqdm(label_path):
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if not line.rstrip("\n").split(" ")[-1] == "":
                phones.append(line.rstrip("\n").split(" ")[-1])
    phones = list(set(phones))
    phones.append("_")
    phones.append("mask")
    phones = sorted(phones)
    return phones

def load_phones_csv() -> list:
    phones = {
    "I": 0,
    "N": 1,
    "U": 2,
    "_": 3,
    "a": 4,
    "b": 5,
    "by": 6,
    "ch": 7,
    "cl": 8,
    "d": 9,
    "dy": 10,
    "e": 11,
    "f": 12,
    "fy": 13,
    "g": 14,
    "gw": 15,
    "gy": 16,
    "h": 17,
    "hy": 18,
    "i": 19,
    "j": 20,
    "k": 21,
    "kw": 22,
    "ky": 23,
    "m": 24,
    "mask": 25,
    "my": 26,
    "n": 27,
    "ny": 28,
    "o": 29,
    "p": 30,
    "pau": 31,
    "py": 32,
    "r": 33,
    "ry": 34,
    "s": 35,
    "sh": 36,
    "sil": 37,
    "t": 38,
    "ts": 39,
    "ty": 40,
    "u": 41,
    "v": 42,
    "w": 43,
    "y": 44,
    "z": 45
}
    phones = list(phones)
    return phones

def get_label_csv(label_path: str, phones: list) -> list:
    label_path = sorted(glob.glob(label_path))
    labels = []
    for path in tqdm(label_path):
        line_list = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if len(line.rstrip("\n").split(" ")) == 3:
                one_phone = line.rstrip("\n").split(" ")[-1]
                if one_phone == "S":
                    pass
                elif one_phone == "_o":
                    one_phone = "o"
                elif one_phone == "_e":
                    one_phone = "e"
                elif one_phone == "_u":
                    one_phone = "u"
                elif one_phone == "__N":
                    one_phone = "N"
                elif one_phone == "ye":
                    pass
                elif one_phone == "_k":
                    one_phone = "k"
                elif one_phone == "_h":
                    one_phone = "h"
                else:
                    line_list.append(phones.index(one_phone))
        labels.append(line_list)
    return labels
