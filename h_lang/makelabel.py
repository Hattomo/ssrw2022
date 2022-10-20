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
    labels = []
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line_list = []
        sentence = line.rstrip("\n").split(" ")
        if len(sentence) < 250:
            for phone in sentence:
                if phone == "S":
                    pass
                elif phone == "_o":
                    phone = "o"
                elif phone == "_e":
                    phone = "e"
                elif phone == "_u":
                    phone = "u"
                elif phone == "__N":
                    phone = "N"
                elif phone == "ye":
                    pass
                elif phone == "_k":
                    phone = "k"
                elif phone == "_h":
                    phone = "h"
                elif phone == "":
                    pass
                else:
                    line_list.append(phones.index(phone))
            labels.append(line_list)
    print(len(labels))
    return labels
