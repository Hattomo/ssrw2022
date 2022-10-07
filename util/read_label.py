# -*- coding: utf-8 -*-

import glob
from tqdm import tqdm

path_list = glob.glob('../data/ROHAN4600/train/ROHAN4600_zundamon_voice_label/*')

phones = []

for path in tqdm(path_list):
    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # print(line.rstrip("\n").split(" ")[-1])
        phones.append(line.rstrip("\n").split(" ")[-1])

phones = list(set(phones))
print(len(phones))
