# -*- coding: utf-8 -*-

import os
import glob
import shutil

data_type = "label"  # tensor, csv, label

# zundamon, itako, metan, sora
# ama, sexy, normal, recitation, tsun, emoWhis
name_emo_list = [("zundamon", "normal"), ("zundamon", "ama"), ("zundamon", "tsun"), ("zundamon", "sexy"),
                 ("zundamon", "recitation"), ("itako", "normal"), ("itako", "ama"), ("itako", "tsun"),
                 ("itako", "sexy"), ("itako", "recitation"), ("metan", "normal"), ("metan", "ama"), ("metan", "tsun"),
                 ("metan", "sexy"), ("metan", "recitation"), ("sora", "emoNormal"), ("sora", "emoAma"),
                 ("sora", "emoTsun"), ("sora", "emoSexy"), ("sora", "emoWhis"), ("sora", "recitation")]

start = 1
for name, emotion in name_emo_list:
    data_path = f"../data/multimodal-label/{name}/emotion/{emotion}"
    target_path = f"../data/train/{data_type}"

    data_list = sorted(glob.glob(f"{data_path}/*.lab"))
    for i, file in enumerate(data_list):
        # print(i + 2900, os.path.join(target_path, f"{i+2900:05d}.pt"))
        print(file, os.path.join(target_path, f"{start:05d}.lab"))
        shutil.copyfile(file, os.path.join(target_path, f"{start:05d}.lab"))
        start += 1
        print(start)


data_path = f"../data/ROHAN4600/train/ROHAN4600_zundamon_voice_label"
target_path = f"../data/train/{data_type}"
data_list = sorted(glob.glob(f"{data_path}/*.lab"))
for i, file in enumerate(data_list):
    # print(i + 2900, os.path.join(target_path, f"{i+2900:05d}.pt"))
    print(file, os.path.join(target_path, f"{start:05d}.lab"))
    shutil.copyfile(file, os.path.join(target_path, f"{start:05d}.lab"))
    start += 1
