# -*- coding: utf-8 -*-

import glob
import os
import subprocess
from tqdm import tqdm

image_path = "/mnt/gold4TB/dataset/ROHAN4600/test/image"
video_path = "/mnt/gold4TB/dataset/ROHAN4600/test/video"
# path_data = glob.glob('*.png')
path_data = sorted(glob.glob(f'{image_path}/*'))

def image2video(paths:str)->None:
    print(paths)
    print(paths[-4:])
    res = os.system(f"ffmpeg -y -i {paths}/%05d.png -vcodec libx264 -pix_fmt yuv420p {video_path}/{paths[-4:]}.mp4")

for file in tqdm(path_data):
    images_path = sorted(glob.glob(f'{file}/*.png'))
    image2video(file)
