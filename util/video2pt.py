# -*- coding: utf-8 -*-

import glob
import os
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

image_path = "../data/ROHAN4600/train/video"
video_path = "../data/ROHAN4600/train/tensor/"
# path_data = glob.glob('*.png')
path_data = sorted(glob.glob(f'{image_path}/*'))

trans = torch.nn.Sequential(
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
    # transforms.ConvertImageDtype(torch.float),
    # transforms.Normalize(mean=[0.5], std=[0.5]),
).to(torch.device('cuda:1'))

def video2pt(paths: str) -> None:
    print(paths)
    print(paths[-4:])
    # res = os.system(f"ffmpeg -y -i {paths}/%05d.png -vcodec libx264 -pix_fmt yuv420p {video_path}/{paths[-4:]}.mp4")

for file in tqdm(path_data):
    video = torchvision.io.read_video(file, pts_unit='sec', output_format="TCHW")[0]
    video = video.to("cuda:1")
    video = trans(video)
    video = video.to("cpu")
    torch.save(video, os.path.join(video_path, file[-8:-4] + ".pt"))
