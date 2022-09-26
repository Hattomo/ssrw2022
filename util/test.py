# -*- coding: utf-8 -*-

import torchvision
import torch

video_path = "/mnt/gold4TB/dataset/ROHAN4600/train/video/0001.mp4"
reader = torchvision.io.VideoReader(video_path, "video")
# video = torchvision.io.read_video(video_path, pts_unit='sec', output_format="TCHW")[0]
video = video.to('cuda:0')
print(video.device)
torch.save(video, "video.pt")
video2 = torch.load("video.pt")
print(video2.device)
