#! /bin/bash

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
cd /mnt/gold4TB/Research/21h1/av-asr-tutorial
conda activate avasrtutorial

python train.py --device cuda:1
python train.py --device cuda:1
python train.py --device cuda:1
python train.py --train-size 4096 --device cuda:1
python train.py --train-size 4096 --device cuda:1
python train.py --train-size 4096 --device cuda:1
python train.py --train-size 2048 --device cuda:1
python train.py --train-size 2048 --device cuda:1
python train.py --train-size 2048 --device cuda:1
python train.py --train-size 1024 --device cuda:1
python train.py --train-size 1024 --device cuda:1
python train.py --train-size 1024 --device cuda:1
python train.py --train-size 512 --device cuda:1
python train.py --train-size 512 --device cuda:1
python train.py --train-size 512 --device cuda:1
