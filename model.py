from typing import Tuple
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, opts):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, opts.lstm_layer, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.opts = opts

    def forward(self, x):
        h0 = torch.randn(self.opts.lstm_layer * 2, x.shape[0], self.opts.lstm_hidden).to(self.opts.device)
        c0 = torch.randn(self.opts.lstm_layer * 2, x.shape[0], self.opts.lstm_hidden).to(self.opts.device)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.fc(x)
        return x

class CNN2LSTMCTC(nn.Module):

    def __init__(self, hidden_size, output_size, opts) -> None:
        super(CNN2LSTMCTC,self).__init__()
        self.img_con1 = self.conv3d_max(1, 16, pooling=(1, 4, 4))
        self.img_con2 = self.conv3d_max(16, 64, pooling=(1, 4, 4))
        self.img_con3 = self.conv3d_avg(64, 256)
        self.lstm = LSTM(10 + 256, hidden_size, output_size, opts)
        self.liner = nn.Linear(136, 10)
        # self.img_con4 = self.conv3d_max(128, 256, pooling=(1, 2, 2))

    def conv3d_max(self, in_channels: int, out_channels: int, pooling: Tuple[int, int, int]) -> nn.Sequential:
        # yapf: disable
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(pooling),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        # yapf: enable
        return conv_layer

    def conv3d_avg(self, in_channels: int, out_channels: int) -> nn.Sequential:
        # yapf: disable
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        # yapf: enable
        return conv_layer

    def forward(self, img, dlib):
        img = torch.permute(img, (0, 2, 1, 3, 4))
        img = self.img_con1(img)
        img = self.img_con2(img)
        img = self.img_con3(img)
        img = torch.permute(img, (0, 2, 1, 3, 4))
        img = torch.flatten(img, start_dim=2)
        dlib = self.liner(dlib)
        x = torch.cat([img, dlib], axis=2)
        x = self.lstm(x)
        return x

class CNN3LSTMCTC(nn.Module):

    def __init__(self, hidden_size, output_size, opts):
        super(CNN3LSTMCTC, self).__init__()
        self.conv3d_block1 = self.conv3d_max(1, 32, (2, 4, 4))
        self.conv3d_block2 = self.conv3d_max(32, 64, (1, 2, 2))
        self.conv3d_block3 = self.conv3d_avg(64, 128)
        self.conv2d_block1 = self.conv2d_max(1, 4, (2, 2))
        self.conv2d_block2 = self.conv2d_max(4, 8, (1, 2))
        self.conv2d_last = nn.Conv2d(8, 16, kernel_size=(3, 3))
        self.conv2d_block3 = self.conv2d_avg(8, 16)
        self.lstm = LSTM(128 + 16, hidden_size, output_size, opts)

    def conv2d_max(self, in_channels: int, out_channels: int, pooling: Tuple[int, int]) -> nn.Sequential:
        # yapf: disable
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)),
            nn.MaxPool2d(pooling),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        # yapf: enable
        return conv_layer

    def conv2d_avg(self, in_channels: int, out_channels: int) -> nn.Sequential:
        # yapf: disable
        conv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm3d(16),
            nn.ReLU()
            )
        # yapf: enable
        return conv_layer

    def conv3d_max(self, in_channels: int, out_channels: int, pooling: Tuple[int, int, int]) -> nn.Sequential:
        # yapf: disable
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3)),
            nn.MaxPool3d(pooling),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        # yapf: enable
        return conv_layer

    def conv3d_avg(self, in_channels: int, out_channels: int) -> nn.Sequential:
        # yapf: disable
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3)),
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        # yapf: enable
        return conv_layer

    def forward(self, img, dlib):
        img = torch.permute(img, (0, 2, 1, 3, 4))
        img = self.conv3d_block1(img)
        img = self.conv3d_block2(img)
        img = self.conv3d_block3(img)
        dlib = dlib.unsqueeze(1)
        dlib = self.conv2d_block1(dlib)
        dlib = self.conv2d_block2(dlib)
        dlib = self.conv2d_last(dlib)
        dlib = dlib.unsqueeze(3)
        dlib = self.conv2d_block3(dlib)
        img = torch.permute(img, (0, 2, 1, 3, 4))
        dlib = torch.permute(dlib, (0, 2, 1, 3, 4))
        img = torch.flatten(img, start_dim=2)
        dlib = torch.flatten(dlib, start_dim=2)
        x = torch.cat([img, dlib], axis=2)
        x = self.lstm(x)
        return x
