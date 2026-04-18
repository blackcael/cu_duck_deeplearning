#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.nn.parameter import Parameter
from PIL import Image


INPUT_SIZE_H = 32
INPUT_SIZE_W = 32

HIDDEN_UNITS = 29

OUTPUT_LAYER_SIZE = 45



class SIMONN_LAYER(nn.Module):
    def __init__(self, input_channels, output_channels, p_dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        return self.net(x)



class SIMONN(nn.Module):
    def __init__(self, imagesize_hw = (INPUT_SIZE_H, INPUT_SIZE_W), color_channels = 1, p_dropout = 0.1):
        super().__init__()
        BASE_CHANNELS = color_channels
        HIDDEN_CHANNELS_L1 = BASE_CHANNELS * 2
        HIDDEN_CHANNELS_L2 = BASE_CHANNELS * 4
        HIDDEN_CHANNELS_L3 = BASE_CHANNELS * 8
        HIDDEN_CHANNELS_L4 = BASE_CHANNELS * 16

        h,w =  imagesize_hw
        self.features = nn.Sequential(
            SIMONN_LAYER(BASE_CHANNELS,      HIDDEN_CHANNELS_L1, p_dropout = p_dropout),
            SIMONN_LAYER(HIDDEN_CHANNELS_L1, HIDDEN_CHANNELS_L2, p_dropout = p_dropout),
            SIMONN_LAYER(HIDDEN_CHANNELS_L2, HIDDEN_CHANNELS_L3, p_dropout = p_dropout),
            SIMONN_LAYER(HIDDEN_CHANNELS_L3, HIDDEN_CHANNELS_L4, p_dropout = p_dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(HIDDEN_CHANNELS_L4 * h * w, OUTPUT_LAYER_SIZE)
        )

        self.net = nn.Sequential(
            self.features,
            self.classifier,
        )

    def forward(self, x):
        return self.net(x)
