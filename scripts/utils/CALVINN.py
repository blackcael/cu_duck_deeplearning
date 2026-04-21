#!/usr/bin/env python3
import torch
import torch.nn as nn


INPUT_SIZE_H = 32
INPUT_SIZE_W = 32

OUTPUT_LAYER_SIZE = 45


class CALVINNStem(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CALVINNResidualBlock(nn.Module):
    def __init__(self, channels, p_dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(p_dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


class CALVINNDownsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CALVINN(nn.Module):
    def __init__(self, imagesize_hw = (INPUT_SIZE_H, INPUT_SIZE_W), color_channels = 1, p_dropout = 0.1):
        super().__init__()
        self.name = "calvinn"

        # A lightweight residual backbone with progressive downsampling.
        c1, c2, c3 = 16, 32, 64
        lin_hidden_layer_units = 64
        self.features = nn.Sequential(
            CALVINNStem(color_channels, c1),
            CALVINNResidualBlock(c1, p_dropout=p_dropout),
            CALVINNDownsample(c1, c2),
            CALVINNResidualBlock(c2, p_dropout=p_dropout),
            CALVINNDownsample(c2, c3),
            CALVINNResidualBlock(c3, p_dropout=p_dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(c3, lin_hidden_layer_units),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(lin_hidden_layer_units, OUTPUT_LAYER_SIZE),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
    
    def get_name(self):
        return self.name
    
    def set_name(self, new_name):
        self.name = new_name
