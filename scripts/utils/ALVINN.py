#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.nn.parameter import Parameter
from PIL import Image

""" 
Based on Dean's paper ALVINN: Autonomous Land Vehicle In a Nueral Network

Input 30 x 32 Video Camera -> We will use a 32 by 32 image (for simplicity)
Middle 29 Hidden Units
45 Output Directions (output Layer) -> (this will require a litle kinematics math to map into motor speeds (our dataset record))
"""
INPUT_SIZE_H = 32
INPUT_SIZE_W = 32

HIDDEN_UNITS = 29

OUTPUT_LAYER_SIZE = 45


class ALVINN(nn.Module):
    def __init__(self, imagesize_hw = (INPUT_SIZE_H, INPUT_SIZE_W)):
        super().__init__()
        self.name = "alvinn"
        h, w = imagesize_hw
        self.net = nn.Sequential(
            nn.Linear(h * w, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, OUTPUT_LAYER_SIZE)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.net(x)
        return x

    def get_name(self):
        return self.name
    
    def set_name(self, new_name):
        self.name = new_name
