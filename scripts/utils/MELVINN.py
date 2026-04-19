#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.nn.parameter import Parameter
from PIL import Image

""" 
Based on Dean's paper ALVINN: Autonomous Land Vehicle In a Nueral Network with alterations to accomadate modern advances in compute capacity

Input 30 x 32 Video Camera -> We will use a 32 by 32 image (for simplicity)
45 Output Directions (output Layer) -> (this will require a litle kinematics math to map into motor speeds (our dataset record))
"""
INPUT_SIZE_H = 32
INPUT_SIZE_W = 32
HIDDEN_UNITS = 128
OUTPUT_LAYER_SIZE = 45

class MELVINN_Layer(nn.Module):
    def __init__(self, input_units, output_units, p_dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_units, output_units),
            nn.BatchNorm1d(output_units),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        return self.net(x)

class MELVINN(nn.Module):
    def __init__(self, imagesize_hw = (INPUT_SIZE_H, INPUT_SIZE_W), color_channels = 1):
        super().__init__()
        self.name = "melvinn"
        h, w = imagesize_hw
        input_units = color_channels * h * w
        self.net = nn.Sequential(
            MELVINN_Layer(input_units, HIDDEN_UNITS),
            MELVINN_Layer(HIDDEN_UNITS, HIDDEN_UNITS),
            MELVINN_Layer(HIDDEN_UNITS, HIDDEN_UNITS),
            MELVINN_Layer(HIDDEN_UNITS, HIDDEN_UNITS),
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
