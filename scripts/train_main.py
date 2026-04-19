#!/usr/bin/env python3


from utils.ALVINN import ALVINN
from utils.MELVINN import MELVINN
from utils.CALVINN import CALVINN
from utils.model_utils import load_model, save_checkpoint, print_model_params

from utils.training_utils import training_pipelpine

 
def train_ALVINN():
    image_size = (32,32)
    alvinn = ALVINN(image_size)
    training_pipelpine(
        alvinn,
        image_size,
        n_minibatch_steps=10000,
        use_blue = True
    )
        
def train_MELVINN():
    image_size = (64,64)
    channels = 3
    use_blue = True
    if use_blue:
        channels = 1
    melvinn = MELVINN(image_size, color_channels=channels)
    training_pipelpine(
        melvinn,
        image_size,
        n_minibatch_steps=10000,
        use_blue = use_blue
    )

def train_CALVINN():
    image_size = (32,32)
    channels = 3
    use_blue = True
    if use_blue:
        channels = 1
    melvinn = CALVINN(image_size, color_channels=channels)
    training_pipelpine(
        melvinn,
        image_size,
        n_minibatch_steps=10000,
        use_blue = use_blue
    )


    

if __name__ == "__main__":
    train_CALVINN()