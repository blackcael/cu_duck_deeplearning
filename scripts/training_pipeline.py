#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils, datasets

from utils.DuckieDriveDataset import DuckieDriveDataset
from utils.ALVINN import ALVINN
from utils.SIMONN import SIMONN
from utils.plotting import plot_training
from utils.model_utils import load_model, save_checkpoint
import utils.fn_utils


from utils.data_player import play_dataset_with_arrow
import matplotlib.pyplot as plt

# Global Constants
DEF_DATA_DIR = "data/"


def loss_fn(logits, truth_angle):
    # Dataset angles are in [-22, 22]; map to ALVINN class bins [0, 44].
    truth_class = torch.clamp(torch.round(truth_angle + 22), 0, 44).long().view(-1)
    return F.cross_entropy(logits, truth_class)

def angle_to_class(truth_angle):
    return torch.clamp(torch.round(truth_angle + 22), 0, 44).long().view(-1)

def class_accuracy_fn(logits, truth_angle):
    pred_class = torch.argmax(logits, dim=1)
    truth_class = angle_to_class(truth_angle)
    return (pred_class == truth_class).float().mean()

def bin_mae_fn(logits, truth_angle):
    pred_angle = logits_to_angle(logits).float()
    return torch.mean(torch.abs(pred_angle - truth_angle.float()))

def logits_to_angle(logits):
    return torch.argmax(logits, dim=1) - 22

def validate(model, validation_dataloader, device):
    losses = []
    class_accuracies = []
    bin_maes = []
    for img, truth_angle in validation_dataloader:
        img = img.to(device)
        truth_angle = truth_angle.to(device)
        with torch.no_grad():
            logits = model(img)
            loss = loss_fn(logits, truth_angle)
            losses.append(loss.item())
            class_accuracies.append(class_accuracy_fn(logits, truth_angle).item())
            bin_maes.append(bin_mae_fn(logits, truth_angle).item())

    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    avg_class_acc = sum(class_accuracies) / len(class_accuracies) if class_accuracies else float("nan")
    avg_bin_mae = sum(bin_maes) / len(bin_maes) if bin_maes else float("nan")
    return avg_loss, avg_class_acc, avg_bin_mae

def print_progress(i_step, total_steps, training_loss = None, validation_loss = None):
    progress_string = f"Progress: {i_step / total_steps:.2f}; Step: {i_step} / {total_steps}"

    if training_loss is not None:
        progress_string += f"; training loss = {training_loss:.2f}"
    if  validation_loss is not None:
        progress_string += f"; validation loss = {validation_loss:.2f}"

    print(progress_string, end = "\r")

def print_memory_summary(device, verbose = False):
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()  # (free_bytes, total_bytes)
        print(f"Memory Info: Free={free / (1024**3):.2f}GB, Total={total / (1024**3):.2f}GB")
        if verbose:
            print(torch.cuda.memory_summary())




def train(model, optimizer, training_dataloader, validation_dataloader, n_optimization_steps, log_interval, val_interval, device):

    # Init Variables
    training_loss = []
    val_loss = []
    training_class_accuracies = []
    validation_class_accuracies = []
    training_bin_maes = []
    validation_bin_maes = []
    val_ts = []
    i = 0

    # Send model to GPU
    model = model.to(device)

    # Training Loop
    while i < n_optimization_steps:
        for img, truth_angle in training_dataloader:

            # Move Data onto the GPU
            img         = img.to(device)
            truth_angle = truth_angle.to(device)

            # Run data through the model and calculate the loss
            logits = model(img)
            loss = loss_fn(logits, truth_angle)
            training_class_accuracies.append(class_accuracy_fn(logits, truth_angle).item())
            training_bin_maes.append(bin_mae_fn(logits, truth_angle).item())

            # Do the Learning Dance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record Progress
            training_loss.append(loss.item())
            if (i % val_interval == 0):
                validation_loss, val_class_accuracy, val_bin_mae = validate(model, validation_dataloader, device)
                val_loss.append(validation_loss)
                validation_class_accuracies.append(val_class_accuracy)
                validation_bin_maes.append(val_bin_mae)
                val_ts.append(i)

            if (i % log_interval == 0):
                print_progress(i, n_optimization_steps, training_loss[-1], val_loss[-1])

            i += 1
            if i >= n_optimization_steps:
                break
    
            
    # Return Data
    return training_loss, training_class_accuracies, val_loss, validation_class_accuracies, training_bin_maes, validation_bin_maes, val_ts

 
def train_ALVINN(
        image_size = (32,32),
        n_minibatch_steps = 10000,
        preload_num_workers = 8,
        turn_data_smoothing_window = 9
):
    # Make results reproducible
    torch.manual_seed(0)

    # Init device and print summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_memory_summary(device)

    # Bring in DataSet
    dataset = DuckieDriveDataset(
        data_dir=DEF_DATA_DIR,
        preload_images=True,
        cache_preloaded_images=True,
        preload_num_workers=preload_num_workers,
        image_size=image_size,
        use_angles=True,
        use_blue_channel=True,
        smooth_labels=True,
        smoothing_window=turn_data_smoothing_window,
    )
    print(f"dataset length: {len(dataset)}")
    print(f"Average Speed in Dataset: {dataset.get_average_speed()}")

    # max_w, min_w = dataset.get_max_min_w(AXEL_LENGTH_MM)
    # print(f" Max W = {max_w}, Min W = {min_w}")
    # mag_max_w = max_w if max_w > abs(min_w) else abs(min_w)

    training_dataset = dataset
    validation_dataset = training_dataset.slice_off_validation_dataset()
    print(f"Training dataset length: {len(training_dataset)}, Validation dataset length:  {len(validation_dataset)}")

    batch_size = 256
    num_workers = 8
    training_dataloader  = DataLoader(training_dataset,    shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Init Model and Optimizer
    alvinn = ALVINN(imagesize_hw=image_size)
    LR = 3e-4
    optimizer = torch.optim.AdamW(params = alvinn.parameters(), lr = LR, weight_decay=1e-2)


    print("Beginning Training!")
    train_results = train(
        alvinn, 
        optimizer, 
        training_dataloader, 
        validation_dataloader,
        n_minibatch_steps,
        log_interval = 10,
        val_interval = 10,
        device = device
    )
    print("\nCompleted Training!")

    checkpoint_path, latest_path = save_checkpoint(
        model=alvinn,
        optimizer=optimizer,
        image_size=image_size,
        n_optimization_steps=n_minibatch_steps,
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Updated latest checkpoint: {latest_path}")
    
    training_loss, training_class_accs, val_loss, validation_class_accs, training_bin_maes, validation_bin_maes, val_ts = train_results
    plot_training(training_loss, training_class_accs, val_loss, validation_class_accs, training_bin_maes, validation_bin_maes, val_ts)
    
    plt.show()
    fig, anim = play_dataset_with_arrow(
        validation_dataset,
        model_device=device,
        model = alvinn,
        fps=30,

    )

    plt.show()

def train_SIMONN(
        image_size = (32, 32),
        n_minibatch_steps = 200000,
        preload_num_workers = 8,
        turn_data_smoothing_window = 9,
        use_blue = False
):
    # Make results reproducible
    torch.manual_seed(0)

    # Init device and print summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_memory_summary(device)

    # Bring in DataSet
    dataset = DuckieDriveDataset(
        data_dir=DEF_DATA_DIR,
        preload_images=True,
        cache_preloaded_images=True,
        preload_num_workers=preload_num_workers,
        image_size=image_size,
        use_angles=True,
        use_blue_channel=use_blue,
        smooth_labels=True,
        smoothing_window=turn_data_smoothing_window,
    )
    print(f"dataset length: {len(dataset)}")
    print(f"Average Speed in Dataset: {dataset.get_average_speed()}")

    # max_w, min_w = dataset.get_max_min_w(AXEL_LENGTH_MM)
    # print(f" Max W = {max_w}, Min W = {min_w}")
    # mag_max_w = max_w if max_w > abs(min_w) else abs(min_w)

    training_dataset = dataset
    validation_dataset = training_dataset.slice_off_validation_dataset()
    print(f"Training dataset length: {len(training_dataset)}, Validation dataset length:  {len(validation_dataset)}")

    batch_size = 256
    num_workers = 8
    training_dataloader   = DataLoader(training_dataset,   shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Init Model and Optimizer
    if len(training_dataset) == 0:
        raise ValueError("Training dataset is empty; cannot infer input channels for SIMONN.")
    sample_img, _ = training_dataset[0]
    input_channels = sample_img.shape[0]
    simonn = SIMONN(imagesize_hw=image_size, color_channels=input_channels)
    LR = 3e-4
    optimizer = torch.optim.AdamW(params = simonn.parameters(), lr = LR, weight_decay=1e-2)


    print("Beginning Training!")
    train_results = train(
        simonn, 
        optimizer, 
        training_dataloader, 
        validation_dataloader,
        n_minibatch_steps,
        log_interval = 10,
        val_interval = 10,
        device = device
    )
    print("\nCompleted Training!")

    checkpoint_path, latest_path = save_checkpoint(
        model=simonn,
        optimizer=optimizer,
        image_size=image_size,
        n_optimization_steps=n_minibatch_steps,
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Updated latest checkpoint: {latest_path}")
    
    training_loss, training_class_accs, val_loss, validation_class_accs, training_bin_maes, validation_bin_maes, val_ts = train_results
    plot_training(training_loss, training_class_accs, val_loss, validation_class_accs, training_bin_maes, validation_bin_maes, val_ts)
    
    plt.show()
    fig, anim = play_dataset_with_arrow(
        validation_dataset,
        model_device=device,
        model = simonn,
        fps=30,
    )

    plt.show()


    

if __name__ == "__main__":
    train_SIMONN(
        n_minibatch_steps=200000,
        turn_data_smoothing_window = 13,
        use_blue = True
    )
