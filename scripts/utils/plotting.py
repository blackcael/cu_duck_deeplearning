import math
from matplotlib import pyplot as plt
import numpy as np

# Generate Figures
def plot_training(training_losses, training_accuracies, val_losses, val_accuracies, training_bin_maes, val_bin_maes, val_ts):
    '''
    Plots training and validation losses and accuracies over epochs
    '''

    epochs = range(0, len(training_losses))

    plt.figure(figsize=(18, 5))
    
    # Plot Losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(val_ts, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot Classification Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, training_accuracies, label='Training Accuracy')
    plt.plot(val_ts, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Class Accuracy')
    plt.legend()

    # Plot Steering Bin MAE (lower is better)
    plt.subplot(1, 3, 3)
    plt.plot(epochs, training_bin_maes, label='Training Bin MAE')
    plt.plot(val_ts, val_bin_maes, label='Validation Bin MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE (angle bins)')
    plt.title('Training and Validation Bin MAE')
    plt.legend()
    
    
    plt.show()
