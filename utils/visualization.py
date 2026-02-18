"""
Module for visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np


# Define visualization functions
def plot_training_curves(train_losses, val_accuracies):
    """
    2 subplot: Loss curve and Val accuracy curve
    """

    # Create figure and axes for subplots also take epochs
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    epochs_loss = np.arange(len(train_losses))
    epochs_val = np.arange(len(val_accuracies))

    # Plot loss curve
    axes[0].plot(epochs_loss, train_losses)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # Plot validation accuracy
    axes[1].plot(epochs_val, val_accuracies)
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)

    # Fix layout
    plt.tight_layout()

    # Save figures
    plt.savefig("/home/cagan/cnn-from-scratch/data/plots/training_curves.png")
    plt.close()

def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix as a heatmap.
    """

    # Create figure and axes for subplots
    fig, ax = plt.subplots(figsize=(6,6))

    # Display the matrix
    im = ax.imshow(cm, cmap='Blues')

    # Add colorbar to plot
    fig.colorbar(im)

    # Add class labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Print values inside of the cells change the color based on threshold
    threshold = cm.max() / 2.

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color=color)
    # Axes labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Save figures
    plt.savefig("/home/cagan/cnn-from-scratch/data/plots/confusion_matrix.png")
    plt.close()

def show_predictions(images, labels, predictions, num_samples=10):
    """
    Take random samples and show predictions on them.
    """

    # Choose random indexes to create indices
    indices = np.random.choice(len(images), num_samples, replace=False)

    # Create a grid
    rows = int(np.ceil(num_samples / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(12, rows*3))
    axes = axes.flatten()

    # Iterate trough indicies
    for i, idx in enumerate(indices):
        # Draw image
        img = images[idx]

        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze()
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        axes[i].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[i].axis("off")

        # Add title
        axes[i].set_title(f"T:{labels[idx]} P:{predictions[idx]}")

        # Border color based on predictions
        for spine in axes[i].spines.values():
            if labels[idx] == predictions[idx]:
                spine.set_edgecolor("green")
            else:
                spine.set_edgecolor("red")
            spine.set_linewidth(3)

    # Save figures
    plt.savefig("/home/cagan/cnn-from-scratch/data/plots/predictions.png")
    plt.close()
