"""
Module for loading MNIST dataset using pytorch and preprocessing the data
"""

from torchvision.datasets import MNIST
import numpy as np


# ----------------------------------------------------------------------------------
# -------------------------- DATA PREPROCESSING FUNCTIONS --------------------------
def to_numpy(tensor):
    """
    A function that converts tensor to numpy array.
    """

    # Convert tensor to numpy array
    arr = tensor.numpy()

    # Cast the datatype to float32
    arr = arr.astype(np.float32)

    return arr

def reshape(images):
    """
    A function that reshapes the input size to (N, 1, 28, 28) from (N, 28, 28)
    """

    # Reshape
    images = images.reshape(len(images), 1, 28, 28)

    return images

def normalize(images):
    """
    A function that normalizes pixel values for better training.
    """

    # Normalize via mean and std values
    images = (images - 0.1307) / 0.3081

    return images

def split_validation(images, labels, val_ratio=0.1):
    """
    A function that splits training set to training and validation set based on validation ratio.
    """

    # Shuffle trough images and labels using random indicies
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    # Calculate train and validation size
    val_size = int((len(images) * val_ratio))
    train_size = int(len(images) - val_size)

    # Split train and val sets
    train_images, train_labels = images[:train_size], labels[:train_size]
    val_images, val_labels = images[train_size:], labels[train_size:]

    return train_images, train_labels, val_images, val_labels

def preprocess(dataset):
    """
    A function that preprocesses the dataset.
    """

    # Convert images to numpy array
    images = to_numpy(dataset.data)

    # Convert labels to numpy array also convert labels to integers
    labels = to_numpy(dataset.targets)
    labels = labels.astype(np.int64)

    # Reshape
    images = reshape(images)

    # Normalize
    images = normalize(images)

    return images, labels
# ----------------------------------------------------------------------------------

def get_batches(images, labels, batch_size, shuffle=True):
    """
    A function for generating random batches
    """

    # Check the shuffle value for randomness
    if shuffle:
        indices = np.random.permutation(len(images))
        images, labels = images[indices], labels[indices]

    # Iterate trough batches
    for i in range(0, len(images), batch_size):
        # Return images and labels and continue returning until the end of the loop
        yield images[i:i+batch_size], labels[i:i+batch_size]

# Define a function for loading MNIST dataset and convert tensors to numpy arrays
def load_mnist():
    """
    Module for loading MNIST dataset and converting it to numpy array.
    """
    
    # Load the train and test splits
    train_dataset = MNIST(
        root='data/raw',
        train=True,
        download=True
    )

    test_dataset = MNIST(
        root='data/raw',
        train=False,
        download=True
    )

    # Preprocess the datasets
    train_images, train_labels = preprocess(train_dataset)
    test_images, test_labels = preprocess(test_dataset)

    # Split the dataset
    train_images, train_labels, val_images, val_labels = split_validation(train_images, train_labels)

    return {
        "train": (train_images, train_labels),
        "val":   (val_images,   val_labels),
        "test":  (test_images,  test_labels)
    }

if __name__ == "__main__":
    # Test out the dataset
    data = load_mnist()
    train_images, train_labels = data["train"]
    val_images, val_labels = data["val"]
    test_images, test_labels = data["test"]
    
    print(train_images.shape)   # (54000, 1, 28, 28)
    print(train_labels.shape)   # (54000,)
    print(val_images.shape)     # (6000, 1, 28, 28)
    print(test_images.shape)    # (10000, 1, 28, 28)
    print(train_images.dtype)   # float32
    print(train_labels.dtype)   # int64
