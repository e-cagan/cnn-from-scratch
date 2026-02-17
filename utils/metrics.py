"""
Module for metrics to evaluate model.
"""

import numpy as np


# Define a function for accuracy metric
def accuracy(predictions, labels):
    """
    Function for measuring accuracy
    """

    # Calculate the accuracy
    return np.sum(predictions == labels) / len(labels)

# Define a function for confusion matrix
def confusion_matrix(predictions, labels, num_classes=10):
    """
    Function for evaluating confusion matrix.
    """

    # Create confusion matrix
    matrix = np.zeros(shape=(num_classes, num_classes))

    # Iterate trough entire evaluation matrix
    for pred, label in zip(predictions, labels):
        matrix[pred, label] += 1

    return matrix
