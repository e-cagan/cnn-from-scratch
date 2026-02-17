"""
Module for softmax layer. (unified with cross entropy loss function.)
"""

import numpy as np
from .base_layer import BaseLayer


# Define a class for softmax layer
class Softmax(BaseLayer):
    """
    A class for sofmax layer.
    """

    def __init__(self):
        super().__init__()

    # Forward and backward propagation functions for softmax layer
    def forward(self, x, labels):
        """
        Forward propagation function for softmax layer.
        """

        # Take the shapes
        batch_size, num_classes = x.shape

        # Apply softmax formula to calculate probabilities
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted_x)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        
        # Clip the probabilities to avoid division by zero error
        probs = np.clip(probs, 1e-10, 1.0)

        # Turn labels to one hot encoded matrix to calculate cross entropy loss
        y = np.eye(num_classes)[labels]                # One-hot encoded matrix

        # Calculate the softmax + cross entropy loss
        loss = np.mean(-np.log(probs[np.arange(batch_size), labels]))

        # Cache batch size, probabilities and one-hot encoded matrix to use on backward propagation
        self.cache["batch_size"] = batch_size
        self.cache["probs"] = probs
        self.cache["y"] = y

        return loss
    
    def backward(self):
        """
        Backward propagation function for softmax layer.
        """

        return (self.cache["probs"] - self.cache["y"]) / self.cache["batch_size"]
