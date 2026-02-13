"""
Module for flatten layer.
"""

import numpy as np
from base_layer import BaseLayer


# Define a class for flatten layer
class Flatten(BaseLayer):
    """
    Class for flatten layer.
    """

    def __init__(self):
        super().__init__()
    
    # Forward and backward propagation functions for ReLU
    def forward(self, x):
        """
        Flatten layer forward propagation function.
        """

        # Take the shape of normal input
        batch_size, channels, height, width = x.shape

        # Store input in the cache to allow backward use it
        self.cache["x"] = x.shape

        # Reshape it
        x = x.reshape(batch_size, channels * height * width)

        return x

    def backward(self, dout):
        """
        Flatten layer backward propagation function.
        """

        # Take the shape of reshaped input (dout) and normal input
        batch_size, features = dout.shape
        _, channels, height, width = self.cache["x"]

        # Reshape it back
        dout = dout.reshape(batch_size, channels, height, width)

        return dout
