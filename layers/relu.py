"""
Module for Rectified Linear Unit (ReLU) activation function.
"""

import numpy as np
from .base_layer import BaseLayer


# Define a class for ReLU activation
class ReLU(BaseLayer):
    """
    Class for ReLU activation function.
    """

    def __init__(self):
        super().__init__()

    # Forward and backward propagation functions for ReLU
    def forward(self, x):
        """
        ReLU forward propagation function.
        """

        # Store the input on the cache to use in backward propagation
        self.cache["x"] = x

        return np.maximum(0, x)

    def backward(self, dout):
        """
        ReLU backward propagation function.
        """
        
        return dout * (self.cache["x"] > 0)
