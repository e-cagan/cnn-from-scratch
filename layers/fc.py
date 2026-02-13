"""
Module for fully connected layer.
"""

import numpy as np
from base_layer import BaseLayer


# Define the fully connected layer
class FCLayer(BaseLayer):
    """
    Fully connected layer class.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)    # Using He init
        self.biases = np.zeros(shape=(out_features,))
        self.dW = None
        self.db = None
    
    # Forward and backward propagations
    def forward(self, x):
        """
        Applying the formula ---> inputs @ weights (transpose matrix) + biases
        """

        # Store the input on the cache to use in backward propagation
        self.cache["x"] = x

        return np.dot(x, self.weights) + self.biases
    
    
    def backward(self, dout):
        """
        Finding derivatives with chain rule.
        """

        self.dW = np.dot(self.cache["x"].T, dout)
        self.db = np.sum(dout, axis=0)
        dX = np.dot(dout, self.weights.T)

        return dX
    
    # Getters and setters for params
    def get_params(self):
        return {
            "W": {"value": self.weights, "grad": self.dW},
            "b": {"value": self.biases, "grad": self.db}
        }

    def set_params(self, params):
        self.weights = params["W"]
        self.biases = params["b"]
    