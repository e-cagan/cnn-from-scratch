"""
Module for base abstract class and methods for models.
"""

import numpy as np
from abc import ABC, abstractmethod


# Define the base optimizer class
class BaseModel(ABC):
    """
    Base optimizer class interface for models.
    """

    def __init__(self):
        super().__init__()
        self.layers = []

    @abstractmethod
    def forward(self, x):
        """
        Forward propagation method for models.
        """

        pass
    
    @abstractmethod
    def backward(self, dout):
        """
        Backward propagation method for models.
        """

        pass
    
    def add_layer(self, layer):
        """
        Function for adding layers to models.
        """

        self.layers.append(layer)
    
    def update(self, optimizer):
        """
        Function for updating weights.
        """

        for layer in self.layers:
            optimizer.update(layer)

    def train(self):
        """
        Function for model training.
        """

        for layer in self.layers:
            layer.training = True

        return
    
    def eval(self):
        """
        Function for model evaluation.
        """

        for layer in self.layers:
            layer.training = False

        return
    
    def save(self, filepath):
        """
        Function for saving models to a file.
        """

        # Initialize empty weights dictionary
        weights_dict = {}

        # Iterate trough layers
        for i, layer in enumerate(self.layers):
            params = layer.get_params()
            if params:                          # Layers that has parameters
                weights_dict[f"layer_{i}"] = {
                    key: params[key]["value"]   # Only value, not grad
                    for key in params
                }

        np.save(filepath, weights_dict)

    def load(self, filepath):
        """
        Function for loading models from a file.
        """

        # Load the dictionary
        weights_dict = np.load(filepath, allow_pickle=True).item()

        # Iterate trough layers and set their weights
        for i, layer in enumerate(self.layers):
            key = f"layer_{i}"
            if key in weights_dict:
                layer.set_params(weights_dict[key])
