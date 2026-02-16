"""
Module for sgd + momentum optimizer.
"""

import numpy as np
from .base_optimizer import BaseOptim


# Define a class for SGD + momentum optimizer
class SGDMomentum(BaseOptim):
    """
    Class for SGD + momentum optimizer.
    """

    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate)
        self.velocities = dict()
        self.momentum = 0.9

    def update(self, layer):
        """
        Update function for SGD + momentum optimizer.
        """

        # Take params
        params = layer.get_params()
        
        # Check layer has params to check
        if not params:
            return
        
        # Iterate trough params
        for key in params.keys():
            # Take the value and gradient for key
            value = params[key]["value"]
            grad = params[key]["grad"]

            # Check if the key exists within velocities dict or not
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(value)

            # Apply SGD + momentum formula
            self.velocities[key] = self.momentum * self.velocities[key] - self.learning_rate * grad
            new_value = value + self.velocities[key]

            # Write it back on params
            layer.set_params({key: new_value})
        
        return
