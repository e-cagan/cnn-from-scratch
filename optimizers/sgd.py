"""
Module for vanilla sgd optimizer.
"""

import numpy as np
from .base_optimizer import BaseOptim


# Define a class for vanilla SGD
class SGD(BaseOptim):
    """
    Class for vanilla SGD.
    """

    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate)

    def update(self, layer):
        """
        Update function for vanilla SGD.
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

            # Apply vanilla SGD formula
            new_value = value - self.learning_rate * grad

            # Write it back on params
            layer.set_params({key: new_value})
        
        return
