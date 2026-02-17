"""
Module for adam optimizer.
"""

import numpy as np
from .base_optimizer import BaseOptim


# Define a class for adam optimizer
class Adam(BaseOptim):
    """
    Class for adam optimizer.
    """

    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate)
        self.beta1 = 0.9         # Momentum coefficient
        self.beta2 = 0.999       # Adaptive learning rate coefficient
        self.epsilon = 1e-8      # A coefficient to prevent division by 0 error
        self.m = {}              # First moment, for every parameter (momentum coefficient)
        self.v = {}              # Second moment, for every parameter (adaptive learning rate coefficient)
        self.t = 0               # Timestep, increased by one per update

    def update(self, layer):
        """
        Update function for adam optimizer.
        """

        # Increment t
        self.t += 1
        
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

            # Assign every layer a unique id
            layer_id = id(layer)

            # Check if the key exists within velocities dict or not
            if (layer_id, key) not in self.m:
                self.m[(layer_id, key)] = np.zeros_like(value)
                self.v[(layer_id, key)] = np.zeros_like(value)

            # Apply adam formula
            # First moment update
            self.m[(layer_id, key)] = self.beta1 * self.m[(layer_id, key)] + (1 - self.beta1) * grad

            # Second moment update
            self.v[(layer_id, key)] = self.beta2 * self.v[(layer_id, key)] + (1 - self.beta2) * grad ** 2

            # Bias correction
            m_hat = self.m[(layer_id, key)] / (1 - self.beta1 ** self.t)
            v_hat = self.v[(layer_id, key)] / (1 - self.beta2 ** self.t)

            # Update weights
            new_value = value - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Write it back on params
            layer.set_params({key: new_value})
        
        return
