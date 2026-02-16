"""
Module for testing the fully connected layer gradient tests and implementation.
"""

# Handle file management
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from layers.fc import FCLayer
from utils.gradient_check import gradient_check


# Test out the implementation
if __name__ == '__main__':
    # Initializing layer and input sample
    layer = FCLayer(in_features=8, out_features=4)
    random_sample = np.random.sample(size=(3, 8))

    # Gradient check
    max_err = gradient_check(layer=layer, x=random_sample)
