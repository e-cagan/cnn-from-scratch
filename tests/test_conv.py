"""
Module for testing the convolution layer gradient tests and implementation.
"""

# Handle file management
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from layers.conv import ConvLayer
from utils.gradient_check import gradient_check


# Test out the implementation
if __name__ == '__main__':
    # Initializing layer and input sample
    layer = ConvLayer(in_channels=1, out_channels=2, kernel_size=3)
    random_sample = np.random.sample(size=(2, 1, 8, 8))

    # Gradient check
    max_err = gradient_check(layer=layer, x=random_sample)
