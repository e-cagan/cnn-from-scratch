"""
Module for MaxPool layer.
"""

import numpy as np
from base_layer import BaseLayer


# Define a class for maxpool layer
class MaxPool(BaseLayer):
    """
    Class for MaxPool layer.
    """

    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
    
    # Forward and backward propagation functions for MaxPool layer
    def forward(self, x):
        """
        Forward propagation function for max pooling.
        """

        # Take credentials and calculate width and height for output
        batch_size, channels, height, width = x.shape
        
        # Output width and height
        out_width = (width - self.pool_size) // self.stride + 1
        out_height = (height - self.pool_size) // self.stride + 1
        
        # Create mask and output array with respect to the shape of the input
        mask = np.zeros(shape=(batch_size, channels, height, width), dtype=bool)
        output = np.zeros(shape=(batch_size, channels, out_height, out_width))

        # Iterate trough window
        for n in range(batch_size):            # Every sample
            for c in range(channels):          # Every channel
                for i in range(out_height):    # Output's every row
                    for j in range(out_width): # Output's every column

                        # Calculate the max of the window and fill out the output and mask windows
                        window = x[n, c, i * self.stride : i * self.stride + self.pool_size, j * self.stride : j * self.stride + self.pool_size]
                        output[n, c, i, j] = np.max(window)

                        # Mask will be consist of 1's and 0's based on the location of max
                        mask[n, c, i * self.stride : i * self.stride + self.pool_size, j * self.stride : j * self.stride + self.pool_size] = (window == np.max(window))
                    
        # Store the output and mask inside of the cache to use it on backward propagation
        self.cache["x"] = x
        self.cache["mask"] = mask

        return output
    
    def backward(self, dout):
        """
        Backward propagation function for max pooling.
        """

        # Take credentials and calculate width and height for output
        batch_size, channels, out_height, out_width = dout.shape

        # Create array for gradients
        dX = np.zeros(shape=self.cache["x"].shape)  

        # Iterate trough window
        for n in range(batch_size):            # Every sample
            for c in range(channels):          # Every channel
                for i in range(out_height):    # Output's every row
                    for j in range(out_width): # Output's every column

                        # Distribute the gradients based on locations at max values
                        scalar_window = dout[n, c, i, j]
                        mask_window = self.cache["mask"][n, c, i * self.stride : i * self.stride + self.pool_size, j * self.stride : j * self.stride + self.pool_size]
                        dX[n, c, i * self.stride : i * self.stride + self.pool_size, j * self.stride : j * self.stride + self.pool_size] += scalar_window * mask_window
        
        return dX
