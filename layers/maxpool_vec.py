"""
Module for vectorized MaxPool layer.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from .base_layer import BaseLayer


# Define a class for maxpool layer
class MaxPoolVec(BaseLayer):
    """
    Class for vectorized MaxPool layer.
    """

    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
    
    # Forward and backward propagation functions for MaxPool layer
    def forward(self, x):
        """
        Forward propagation function for vectorized max pooling.
        """

        # Calculate shape and output width, height
        batch, channels, H, W = x.shape
        out_H = (H - self.pool_size) // self.stride + 1
        out_W = (W - self.pool_size) // self.stride + 1

        # Calculate strides
        batch_stride = x.strides[0]
        channel_stride = x.strides[1]
        row_stride = x.strides[2]
        col_stride = x.strides[3]

        strides = (
            batch_stride,                    # Batch dimension
            channel_stride,                  # Channel dimension
            self.stride * row_stride,        # Output row (skip rows in range of stride)
            self.stride * col_stride,        # Output col (skip columns in range of stride)
            row_stride,                      # Row inside of the pool window
            col_stride                       # Column inside of the pool window
        )

        # Create window view
        shape = (batch, channels, out_H, out_W, self.pool_size, self.pool_size)
        windows = as_strided(x, shape=shape, strides=strides)

        # Take the output
        output = windows.max(axis=(4, 5))  # Calculate max considering only last 2 dimensions

        # Flatten the windows and find max index
        windows_flat = windows.reshape(batch, channels, out_H, out_W, -1)
        argmax = windows_flat.argmax(axis=-1)  # Shape: (batch, channels, out_H, out_W)

        # Store input shape and max index on cache
        self.cache["x_shape"] = x.shape
        self.cache["argmax"] = argmax

        return output

    def backward(self, dout):
        """
        Backward propagation function for vectorized max pooling.
        """

        # Unpack the values from the cahce and create matrix for dX
        x_shape = self.cache["x_shape"]
        argmax = self.cache["argmax"]  # (batch, channels, out_H, out_W)

        batch, channels, H, W = x_shape
        _, _, out_H, out_W = dout.shape

        dX = np.zeros(x_shape)

        # Convert argmax to 2D position
        pool_row = argmax // self.pool_size  # 0 < row < pool_size-1
        pool_col = argmax % self.pool_size   # 0 < col < pool_size-1

        # Create output indexes via meshgrid
        out_i, out_j = np.meshgrid(np.arange(out_H), np.arange(out_W), indexing='ij')

        # Position in the input
        input_row = out_i[None, None, :, :] * self.stride + pool_row
        input_col = out_j[None, None, :, :] * self.stride + pool_col
        # Shape: (batch, channels, out_H, out_W)

        # Route the gradient
        batch_idx = np.arange(batch)[:, None, None, None]
        channel_idx = np.arange(channels)[None, :, None, None]

        np.add.at(dX, 
                (batch_idx, channel_idx, input_row, input_col), 
                dout)
        
        return dX
