"""
Module for convolution layer.
"""

import numpy as np
from .base_layer import BaseLayer


class ConvLayer(BaseLayer):
    """
    A class for convolution layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weigths (He initialization) and biases
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.biases = np.zeros(shape=(out_channels,))
        self.dW = None
        self.db = None

    # Functions for forward and backward propagation
    def forward(self, x):
        """
        Forward propagation function for convolution layer.
        """

        # Extract the values in the input
        batch_size, in_channels, height, width = x.shape

        # Apply padding to matrix to create input
        padded_input = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))    #  batch, channel, height, width
                                                                                                                #  (zero), (zero), (padding), (padding) --> Padding types and amounts

        # Calculate output height and width
        out_H = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_W = (width  + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Create output matrix
        output = np.zeros(shape=(batch_size, self.out_channels, out_H, out_W))

        # Iterate trough input
        for n in range(batch_size):
            for f in range(self.out_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        
                        # Apply filter to input matrix
                        # Slice the input matrix
                        sliced_matrix = padded_input[n, :, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size]

                        # Apply filter to sliced matrix
                        filter_applied = np.sum(sliced_matrix * self.weights[f])

                        # Add biases
                        output[n, f, i, j] = filter_applied + self.biases[f]

        # Cache padded input matrix and weights in the cache in order to use them in the backward propagations
        self.cache["padded_input"] = padded_input
        self.cache["weights"] = self.weights

        return output

    def backward(self, dout):
        """
        Backward propagation function for convolution layer.
        """

        # Unpack the values of the dout shape
        batch_size, out_channels, out_H, out_W = dout.shape

        # Take the padded input matrix and weights from cache
        padded_input = self.cache["padded_input"]
        weights = self.cache["weights"]
        
        # Find the bias gradients and create weight grads matrix
        db = np.sum(dout, axis=(0, 2, 3))
        dW = np.zeros(shape=weights.shape)
        dX = None

        # Create padded input gradient matrix
        dX_padded = np.zeros(shape=padded_input.shape)

        # Iterate trough output
        for n in range(batch_size):
            for f in range(out_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        
                        # Slice the input matrix
                        sliced_matrix = padded_input[n, :, i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size]

                        # Take the gradients for weights
                        dW[f] += sliced_matrix * dout[n, f, i, j]

                        # Take the gradients for padded matrix
                        dX_padded[n, :, i * self.stride : i * self.stride + self.kernel_size, j * self.stride : j * self.stride + self.kernel_size] += self.weights[f] * dout[n, f, i, j]
        
        # Discard the paddings based on padding size or not
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        elif self.padding == 0:
            dX = dX_padded
        else:
            dX = None

        # Store the gradients for weights and biases
        self.dW = dW
        self.db = db
        
        return dX

    # Getter and setter functions for parameters
    def get_params(self):
        return {
            "W": {"value": self.weights, "grad": self.dW},
            "b": {"value": self.biases, "grad": self.db}
        }

    def set_params(self, params):
        # Check if key exists before setting it
        if "W" in params:
            self.weights = params["W"]
        if "b" in params:
            self.biases = params["b"]
