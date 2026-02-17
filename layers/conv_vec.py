"""
Vectorized implementation of convolution layer.
"""

import numpy as np
from .base_layer import BaseLayer


# Define helper functions to transform image to column or otherwise for windows
def img2col(x, kernel_size, stride, padding):
    """
    A helper function for converting image to column.
    """

    # Unpack the input values based on shape
    batch_size, channels, H, W = x.shape

    # Calculate output size
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1

    # Create column matrix for output
    col = np.zeros(shape=(batch_size * out_H * out_W, channels * kernel_size * kernel_size))

    # Iterate trough windows
    col_idx = 0
    for n in range(batch_size):
        for i in range(out_H):
            for j in range(out_W):
                # Window extract et
                window = x[n, :, i * stride : i * stride + kernel_size, j * stride : j * stride + kernel_size]
                # Flatten et
                col[col_idx] = window.flatten()
                col_idx += 1

    return col

def col2img(col, output_shape, kernel_size, stride, padding):
    """
    A helper function for converting column matrix to image.
    """

    # Unpack the output shape
    batch_size, channels, H, W = output_shape

    # Calculate height and width backwards
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1

    # Create image matrix
    img = np.zeros(shape=output_shape)

    # Revert the columns back to image
    col_idx = 0
    for n in range(batch_size):
        for i in range(out_H):
            for j in range(out_W):
                # Column'u al, reshape et
                window = col[col_idx].reshape(channels, kernel_size, kernel_size)
                # Image'a ekle (+=, overlapping için)
                img[n, :, i*stride : i*stride+kernel_size, j*stride : j*stride+kernel_size] += window
                col_idx += 1

    return img


class ConvLayerVec(BaseLayer):
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
        Forward propagation function for vectorized convolution layer.
        """

        # Apply padding to matrix to create input
        padded_input = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))    #  batch, channel, height, width
                                                                                                                #  (zero), (zero), (padding), (padding) --> Padding types and amounts

        # Convert padded matrix to column matrix
        padded_x = img2col(padded_input, self.kernel_size, self.stride, padding=0)

        # Reshape the weight matrix
        W_col = self.weights.reshape(self.out_channels, -1)       # -1 to calculate automatically

        # Calculate the output matrix
        out_matrix = np.dot(padded_x, W_col.T)

        # Add biases
        out_matrix += self.biases

        # Calculate the output width and height
        batch, in_channels, H, W = x.shape
        out_H = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        out_W = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        # Reshape the output matrix
        output = out_matrix.reshape(batch, out_H, out_W, self.out_channels)
        output = output.transpose(0, 3, 1, 2)  # (batch, out_channels, out_H, out_W)

        # Cache the padded input matrix, shape of the input and weight matrix of column matrix to use it on backward propagation
        self.cache["col"] = padded_x
        self.cache["x_shape"] = x.shape
        self.cache["W_col"] = W_col

        return output

    def backward(self, dout):
        """
        Backward propagation function for vectorized convolution layer.
        """

        # Unpack the values from cache
        padded_x = self.cache["col"]
        x_shape = self.cache["x_shape"]
        W_col = self.cache["W_col"]

        # Reshape the output dimension
        batch, out_channels, out_H, out_W = dout.shape
        dout_col = dout.transpose(0, 2, 3, 1).reshape(-1, out_channels)

        # Calculate weight gradients
        dW = np.dot(padded_x.T, dout_col)
        dW = dW.T.reshape(self.weights.shape)

        # Calculate bias gradients
        db = np.sum(dout_col, axis=0)

        # Calculate input gradients
        dX_col = np.dot(dout_col, W_col)

        # Convert column to image back
        batch, in_channels, H, W = x_shape
        padded_shape = (batch, in_channels, H + 2*self.padding, W + 2*self.padding)
        dX_padded = col2img(dX_col, padded_shape, self.kernel_size, self.stride, padding=0)

        # Extract the paddings (if any)
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        # Store the weight and bias gradients
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
