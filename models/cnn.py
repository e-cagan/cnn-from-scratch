"""
Module to define and use the implemented CNN network.
"""

import numpy as np
from .base_model import BaseModel

# Import layers
from layers.conv import ConvLayer
from layers.conv_vec import ConvLayerVec
from layers.fc import FCLayer
from layers.flatten import Flatten
from layers.maxpool import MaxPool
from layers.relu import ReLU
from layers.softmax import Softmax


# Define a class for CNN network
class CNNModel(BaseModel):
    """
    A class for CNN network.
    """

    def __init__(self):
        super().__init__()
        
        # Add layers to model
        # First convolutional layer pattern with ReLU activation
        self.add_layer(ConvLayerVec(in_channels=1, out_channels=32, kernel_size=5, padding=2))
        self.add_layer(ReLU())
        self.add_layer(MaxPool(pool_size=2, stride=2))
        
        # Second convolutional layer pattern with ReLU activation
        self.add_layer(ConvLayerVec(in_channels=32, out_channels=64, kernel_size=5, padding=2))
        self.add_layer(ReLU())
        self.add_layer(MaxPool(pool_size=2, stride=2))

        # Flatten layer
        self.add_layer(Flatten())

        # Fully connected layer with ReLU activation
        self.add_layer(FCLayer(in_features=3136, out_features=128))
        self.add_layer(ReLU())

        # Output layer with softmax activation
        self.add_layer(FCLayer(in_features=128, out_features=10))
        self.add_layer(Softmax())

    # Forward and backward propagation functions
    def forward(self, x, labels):
        """
        Forward propagation function for CNN.
        """

        # Iterate trough layers
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:   # Last layer -> Softmax (Handled differently from others)
                x = layer.forward(x, labels)
            else:
                x = layer.forward(x)

        # Return loss
        return x
    
    def backward(self):
        """
        Backward propagation function for CNN.
        """

        # Call softmax backpropagation
        dout = self.layers[-1].backward()

        # Iterate trough rest of the labels in the reverse order
        for layer in reversed(self.layers[:-1]):
            dout = layer.backward(dout)

        return dout
