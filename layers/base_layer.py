"""
Module for base abstract class and methods for layers.
"""

from abc import ABC, abstractmethod

# Define the base layer class
class BaseLayer(ABC):
    """
    Base layer class interface for layers.
    """
    
    def __init__(self):
        self.cache = {}
        self.training = True

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    # Getter and setter methods
    def get_params(self):
        return dict()
    
    def set_params(self, params):
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
    
    def __call__(self, x):
        return self.forward(x)
