"""
Module for base abstract class and methods for optimizers.
"""

from abc import ABC, abstractmethod


# Define the base optimizer class
class BaseOptim(ABC):
    """
    Base optimizer class interface for optimizers.
    """

    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, layer):
        """
        Update method for optimizers
        """

        pass
