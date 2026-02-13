"""
Configuration file for hyperparameters.
"""

# Architecture
NUM_CONV1_FILTERS = 32
NUM_CONV2_FILTERS = 64
KERNEL_SIZE = 5
FC_HIDDEN_SIZE = 128

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20
MOMENTUM = 0.9

# Regularization
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 0.0001

# Data
DATA_PATH = "data/raw"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
