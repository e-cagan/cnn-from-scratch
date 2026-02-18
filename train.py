"""
Module for training the model.
"""

import numpy as np

# Import implemented components
from models.cnn import CNNModel
from optimizers.adam import Adam
from data.load_mnist import get_batches, load_mnist
from utils.visualization import plot_training_curves

# CONSTNANTS
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20
CHECKPOINT_PATH = "checkpoints/best_model.npy"

# Define a function for evaluation
def evaluate(model, images, labels, batch_size):
    """
    Function for evaluating model. Calculates validation accuracy.
    """

    # Evaluate model
    model.eval()

    # Initialize the variables for correct and total amount of predictions
    correct = 0
    total = 0

    # Iterate trough batches
    for batch_images, batch_labels in get_batches(images, labels, batch_size, shuffle=False):
       # Iterate trough layers except softmax
       x = batch_images
       for layer in model.layers[:-1]:
           x = layer.forward(x)
       
       # Predict the result
       predictions = np.argmax(x, axis=1)
       
       # Count the correct predictions and total
       correct += np.sum(predictions == batch_labels)
       total += len(batch_labels)

    return correct / total

def train():
    """
    Function for training model. Saves the best model while training.
    """

    # Load data
    data = load_mnist()
    train_images, train_labels = data["train"]
    val_images, val_labels = data["val"]
    test_images, test_labels = data["test"]

    # Models and metrics
    model = CNNModel()
    optimizer = Adam(learning_rate=LEARNING_RATE)

    best_val_accuracy = 0
    train_losses = []
    val_accuracies = []

    # Iterate amount of epochs
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_losses = []
        
        # Iterate trough batches
        for batch_images, batch_labels in get_batches(train_images, train_labels, BATCH_SIZE):
            loss = model.forward(batch_images, batch_labels)
            model.backward()
            model.update(optimizer)
            epoch_losses.append(loss)
        
        # End of the epoch
        mean_loss = np.mean(epoch_losses)
        train_losses.append(mean_loss)
        
        # Validation
        val_acc = evaluate(model, val_images, val_labels, BATCH_SIZE)
        val_accuracies.append(val_acc)
        
        # Log
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {mean_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save the best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            model.save(CHECKPOINT_PATH)
            print(f"Best model saved (val_acc: {val_acc:.4f})")

    # Test accuracy
    model.load(CHECKPOINT_PATH)
    test_acc = evaluate(model, test_images, test_labels, BATCH_SIZE)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Plot the learning curve
    plot_training_curves(train_losses, val_accuracies)

# Test the train function
if __name__ == '__main__':
    train()
