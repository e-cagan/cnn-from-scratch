"""
Module for evaluating model.
"""

import numpy as np

from models.cnn import CNNModel
from data.load_mnist import load_mnist, get_batches
from utils.visualization import plot_confusion_matrix, show_predictions
from utils.metrics import confusion_matrix
from train import evaluate


# CONSTNANTS
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20
CHECKPOINT_PATH = "checkpoints/fully_trained_best_model.npy"


def main():
    # Load model and data
    model = CNNModel()
    model.load(CHECKPOINT_PATH)
    data = load_mnist()
    test_images, test_labels = data["test"]
    class_names = None
    
    # Calculate test accuracy
    test_acc = evaluate(model, test_images, test_labels, BATCH_SIZE)
    
    # Take predictions using entire test set
    predictions = []
    for batch in get_batches(test_images, test_labels, BATCH_SIZE, shuffle=False):
        x = batch[0]
        for layer in model.layers[:-1]:
            x = layer.forward(x)
        preds = np.argmax(x, axis=1)
        predictions.extend(preds)
        class_names = sorted(list(set(test_labels)))
        class_names = [str(c) for c in class_names]
    
    # Confusion matrix
    cm = confusion_matrix(predictions, test_labels)
    
    # Print results
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Visualize results
    plot_confusion_matrix(cm, class_names)
    show_predictions(test_images, test_labels, predictions)

if __name__ == '__main__':
    main()
