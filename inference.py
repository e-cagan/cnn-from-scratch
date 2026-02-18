"""
Module for inferencing predictions.
"""

import matplotlib.pyplot as plt
import numpy as np

from models.cnn import CNNModel
from data.load_mnist import load_mnist


# CONSTNANTS
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20
CHECKPOINT_PATH = "checkpoints/fully_trained_best_model.npy"


def predict_single(model, image):
    """
    A function for predicting a single output.
    """

    # Preprocess the image
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    # Use image as an input
    x = image
    for layer in model.layers[:-1]:
        x = layer.forward(x)
    
    # Return the most confident prediction
    pred = np.argmax(x, axis=1)

    return pred[0]

def main():
    model = CNNModel()
    model.load(CHECKPOINT_PATH)
    
    # Load a sample MNIST image
    data = load_mnist()
    test_images, test_labels = data["test"]
    
    idx = np.random.randint(len(test_images))
    image = test_images[idx]
    true_label = test_labels[idx]
    
    pred = predict_single(model, image)
    
    print(f"Prediction: {pred}, True: {true_label}")
    
    # Visualize
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Pred: {pred}, True: {true_label}")

    # Save figure
    plt.savefig("/home/cagan/cnn-from-scratch/data/plots/pred_figure.png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
