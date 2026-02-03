import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10

# Path to the saved model
MODEL_PATH = "models_tf_deep/3layer_relu/best_model.keras"  # Update for other models if needed

# Path to the CIFAR-10 dataset
CIFAR_PATH = "/content/drive/MyDrive/datasets/cifar-10-batches-py"

# Load the test data
(_, _), (X_test, y_test) = load_cifar10(CIFAR_PATH)

# Preprocess the test data (same as training preprocessing)
X_test = X_test.astype(np.float32)
mean_img = X_test.mean(axis=0, keepdims=True)  # Use training mean if available
std_img = X_test.std(axis=0, keepdims=True) + 1e-8
X_test -= mean_img
X_test /= std_img

# Load the saved model
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")