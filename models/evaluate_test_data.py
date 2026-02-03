import sys
import os
# Ensure the project root is added to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10

# Path to the CIFAR-10 dataset
CIFAR_PATH = r"C:\Users\user\OneDrive - TechnoVal\Desktop\Scripts\ML\cv-transition-lab\data\cifar-10-batches-py"

# Path to the directory containing all models
MODELS_DIR = "models_tf_deep"

# Load the test data
print("[LOADING TEST DATA...]")
(X_train, y_train), (X_test, y_test) = load_cifar10(CIFAR_PATH)

# Flatten if needed
if X_test.ndim == 4:
    X_test = X_test.reshape(X_test.shape[0], -1)

X_test = X_test.astype(np.float32)

# Preprocess the test data using training statistics
print("[PREPROCESSING TEST DATA...]")
mean_val = X_train.mean(axis=0, keepdims=True) if X_train.ndim == 2 else X_train.reshape(X_train.shape[0], -1).mean(axis=0, keepdims=True)
std_val = X_train.std(axis=0, keepdims=True) + 1e-8 if X_train.ndim == 2 else X_train.reshape(X_train.shape[0], -1).std(axis=0, keepdims=True) + 1e-8

X_test -= mean_val
X_test /= std_val

print(f"Test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Iterate through all models in the directory
for model_name in os.listdir(MODELS_DIR):
    model_path = os.path.join(MODELS_DIR, model_name, "best_model.keras")
    if os.path.exists(model_path):
        print(f"\nEvaluating model: {model_name}")
        try:
            # Load the model
            model = keras.models.load_model(model_path)
            print("Model loaded successfully!")

            # Evaluate the model on the test data
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
        except Exception as e:
            print(f"Failed to evaluate model {model_name}: {e}")
    else:
        print(f"Model file not found for {model_name}")