"""
Test evaluation script for final trained weights (activation_only mode).
Loads final weights for each activation function and evaluates on CIFAR-10 test set.
"""

import numpy as np
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = r"C:\Users\user\OneDrive - TechnoVal\Desktop\Scripts\ML\cv-transition-lab"
sys.path.insert(0, PROJECT_ROOT)

from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10

# ============================================================
# LOAD TEST DATA & PREPROCESSING
# ============================================================
print("[LOADING TEST DATA...]")
CIFAR_PATH = r"C:\Users\user\OneDrive - TechnoVal\Desktop\Scripts\ML\cv-transition-lab\data\cifar-10-batches-py\\"
(X_train, y_train), (X_test, y_test) = load_cifar10(CIFAR_PATH)

# Flatten if needed
if X_test.ndim == 4:
    X_test = X_test.reshape(X_test.shape[0], -1)

X_test = X_test.astype(np.float32)

# Load preprocessing params from training (using training mean/std for consistency)
# For now, use training set statistics (in practice, these would be saved)
print("[PREPROCESSING TEST DATA...]")
# We'll compute from training set loaded in this script, or you can save/load preprocessing
mean_val = X_train.mean(axis=0, keepdims=True) if X_train.ndim == 2 else X_train.reshape(X_train.shape[0], -1).mean(axis=0, keepdims=True)
std_val = X_train.std(axis=0, keepdims=True) + 1e-8 if X_train.ndim == 2 else X_train.reshape(X_train.shape[0], -1).std(axis=0, keepdims=True) + 1e-8

X_test -= mean_val
X_test /= std_val

print(f"Test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ============================================================
# LOAD ACTIVATIONS & HELPER FUNCTIONS
# ============================================================

ACTIVATIONS = {
    "relu": lambda x: np.maximum(0, x),
    "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
    "tanh": lambda x: np.tanh(x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
    "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
}

def forward_pass(X, W1, b1, W2, b2, activation_fn):
    """Simple forward pass for 2-layer net."""
    z1 = X @ W1 + b1
    a1 = activation_fn(z1)
    scores = a1 @ W2 + b2
    return scores

def predict(X, W1, b1, W2, b2, activation_fn):
    """Predict class labels."""
    scores = forward_pass(X, W1, b1, W2, b2, activation_fn)
    return np.argmax(scores, axis=1)

# ============================================================
# EVALUATE EACH ACTIVATION FUNCTION - BOTH MODES
# ============================================================

final_dir = os.path.join(os.path.dirname(__file__), "final")
activations_to_test = ["relu", "leaky_relu", "tanh", "sigmoid", "gelu"]
modes_to_test = ["activation_only", "best_practice"]

results_by_mode = {}

for mode in modes_to_test:
    print("\n" + "="*90)
    print(f"TEST SET ACCURACY - {mode.upper()} MODE")
    print("="*90)
    
    results = {}
    
    for act in activations_to_test:
        # Load weights
        weights_file = os.path.join(final_dir, f"NN_{act}_{mode}_final_weights.npz")
        meta_file = os.path.join(final_dir, f"NN_{act}_{mode}_final_metadata.json")
        
        if not os.path.exists(weights_file):
            print(f"✗ {act:12s} | Weights file not found")
            continue
        
        # Load weights
        npz = np.load(weights_file)
        W1 = npz["W1"]
        b1 = npz["b1"]
        W2 = npz["W2"]
        b2 = npz["b2"]
        
        # Load metadata
        metadata = {}
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                metadata = json.load(f)
        
        # Evaluate on test set
        test_preds = predict(X_test, W1, b1, W2, b2, ACTIVATIONS[act])
        test_acc = np.mean(test_preds == y_test)
        
        train_cv_acc = metadata.get("final_cv_accuracy", "N/A")
        
        results[act] = {
            "test_acc": test_acc,
            "train_cv_acc": train_cv_acc,
            "metadata": metadata
        }
        
        print(f"✓ {act:12s} | Train CV: {train_cv_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    results_by_mode[mode] = results
    
    # Summary for this mode
    print("\n" + "-"*90)
    print(f"Summary ({mode})")
    print("-"*90)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["test_acc"], reverse=True)
    for act, res in sorted_results:
        print(f"  {act:12s} | Test Acc: {res['test_acc']:.4f}")
    
    best_act = sorted_results[0][0]
    best_acc = sorted_results[0][1]['test_acc']
    print(f"\n  ⭐ Best in {mode}: {best_act} ({best_acc:.4f})")

# ============================================================
# FINAL COMPARISON
# ============================================================
print("\n\n" + "="*90)
print("FINAL COMPARISON - BEST FROM EACH MODE")
print("="*90)

for mode in modes_to_test:
    if mode in results_by_mode:
        sorted_results = sorted(results_by_mode[mode].items(), key=lambda x: x[1]["test_acc"], reverse=True)
        if sorted_results:
            best_act = sorted_results[0][0]
            best_acc = sorted_results[0][1]['test_acc']
            train_cv = sorted_results[0][1]['train_cv_acc']
            print(f"{mode:20s} | Best: {best_act:12s} | Train CV: {train_cv:.4f} | Test Acc: {best_acc:.4f}")

print("\n[EVALUATION COMPLETE]")
