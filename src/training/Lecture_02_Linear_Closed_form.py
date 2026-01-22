import numpy as np
import sys
sys.path.append('C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab')
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10
path = 'C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\data\\cifar-10-batches-py\\'
(X_train, y_train), (X_test, y_test) = load_cifar10(path)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

rng = np.random.default_rng(42)
idx = rng.permutation(X_train.shape[0])
N = 10000  # CV size
cv_idx = idx[:N]
tr_idx = idx[N:]  # remaining go to train

X_cv = X_train[cv_idx]
y_cv = y_train[cv_idx]
X_tr = X_train[tr_idx]
y_tr = y_train[tr_idx]

print("Train:", np.shape(X_tr), np.shape(y_tr))
print("CV:", np.shape(X_cv), np.shape(y_cv))
print("Test:", np.shape(X_test), np.shape(y_test))

# Step number one, data preprocessing - mean subtraction
# Centering data around zero helps numerical stability and often improves accruacy

# 1. mean subtraction
mean_img = X_tr.mean(axis=0)
X_tr = X_tr - mean_img
X_cv = X_cv - mean_img

# Visualize the effect of mean subtraction
print("\n" + "="*50)
print("MEAN SUBTRACTION EFFECT")
print("="*50)
X_train_subset = X_train[tr_idx]  # Original training subset before preprocessing
print("Before mean subtraction:")
print(f"  Min: {X_train_subset.min():.2f}, Max: {X_train_subset.max():.2f}")
print(f"  Mean: {X_train_subset.mean():.2f}, Std: {X_train_subset.std():.2f}")

print("\nAfter mean subtraction:")
print(f"  Min: {X_tr.min():.2f}, Max: {X_tr.max():.2f}")
print(f"  Mean: {X_tr.mean():.2f}, Std: {X_tr.std():.2f}")

# 2. Feature scaling (normalization)
# IMPORTANT: Compute std ONLY from training data
std_img = X_tr.std(axis=0)
X_tr = X_tr / std_img
X_cv = X_cv / std_img

# Visualize the effect of feature scaling
print("\n" + "="*50)
print("FEATURE SCALING EFFECT")
print("="*50)
print("After feature scaling:")
print(f"  Min: {X_tr.min():.2f}, Max: {X_tr.max():.2f}")
print(f"  Mean: {X_tr.mean():.2f}, Std: {X_tr.std():.2f}")
print("  (Features now normalized - much more suitable for regularization!)")

# 3. Add Bias

# Add Bias column (ones) to the right of features
X_tr = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
X_cv = np.hstack([X_cv, np.ones((X_cv.shape[0], 1))])
X_test = X_test - mean_img  # centering test
X_test = X_test / std_img   # scale test (using training std!)
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

print("After bias, train shape:", X_tr.shape)  # Should be (40000, 3073)
print("After bias, CV shape:", X_cv.shape)      # Should be (10000, 3073)
print("After bias, test shape:", X_test.shape)  # Should be (10000, 3073)

# 3. One-hot encoding (we want 1 for correct class and 0 else-where)

num_classes = 10

def one_hot_encoding(labels, num_classes):
    """Convert integer labels to one-hot vectors"""
    N = labels.shape[0]
    Y = np.zeros((N, num_classes))
    Y[np.arange(N),labels] = 1
    return Y

Y_tr = one_hot_encoding(y_tr, num_classes)
# We don't need one-hot for CV/test since we'll use argmax for predictions

print("Y_tr shape:", Y_tr.shape)  # Should be (40000, 10)
print("First 3 labels:", y_tr[:3])
print("First 3 one-hot:\n", Y_tr[:3])

# ============================================================
# SECTION 2: HYPERPARAMETER TUNING (Lambda search on CV)
# ============================================================

# Compute X^T X (needed for all lambda values)
XTX = np.dot(X_tr.T, X_tr)
print("\nX^T X shape:", XTX.shape)

# Compute X^T Y (needed for all lambda values)
XTY = X_tr.T @ Y_tr
print("X^T Y shape:", XTY.shape)

# Identity matrix for regularization (don't regularize bias)
I = np.eye(3073)
I[-1, -1] = 0


lambda_values = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10]
cv_accuracies = []

for lambda_reg in lambda_values:
    reg_term = lambda_reg * I
    W = np.linalg.solve(XTX + reg_term, XTY)
    print(f"λ={lambda_reg:7.1e}  →  W[0,0]={W[0,0]:.6f}")  # Check if W changes
    scores_cv = X_cv @ W
    y_pred_cv = scores_cv.argmax(axis=1)
    acc_cv = np.mean(y_pred_cv == y_cv)
    cv_accuracies.append(acc_cv)
    print(f"                    CV Accuracy: {acc_cv:.4f}")

best_idx = np.argmax(cv_accuracies)
best_lambda = lambda_values[best_idx]
best_acc = cv_accuracies[best_idx]
print(f"\n✓ Best λ={best_lambda} with CV Accuracy: {best_acc:.4f}")

# ============================================================
# SECTION 3: FINAL EVALUATION (Best lambda on Test)
# ============================================================

lambda_reg = best_lambda  # Use the best lambda found from CV tuning
reg_term = lambda_reg * I

W = np.linalg.solve(XTX + reg_term, XTY)
bias = W[-1, :]  # Last row
print("\nBias per class:", bias)
print("W shape:", W.shape)

# Evaluate on CV
scores_cv = X_cv @ W
y_pred_cv = scores_cv.argmax(axis=1)
acc_cv = np.mean(y_pred_cv == y_cv)
print(f"\nCV Accuracy (lambda={lambda_reg}): {acc_cv:.4f}")

# Evaluate on Test
scores_test = X_test @ W
y_pred_test = scores_test.argmax(axis=1)
acc_test = np.mean(y_pred_test == y_test)
print(f"Test Accuracy (lambda={lambda_reg}): {acc_test:.4f}")

