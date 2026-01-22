import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab')
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10
path = 'C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\data\\cifar-10-batches-py\\'
(X_train, y_train), (X_test, y_test) = load_cifar10(path)

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

# ============================================================
# SECTION 1: DATA PREPROCESSING
# ============================================================

# 1. Mean subtraction
mean_img = X_tr.mean(axis=0)
X_tr = X_tr - mean_img
X_cv = X_cv - mean_img

# 2. Feature scaling (normalization)
std_img = X_tr.std(axis=0)
X_tr = X_tr / std_img
X_cv = X_cv / std_img

print("\n" + "="*50)
print("FEATURE SCALING EFFECT")
print("="*50)
print("After feature scaling:")
print(f"  Min: {X_tr.min():.2f}, Max: {X_tr.max():.2f}")
print(f"  Mean: {X_tr.mean():.2f}, Std: {X_tr.std():.2f}")

# 3. Add Bias
X_tr = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
X_cv = np.hstack([X_cv, np.ones((X_cv.shape[0], 1))])
X_test = X_test - mean_img  # centering test
X_test = X_test / std_img   # scale test
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

print("After bias, train shape:", X_tr.shape)
print("After bias, CV shape:", X_cv.shape)
print("After bias, test shape:", X_test.shape)

# 4. One-hot encoding
num_classes = 10

def one_hot_encoding(labels, num_classes):
    """Convert integer labels to one-hot vectors"""
    N = labels.shape[0]
    Y = np.zeros((N, num_classes))
    Y[np.arange(N), labels] = 1
    return Y

Y_tr = one_hot_encoding(y_tr, num_classes)

print("Y_tr shape:", Y_tr.shape)

# ============================================================
# SECTION 2: GRADIENT DESCENT TRAINING
# ============================================================

print("\n" + "="*60)
print("GRADIENT DESCENT TRAINING")
print("="*60)

# Hyperparameters
learning_rate = 0.001
num_iterations = 500
lambda_reg = 10.0  # Match closed-form's best lambda

# Initialize weights randomly
np.random.seed(42)
W = np.random.randn(3073, 10) * 0.01

# Storage for visualization
loss_history = []
mse_history = []
reg_history = []

print(f"Initial W shape: {W.shape}")
print(f"Learning rate: {learning_rate}")
print(f"Num iterations: {num_iterations}")
print(f"Lambda (regularization): {lambda_reg}")

for iteration in range(num_iterations):
    # Forward pass: compute predictions
    scores = X_tr @ W  # shape: (40000, 10)
    
    # Step 2: Compute MSE (Prediction Error)
    errors = scores - Y_tr  # (40000, 10)
    mse = np.mean(errors**2)  # Scalar

    # Step 3: Compute Regularization (Don't regularize bias)
    reg = lambda_reg * np.sum(W[:-1]**2)  # W[:-1] excludes bias
    
    # Step 4: Total loss
    total_loss = mse + reg
    
    # Store for visualization
    loss_history.append(total_loss)
    mse_history.append(mse)
    reg_history.append(reg)
    
    if (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration+1}: Loss = {total_loss:.6f}, MSE = {mse:.6f}, Reg = {reg:.6f}")
    
    # Compute gradients (scaled by number of samples)
    dW = (X_tr.T @ errors) / len(X_tr) + lambda_reg * W
    
    # Don't regularize bias gradient
    dW[-1] = (X_tr[:, -1] @ errors) / len(X_tr)
    
    # Update weights (move downhill)
    W = W - learning_rate * dW

# ============================================================
# SECTION 3: EVALUATION
# ============================================================

print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

# Evaluate on CV
scores_cv = X_cv @ W
y_pred_cv = scores_cv.argmax(axis=1)
acc_cv = np.mean(y_pred_cv == y_cv)
print(f"\nCV Accuracy: {acc_cv:.4f}")

# Evaluate on Test
scores_test = X_test @ W
y_pred_test = scores_test.argmax(axis=1)
acc_test = np.mean(y_pred_test == y_test)
print(f"Test Accuracy: {acc_test:.4f}")

# Show bias values
bias = W[-1, :]
print(f"\nBias per class: {bias}")
print("W shape:", W.shape)

print("\n✓ Training complete!")

# ============================================================
# SECTION 4: VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("VISUALIZATION: GRADIENT DESCENT CONVERGENCE")
print("="*60)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Total Loss over iterations
axes[0].plot(loss_history, linewidth=2, label='Total Loss', color='red')
axes[0].set_xlabel('Iteration', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Gradient Descent: Total Loss Convergence', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)

# Plot 2: MSE vs Regularization components
axes[1].plot(mse_history, linewidth=2, label='MSE (data fit)', color='blue')
axes[1].plot(reg_history, linewidth=2, label='Regularization (λ||W||²)', color='green')
axes[1].plot(loss_history, linewidth=2, label='Total Loss', color='red', linestyle='--')
axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel('Loss Component', fontsize=12)
axes[1].set_title('Loss Components Over Training', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\reports\\gradient_descent_convergence.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to: reports/gradient_descent_convergence.png")
plt.show()

# Print convergence statistics
print(f"\nConvergence Statistics:")
print(f"  Initial Loss: {loss_history[0]:.6f}")
print(f"  Final Loss: {loss_history[-1]:.6f}")
print(f"  Loss Reduction: {loss_history[0] - loss_history[-1]:.6f} ({(1 - loss_history[-1]/loss_history[0])*100:.1f}%)")
print(f"  Final MSE: {mse_history[-1]:.6f}")
print(f"  Final Regularization: {reg_history[-1]:.6f}")
