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
# SECTION 2: GRADIENT DESCENT WITH L2 REGULARIZATION (RIDGE)
# ============================================================

print("\n" + "="*60)
print("GRADIENT DESCENT TRAINING - L2 REGULARIZATION (RIDGE)")
print("="*60)

# Hyperparameters
learning_rate = 0.001
num_iterations = 500
lambda_reg = 10.0

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
print(f"Regularization type: L2 (Ridge)")

for iteration in range(num_iterations):
    # Forward pass: compute predictions
    scores = X_tr @ W  # shape: (40000, 10)
    
    # Compute MSE (Prediction Error)
    errors = scores - Y_tr  # (40000, 10)
    mse = np.mean(errors**2)  # Scalar

    # L2 Regularization: λ * ||W||²
    reg = lambda_reg * np.sum(W[:-1]**2)  # W[:-1] excludes bias
    
    # Total loss
    total_loss = mse + reg
    
    # Store for visualization
    loss_history.append(total_loss)
    mse_history.append(mse)
    reg_history.append(reg)
    
    if (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration+1}: Loss = {total_loss:.6f}, MSE = {mse:.6f}, Reg = {reg:.6f}")
    
    # Compute gradients - L2 gradient: 2λW
    dW = (X_tr.T @ errors) / len(X_tr) + 2 * lambda_reg * W
    
    # Don't regularize bias gradient
    dW[-1] = (X_tr[:, -1] @ errors) / len(X_tr)
    
    # Update weights (move downhill)
    W = W - learning_rate * dW

# ============================================================
# SECTION 3: EVALUATION
# ============================================================

print("\n" + "="*60)
print("FINAL EVALUATION - L2 REGULARIZATION")
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

print("\n✓ L2 Training complete!")

# Store L2 results
W_l2 = W.copy()
acc_cv_l2 = acc_cv
acc_test_l2 = acc_test
loss_history_l2 = loss_history.copy()

# ============================================================
# SECTION 4: GRADIENT DESCENT WITH L1 REGULARIZATION (LASSO)
# ============================================================

print("\n" + "="*60)
print("GRADIENT DESCENT TRAINING - L1 REGULARIZATION (LASSO)")
print("="*60)

# Reset weights
np.random.seed(42)
W = np.random.randn(3073, 10) * 0.01

# Storage for visualization
loss_history = []
mse_history = []
reg_history = []

print(f"Learning rate: {learning_rate}")
print(f"Num iterations: {num_iterations}")
print(f"Lambda (regularization): {lambda_reg}")
print(f"Regularization type: L1 (Lasso)")

for iteration in range(num_iterations):
    # Forward pass
    scores = X_tr @ W
    errors = scores - Y_tr
    mse = np.mean(errors**2)

    # L1 Regularization: λ * ||W||₁ (sum of absolute values)
    reg = lambda_reg * np.sum(np.abs(W[:-1]))
    
    total_loss = mse + reg
    
    loss_history.append(total_loss)
    mse_history.append(mse)
    reg_history.append(reg)
    
    if (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration+1}: Loss = {total_loss:.6f}, MSE = {mse:.6f}, Reg = {reg:.6f}")
    
    # Compute gradients - L1 gradient: λ * sign(W)
    dW = (X_tr.T @ errors) / len(X_tr) + lambda_reg * np.sign(W)
    
    # Don't regularize bias
    dW[-1] = (X_tr[:, -1] @ errors) / len(X_tr)
    
    W = W - learning_rate * dW

# Evaluation
print("\n" + "="*60)
print("FINAL EVALUATION - L1 REGULARIZATION")
print("="*60)

scores_cv = X_cv @ W
y_pred_cv = scores_cv.argmax(axis=1)
acc_cv = np.mean(y_pred_cv == y_cv)
print(f"\nCV Accuracy: {acc_cv:.4f}")

scores_test = X_test @ W
y_pred_test = scores_test.argmax(axis=1)
acc_test = np.mean(y_pred_test == y_test)
print(f"Test Accuracy: {acc_test:.4f}")

# Count sparse weights (exactly zero or very close)
num_zero_weights = np.sum(np.abs(W[:-1]) < 1e-6)
total_weights = W[:-1].size
sparsity_percent = (num_zero_weights / total_weights) * 100
print(f"\nSparsity: {num_zero_weights}/{total_weights} weights are zero ({sparsity_percent:.2f}%)")
print("W shape:", W.shape)

print("\n✓ L1 Training complete!")

# Store L1 results
W_l1 = W.copy()
acc_cv_l1 = acc_cv
acc_test_l1 = acc_test
loss_history_l1 = loss_history.copy()

# ============================================================
# SECTION 5: GRADIENT DESCENT WITH ELASTIC NET (L1 + L2)
# ============================================================

print("\n" + "="*60)
print("GRADIENT DESCENT TRAINING - ELASTIC NET (L1 + L2)")
print("="*60)

# Elastic Net hyperparameters
alpha = 0.5  # Balance between L1 and L2 (0=L2 only, 1=L1 only)
lambda_l1 = lambda_reg * alpha
lambda_l2 = lambda_reg * (1 - alpha)

# Reset weights
np.random.seed(42)
W = np.random.randn(3073, 10) * 0.01

# Storage
loss_history = []
mse_history = []
reg_history = []

print(f"Learning rate: {learning_rate}")
print(f"Num iterations: {num_iterations}")
print(f"Lambda (total): {lambda_reg}")
print(f"Alpha (L1/L2 balance): {alpha}")
print(f"  L1 weight: {lambda_l1}")
print(f"  L2 weight: {lambda_l2}")
print(f"Regularization type: Elastic Net")

for iteration in range(num_iterations):
    # Forward pass
    scores = X_tr @ W
    errors = scores - Y_tr
    mse = np.mean(errors**2)

    # Elastic Net: α*λ*||W||₁ + (1-α)*λ*||W||²
    reg_l1 = lambda_l1 * np.sum(np.abs(W[:-1]))
    reg_l2 = lambda_l2 * np.sum(W[:-1]**2)
    reg = reg_l1 + reg_l2
    
    total_loss = mse + reg
    
    loss_history.append(total_loss)
    mse_history.append(mse)
    reg_history.append(reg)
    
    if (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration+1}: Loss = {total_loss:.6f}, MSE = {mse:.6f}, Reg = {reg:.6f}")
    
    # Elastic Net gradient: λ_l1 * sign(W) + 2 * λ_l2 * W
    dW = (X_tr.T @ errors) / len(X_tr) + lambda_l1 * np.sign(W) + 2 * lambda_l2 * W
    
    # Don't regularize bias
    dW[-1] = (X_tr[:, -1] @ errors) / len(X_tr)
    
    W = W - learning_rate * dW

# Evaluation
print("\n" + "="*60)
print("FINAL EVALUATION - ELASTIC NET")
print("="*60)

scores_cv = X_cv @ W
y_pred_cv = scores_cv.argmax(axis=1)
acc_cv = np.mean(y_pred_cv == y_cv)
print(f"\nCV Accuracy: {acc_cv:.4f}")

scores_test = X_test @ W
y_pred_test = scores_test.argmax(axis=1)
acc_test = np.mean(y_pred_test == y_test)
print(f"Test Accuracy: {acc_test:.4f}")

num_zero_weights = np.sum(np.abs(W[:-1]) < 1e-6)
sparsity_percent = (num_zero_weights / total_weights) * 100
print(f"\nSparsity: {num_zero_weights}/{total_weights} weights are zero ({sparsity_percent:.2f}%)")
print("W shape:", W.shape)

print("\n✓ Elastic Net Training complete!")

# Store Elastic Net results
W_elastic = W.copy()
acc_cv_elastic = acc_cv
acc_test_elastic = acc_test
loss_history_elastic = loss_history.copy()

# ============================================================
# SECTION 6: COMPARISON
# ============================================================

print("\n" + "="*70)
print("REGULARIZATION COMPARISON SUMMARY")
print("="*70)

print(f"\n{'Method':<15} {'CV Acc':<10} {'Test Acc':<10} {'Sparsity %':<12}")
print("-" * 70)

# L2
l2_sparsity = (np.sum(np.abs(W_l2[:-1]) < 1e-6) / total_weights) * 100
print(f"{'L2 (Ridge)':<15} {acc_cv_l2:<10.4f} {acc_test_l2:<10.4f} {l2_sparsity:<12.2f}")

# L1
l1_sparsity = (np.sum(np.abs(W_l1[:-1]) < 1e-6) / total_weights) * 100
print(f"{'L1 (Lasso)':<15} {acc_cv_l1:<10.4f} {acc_test_l1:<10.4f} {l1_sparsity:<12.2f}")

# Elastic Net
elastic_sparsity = (np.sum(np.abs(W_elastic[:-1]) < 1e-6) / total_weights) * 100
print(f"{'Elastic Net':<15} {acc_cv_elastic:<10.4f} {acc_test_elastic:<10.4f} {elastic_sparsity:<12.2f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss convergence
axes[0].plot(loss_history_l2, linewidth=2, label='L2 (Ridge)', color='blue')
axes[0].plot(loss_history_l1, linewidth=2, label='L1 (Lasso)', color='red')
axes[0].plot(loss_history_elastic, linewidth=2, label='Elastic Net', color='green')
axes[0].set_xlabel('Iteration', fontsize=12)
axes[0].set_ylabel('Total Loss', fontsize=12)
axes[0].set_title('Loss Convergence Comparison', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot 2: Weight distribution
axes[1].hist(W_l2[:-1].ravel(), bins=50, alpha=0.6, label='L2', color='blue')
axes[1].hist(W_l1[:-1].ravel(), bins=50, alpha=0.6, label='L1', color='red')
axes[1].hist(W_elastic[:-1].ravel(), bins=50, alpha=0.6, label='Elastic Net', color='green')
axes[1].set_xlabel('Weight Value', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Weight Distribution', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\reports\\regularization_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to: reports/regularization_comparison.png")
plt.show()

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("="*70)
print("""
- L2 (Ridge): Shrinks all weights smoothly, no sparsity
- L1 (Lasso): Drives many weights to zero, creates sparse model
- Elastic Net: Balances both, some sparsity + smooth weights
""")
