import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab')
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10

print("="*60)
print("SECOND-ORDER OPTIMIZATION - LINEAR SOFTMAX - CIFAR-10")
print("="*60)

# ============================================================
# SECTION 1: LOAD AND PREPARE DATA
# ============================================================

path = 'C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\data\\cifar-10-batches-py\\'
(X_train, y_train), (X_test, y_test) = load_cifar10(path)

rng = np.random.default_rng(42)
idx = rng.permutation(X_train.shape[0])
N = 10000  # CV size
cv_idx = idx[:N]
tr_idx = idx[N:]

X_cv = X_train[cv_idx]
y_cv = y_train[cv_idx]
X_tr = X_train[tr_idx]
y_tr = y_train[tr_idx]

print("\nDataset sizes:")
print(f"  Train: {X_tr.shape[0]} samples")
print(f"  CV: {X_cv.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# ============================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================

print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

preprocessing_params = np.load('models/preprocessing_params.npy', allow_pickle=True).item()
mean_img = np.array(preprocessing_params['mean_img'])
std_img = np.array(preprocessing_params['std_img'])

X_tr = X_tr - mean_img
X_tr = X_tr / std_img
X_cv = X_cv - mean_img
X_cv = X_cv / std_img
X_test = X_test - mean_img
X_test = X_test / std_img

# Add bias term
X_tr = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
X_cv = np.hstack([X_cv, np.ones((X_cv.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

print(f"\nAfter preprocessing:")
print(f"  Train shape: {X_tr.shape}")
print(f"  CV shape: {X_cv.shape}")
print(f"  Test shape: {X_test.shape}")

# ============================================================
# SECTION 3: LOAD METADATA
# ============================================================

import json
with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"\nL2 Ridge Baseline (for reference):")
print(f"  CV Accuracy: {metadata['l2_ridge']['cv_accuracy']:.4f}")
print(f"  Test Accuracy: {metadata['l2_ridge']['test_accuracy']:.4f}")

# ============================================================
# SECTION 4: SOFTMAX AND LOSS FUNCTIONS
# ============================================================

print("\n" + "="*60)
print("SOFTMAX AND LOSS FUNCTIONS")
print("="*60)

def softmax(scores):
    """Numerically stable softmax"""
    scores_shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return probs

def cross_entropy_loss(probs, y_true, W, lambda_reg):
    """Cross-entropy loss with L2 regularization"""
    N = len(y_true)
    true_class_probs = probs[np.arange(N), y_true]
    ce_loss = -np.mean(np.log(np.clip(true_class_probs, 1e-10, 1.0)))
    l2_penalty = lambda_reg * np.sum(W[:-1, :]**2)
    return ce_loss + l2_penalty

def softmax_gradient(X, probs, y_true, W, lambda_reg):
    """Gradient of cross-entropy + L2"""
    N = len(y_true)
    target = probs.copy()
    target[np.arange(N), y_true] -= 1
    dW = X.T @ target / N
    dW[:-1, :] += 2 * lambda_reg * W[:-1, :]
    return dW

print("\n[OK] Softmax and loss functions defined")

# ============================================================
# SECTION 5: HESSIAN COMPUTATION
# ============================================================

print("\n" + "="*60)
print("HESSIAN COMPUTATION")
print("="*60)

print("""
For softmax + cross-entropy with L2 regularization:

Hessian H = d²L/dW²

For a linear model, the Hessian can be approximated as:
  H ≈ (1/N) X.T @ D @ X + 2*lambda*I
  
where D is a diagonal matrix of softmax curvatures

For efficient computation on large problems:
  - Full Hessian: O(D² memory) - prohibitive
  - Hessian-vector product: O(D memory) - efficient
  - Diagonal Hessian: O(D memory) - fast approximation
""")

def compute_hessian_diagonal(X, probs, y_true, W, lambda_reg):
    """
    Compute diagonal of Hessian efficiently
    
    For softmax: H_ii ≈ sum_j(p_j * (1 - p_j) * x_i²)
    """
    N, D = X.shape
    C = probs.shape[1]  # num classes
    
    H_diag = np.zeros((D, C))
    
    # Compute curvature for each sample
    # For softmax: curvature = p_j * (1 - p_j) for off-diagonal
    # and similar for diagonal
    for i in range(N):
        x_i = X[i:i+1, :].T  # (D, 1)
        p_i = probs[i, :]    # (C,)
        
        # Hessian curvature: X.T @ diag(p_j * (1 - p_j)) @ X
        curvatures = p_i * (1 - p_i)  # (C,)
        
        # H_ii = sum_j(x_i² * curvature_j)
        H_diag += x_i * curvatures.reshape(1, -1)
    
    H_diag /= N
    
    # Add L2 regularization (only to pixel weights, not bias)
    H_diag[:-1, :] += 2 * lambda_reg
    
    return H_diag

print("\n[OK] Hessian diagonal computation implemented")

# ============================================================
# SECTION 6: SECOND-ORDER OPTIMIZERS
# ============================================================

print("\n" + "="*60)
print("SECOND-ORDER OPTIMIZATION METHODS")
print("="*60)

# ============================================================
# QUASI-NEWTON (L-BFGS)
# ============================================================

print("\n" + "-"*60)
print("METHOD 1: L-BFGS (LIMITED MEMORY BFGS)")
print("-"*60)

print("""
L-BFGS = Limited Memory BFGS

Key Ideas:
  1. Approximate Hessian with rank-1 updates
  2. Store only m recent gradient differences (usually m=20)
  3. Compute search direction without explicit Hessian
  4. Reduces O(D²) memory to O(m*D)

Update rule (implicit):
  W_new = W - alpha * H_approx^-1 * grad_L
  
where H_approx is updated using BFGS formula.

Advantages:
  - Second-order convergence
  - O(m*D) memory (vs O(D²) for Newton)
  - Quasi-Newton: doesn't require true Hessian
  
Disadvantages:
  - More complex to implement
  - Line search required (expensive)
  - Still O(D) per iteration
""")

def lbfgs(X_train, y_train, X_cv, y_cv, W_init, lambda_reg, max_iterations=50, m=20):
    """
    L-BFGS optimization
    
    Inputs:
        X_train, y_train: training data
        X_cv, y_cv: CV data
        W_init: initial weights
        lambda_reg: regularization strength
        max_iterations: max iterations
        m: memory size for BFGS (default 20)
    
    Outputs:
        W: trained weights
        train_losses: list of training losses
        cv_accuracies: list of CV accuracies
    """
    W = W_init.copy()
    D, C = W.shape
    
    train_losses = []
    cv_accuracies = []
    
    # Initialize for BFGS
    s_history = []  # Weight differences
    y_history = []  # Gradient differences
    
    # Compute initial gradient
    scores_train = X_train @ W
    probs_train = softmax(scores_train)
    grad = softmax_gradient(X_train, probs_train, y_train, W, lambda_reg)
    
    for iteration in range(max_iterations):
        # Compute loss
        train_loss = cross_entropy_loss(probs_train, y_train, W, lambda_reg)
        train_losses.append(train_loss)
        
        # Compute search direction using L-BFGS formula (two-loop recursion)
        q = grad.copy()
        
        # First loop (backward through history)
        alpha_history = []
        for i in range(len(s_history) - 1, -1, -1):
            s_i = s_history[i]
            y_i = y_history[i]
            rho_i = 1.0 / np.sum(y_i * s_i)
            alpha_i = rho_i * np.sum(s_i * q)
            alpha_history.insert(0, alpha_i)
            q = q - alpha_i * y_i
        
        # Use diagonal Hessian as initial approximation
        H_diag = compute_hessian_diagonal(X_train, probs_train, y_train, W, lambda_reg)
        H_diag_inv = 1.0 / (H_diag + 1e-8)
        r = H_diag_inv * q
        
        # Second loop (forward through history)
        for i in range(len(s_history)):
            s_i = s_history[i]
            y_i = y_history[i]
            rho_i = 1.0 / np.sum(y_i * s_i)
            beta_i = rho_i * np.sum(y_i * r)
            r = r + s_i * (alpha_history[i] - beta_i)
        
        # Search direction (Newton direction approximation)
        p = -r
        
        # Line search (simple backtracking)
        step_size = 1.0
        for ls_iter in range(20):
            W_new = W + step_size * p
            
            # Compute new loss
            scores_train_new = X_train @ W_new
            probs_train_new = softmax(scores_train_new)
            loss_new = cross_entropy_loss(probs_train_new, y_train, W_new, lambda_reg)
            
            # Armijo condition
            if loss_new <= train_loss - 0.0001 * step_size * np.sum(grad * p):
                break
            
            step_size *= 0.5
        
        # Update weights
        grad_old = grad.copy()
        W = W_new
        probs_train = probs_train_new
        
        # Update gradient
        grad = softmax_gradient(X_train, probs_train, y_train, W, lambda_reg)
        
        # Update BFGS history
        s_new = step_size * p
        y_new = grad - grad_old
        
        s_history.append(s_new)
        y_history.append(y_new)
        
        # Keep only m recent updates
        if len(s_history) > m:
            s_history.pop(0)
            y_history.pop(0)
        
        # Evaluate on CV set
        scores_cv = X_cv @ W
        probs_cv = softmax(scores_cv)
        predictions_cv = np.argmax(probs_cv, axis=1)
        cv_accuracy = np.mean(predictions_cv == y_cv)
        cv_accuracies.append(cv_accuracy)
        
        print(f"  Iter {iteration+1:3d}: Train Loss = {train_loss:.4f}, CV Accuracy = {cv_accuracy:.4f}, Step Size = {step_size:.4f}")
    
    return W, train_losses, cv_accuracies

print("\n[OK] L-BFGS optimizer implemented!")

# ============================================================
# NATURAL GRADIENT DESCENT
# ============================================================

print("\n" + "-"*60)
print("METHOD 2: NATURAL GRADIENT DESCENT")
print("-"*60)

print("""
Natural Gradient Descent (Fisher Information Matrix)

Key Ideas:
  1. Use Fisher Information Matrix (approximates Hessian)
  2. Update rule: W = W - lr * F^-1 * grad_L
  3. For softmax: F ≈ (1/N) X.T @ diag(softmax_curvatures) @ X
  4. Invariant to reparameterization

Why it's useful:
  - Better conditioned than standard gradient
  - More geometrically sound
  - Works well for probabilistic models
  
Diagonal approximation (simpler):
  - F_diag ≈ mean(X² * softmax_curvatures)
  - Still captures second-order info
  - O(D) memory and computation
""")

def natural_gradient_descent(X_train, y_train, X_cv, y_cv, W_init, learning_rate, lambda_reg, num_epochs, batch_size=256, max_iterations=None):
    """
    Natural Gradient Descent with diagonal Fisher Information Matrix
    
    Inputs:
        X_train, y_train: training data
        X_cv, y_cv: CV data
        W_init: initial weights
        learning_rate: step size
        lambda_reg: regularization
        num_epochs: number of epochs
        batch_size: mini-batch size
        max_iterations: max mini-batch updates
    
    Outputs:
        W: trained weights
        train_losses: training losses per epoch
        cv_accuracies: CV accuracies per epoch
    """
    W = W_init.copy()
    train_losses = []
    cv_accuracies = []
    
    num_batches = len(X_train) // batch_size
    total_iterations = 0
    
    for epoch in range(num_epochs):
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
        
        epoch_loss = 0
        batches_processed = 0
        
        for batch in range(num_batches):
            if max_iterations is not None and total_iterations >= max_iterations:
                break
            
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass
            scores_batch = X_batch @ W
            probs_batch = softmax(scores_batch)
            
            # Compute loss
            batch_loss = cross_entropy_loss(probs_batch, y_batch, W, lambda_reg)
            epoch_loss += batch_loss
            batches_processed += 1
            total_iterations += 1
            
            # Compute gradient
            dW = softmax_gradient(X_batch, probs_batch, y_batch, W, lambda_reg)
            
            # Compute diagonal Fisher Information Matrix
            # For softmax: F_d,c ≈ E[X_d² * p_c * (1 - p_c)]
            N = len(y_batch)
            
            # Vectorized computation
            # X_batch: (N, D), probs_batch: (N, C)
            X_sq = X_batch ** 2  # (N, D)
            p_times_1mp = probs_batch * (1.0 - probs_batch)  # (N, C)
            
            # F_diag[d, c] = sum_n(X_sq[n,d] * p_times_1mp[n,c]) / N
            F_diag = X_sq.T @ p_times_1mp / N  # (D, C)
            
            F_diag[:-1, :] += 2 * lambda_reg  # Add L2 regularization term
            
            # Clamp to avoid division by very small numbers
            F_diag_clipped = np.maximum(F_diag, 1e-6)
            
            # Natural gradient: element-wise division by Fisher diagonal
            natural_grad = dW / F_diag_clipped
            
            # Update weights
            W = W - learning_rate * natural_grad
        
        if batches_processed > 0:
            avg_epoch_loss = epoch_loss / batches_processed
            train_losses.append(avg_epoch_loss)
        
        # Evaluate on CV
        scores_cv = X_cv @ W
        probs_cv = softmax(scores_cv)
        predictions_cv = np.argmax(probs_cv, axis=1)
        cv_accuracy = np.mean(predictions_cv == y_cv)
        cv_accuracies.append(cv_accuracy)
        
        print(f"  Epoch {epoch+1:3d} (Iter {total_iterations:4d}): Train Loss = {avg_epoch_loss:.4f}, CV Accuracy = {cv_accuracy:.4f}")
        
        if max_iterations is not None and total_iterations >= max_iterations:
            break
    
    return W, train_losses, cv_accuracies

print("\n[OK] Natural Gradient Descent implemented!")

# ============================================================
# SECTION 7: TRAINING
# ============================================================

print("\n" + "="*60)
print("TRAINING COMPARISON - SECOND-ORDER METHODS")
print("="*60)

learning_rate = 0.001
lambda_reg = 0.1

print(f"\nTraining Configuration:")
print(f"  Learning rate: {learning_rate}")
print(f"  Lambda: {lambda_reg}")

# ============================================================
# L-BFGS
# ============================================================

print("\n" + "-"*60)
print("METHOD 1: L-BFGS (SKIPPED - too slow)")
print("-"*60)
print("\n[OK] L-BFGS skipped (Iter 89 showed it's too slow with line search)")

# Use dummy values for comparison
best_cv_lbfgs = 0.4080
best_iter_lbfgs = 89
train_losses_lbfgs = []
cv_accuracies_lbfgs = []

# ============================================================
# NATURAL GRADIENT
# ============================================================

print("\n" + "-"*60)
print("METHOD 2: NATURAL GRADIENT DESCENT (100 epochs)")
print("-"*60)

np.random.seed(42)
W_ng_init = np.random.randn(X_tr.shape[1], 10) * 0.01

W_ng, train_losses_ng, cv_accuracies_ng = natural_gradient_descent(
    X_tr, y_tr, X_cv, y_cv, W_ng_init,
    learning_rate=learning_rate,
    lambda_reg=lambda_reg,
    num_epochs=100,
    batch_size=256,
    max_iterations=None
)

best_cv_ng = np.max(cv_accuracies_ng)
best_iter_ng = np.argmax(cv_accuracies_ng)

print(f"\nNatural Gradient Results:")
print(f"  Best CV Accuracy: {best_cv_ng:.4f} (at iteration {best_iter_ng+1})")
print(f"  Final CV Accuracy: {cv_accuracies_ng[-1]:.4f}")

# ============================================================
# TEST SET EVALUATION
# ============================================================

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

# L-BFGS (skipped)
acc_test_lbfgs = 0.0

# Natural Gradient
scores_test_ng = X_test @ W_ng
probs_test_ng = softmax(scores_test_ng)
preds_test_ng = np.argmax(probs_test_ng, axis=1)
acc_test_ng = np.mean(preds_test_ng == y_test)

print(f"\nTest Set Results:")
print(f"  L-BFGS:            SKIPPED (too slow)")
print(f"  Natural Gradient:  {acc_test_ng:.4f}")
print(f"  L2 Ridge Baseline: {metadata['l2_ridge']['test_accuracy']:.4f}")

# ============================================================
# SUMMARY COMPARISON
# ============================================================

print("\n" + "="*60)
print("SUMMARY - SECOND-ORDER OPTIMIZATION")
print("="*60)

print("\nSecond-Order Methods:")
print(f"  L-BFGS:            SKIPPED (slow with line search)")
print(f"  Natural Gradient:  Best CV = {best_cv_ng:.4f}, Test = {acc_test_ng:.4f}, Iterations = {len(train_losses_ng)}")

print(f"\nFirst-Order Baseline (from Lecture 03):")
print(f"  SGD+Momentum:      Best CV = 0.40+, Test = 0.40+")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

print("""
Comparing Second-Order vs First-Order:

1. L-BFGS:
   - Uses implicit second-order information
   - Fewer iterations needed
   - Line search adds overhead
   - Better for well-conditioned problems

2. Natural Gradient:
   - Approximates Fisher Information
   - More stable than standard gradient
   - Works well with probabilistic models
   - Diagonal approximation keeps O(D) cost

3. Convergence Patterns:
   - Second-order: faster initial convergence
   - First-order: simpler, often more stable
   - Trade-off: complexity vs convergence speed
""")

# ============================================================
# VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: CV Accuracy
ax1 = axes[0]
ax1.plot(np.arange(1, len(cv_accuracies_ng) + 1), cv_accuracies_ng, 's-', 
         label='Natural Gradient (100 iter)', linewidth=2, markersize=6, color='blue')
ax1.axhline(y=metadata['l2_ridge']['cv_accuracy'], color='green', linestyle='--', 
            linewidth=2, label='L2 Ridge Baseline')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('CV Accuracy', fontsize=12)
ax1.set_title('Natural Gradient: CV Accuracy', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Training Loss
ax2 = axes[1]
ax2.plot(np.arange(1, len(train_losses_ng) + 1), train_losses_ng, 's-', 
         label='Natural Gradient (100 iter)', linewidth=2, markersize=6, color='blue')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Training Loss', fontsize=12)
ax2.set_title('Natural Gradient: Training Loss', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/second_order_optimization_comparison.png', dpi=150, bbox_inches='tight')
print("\n[OK] Saved visualization: results/second_order_optimization_comparison.png")
plt.close()

print("\n" + "="*60)
print("SECOND-ORDER OPTIMIZATION COMPLETE")
print("="*60)
