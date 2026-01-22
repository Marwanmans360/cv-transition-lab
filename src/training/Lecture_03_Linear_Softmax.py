import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab')
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10

print("="*60)
print("LINEAR CLASSIFIER WITH SOFTMAX - CIFAR-10")
print("="*60)

# ============================================================
# SECTION 1: LOAD DATA
# ============================================================

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

# Load preprocessing parameters from previous training
preprocessing_params = np.load('models/preprocessing_params.npy', allow_pickle=True).item()
mean_img = np.array(preprocessing_params['mean_img'])
std_img = np.array(preprocessing_params['std_img'])

print("\nUsing saved preprocessing parameters:")
print(f"  Mean image shape: {mean_img.shape}")
print(f"  Std image shape: {std_img.shape}")

# Apply same preprocessing
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
# SECTION 3: LOAD INITIAL WEIGHTS
# ============================================================

print("\n" + "="*60)
print("LOADING INITIAL WEIGHTS FROM L2 (RIDGE)")
print("="*60)

# Load the best weights from L2 regularization
W = np.load('models/W_l2_ridge.npy')

print(f"\nLoaded weights shape: {W.shape}")
print(f"  Features: {W.shape[0]}")
print(f"  Classes: {W.shape[1]}")

# Load metadata to see what hyperparameters were used
import json
with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"\nL2 Ridge hyperparameters used:")
print(f"  Learning Rate: {metadata['l2_ridge']['best_learning_rate']}")
print(f"  Lambda: {metadata['l2_ridge']['best_lambda']}")
print(f"  CV Accuracy: {metadata['l2_ridge']['cv_accuracy']:.4f}")
print(f"  Test Accuracy: {metadata['l2_ridge']['test_accuracy']:.4f}")

# ============================================================
# INITIALIZE WITH RANDOM WEIGHTS INSTEAD
# ============================================================

print("\n" + "="*60)
print("USING RANDOM WEIGHT INITIALIZATION")
print("="*60)

# Random initialization with small values using seed for reproducibility
np.random.seed(42)
W = np.random.randn(W.shape[0], W.shape[1]) * 0.01

print(f"\nInitialized random weights with shape: {W.shape}")
print(f"  Weight scale: 0.01 (small random values)")
print(f"  Random seed: 42 (for reproducibility)")
print(f"  Reason: L2 Ridge weights (MSE) not suited for softmax (cross-entropy)")

# ============================================================
# SECTION 4: SOFTMAX IMPLEMENTATION
# ============================================================

print("\n" + "="*60)
print("SOFTMAX CLASSIFIER TRAINING")
print("="*60)

num_classes = 10

print("\n" + "="*60)
print("STEP 1: BUILD THE SOFTMAX FUNCTION")
print("="*60)

print("""
Softmax converts raw scores into probabilities:
  softmax(s_i) = exp(s_i) / sum_j(exp(s_j))

For a batch of N samples with C classes:
  Input scores: shape (N, C)
  Output probabilities: shape (N, C)
  
""")

def softmax(scores):
    """
    Compute softmax probabilities from scores (numerically stable version)
    
    Input: 
        scores: shape (N, C) where N = batch size, C = number of classes
    Output: 
        probs: shape (N, C) with probabilities summing to 1 for each sample
    """
    # Subtract max for numerical stability (prevents overflow)
    scores_shifted = scores - scores.max(axis=1, keepdims=True)
    
    # Compute exp and normalize
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    return probs

print("\n✓ Softmax function implemented!")
print("\nFunction breakdown:")
print("  1. scores_shifted = scores - scores.max(axis=1, keepdims=True)")
print("     → Shifts each sample's scores so max = 0")
print("  2. exp_scores = np.exp(scores_shifted)")
print("     → Compute exponentials (now in safe range)")
print("  3. probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)")
print("     → Normalize so probabilities sum to 1")

print("\n" + "="*60)
print("STEP 2: CROSS-ENTROPY LOSS")
print("="*60)

def cross_entropy_loss(probs, y_true, W, lambda_reg):
    """
    Compute cross-entropy loss with L2 regularization
    
    Inputs:
        probs: shape (N, C) - softmax probabilities
        y_true: shape (N,) - true class labels (0-9)
        W: shape (D, C) - weights
        lambda_reg: float - regularization strength
    
    Output:
        loss: scalar
    """
    N = len(y_true)
    
    # Get probability of true class for each sample using fancy indexing
    # probs[i, y_true[i]] gives us the probability for the correct class
    true_class_probs = probs[np.arange(N), y_true]
    
    # Compute cross-entropy: -log(true_class_prob)
    # Average over all samples
    ce_loss = -np.mean(np.log(true_class_probs))
    
    # Add L2 regularization (exclude bias term)
    l2_penalty = lambda_reg * np.sum(W[:-1, :]**2)
    
    # Total loss
    total_loss = ce_loss + l2_penalty
    
    return total_loss

print("\n✓ Cross-entropy loss function implemented!")
print("\nFunction breakdown:")
print("  1. true_class_probs = probs[np.arange(N), y_true]")
print("     → Extract probability for correct class using fancy indexing")
print("  2. ce_loss = -np.mean(np.log(true_class_probs))")
print("     → Compute cross-entropy")
print("  3. l2_penalty = lambda_reg * np.sum(W[:-1, :]**2)")
print("     → Add L2 regularization (exclude bias: W[:-1, :])")
print("  4. total_loss = ce_loss + l2_penalty")
print("     → Total loss")

print("\n" + "="*60)
print("STEP 3: SOFTMAX GRADIENT")
print("="*60)

def softmax_gradient(X, probs, y_true, W, lambda_reg):
    """
    Compute gradient of cross-entropy loss w.r.t. weights
    
    For softmax + cross-entropy, the gradient simplifies to:
        dL/dW = X.T @ (probs - y_one_hot) / N + 2*lambda*W
    
    Inputs:
        X: shape (N, D) - input features (including bias column)
        probs: shape (N, C) - softmax probabilities
        y_true: shape (N,) - true class labels
        W: shape (D, C) - weights
        lambda_reg: float - regularization strength
    
    Output:
        dW: shape (D, C) - gradient
    """
    N = len(y_true)
    
    # Create target: probs - y_one_hot
    # Start with probs (all probabilities)
    # Subtract 1 from the true class column for each sample
    target = probs.copy()
    target[np.arange(N), y_true] -= 1  # This is (probs - y_one_hot)
    
    # Compute gradient
    dW = X.T @ target / N
    
    # Add L2 regularization gradient (only on pixel weights, NOT bias)
    dW[:-1, :] += 2 * lambda_reg * W[:-1, :]
    
    return dW

print("\n✓ Softmax gradient function implemented!")
print("\nFunction breakdown:")
print("  1. target = probs.copy()")
print("     → Start with softmax probabilities")
print("  2. target[np.arange(N), y_true] -= 1")
print("     → Subtract 1 from true class (now: probs - y_one_hot)")
print("  3. dW = X.T @ target / N")
print("     → Compute gradient")
print("  4. dW[:-1, :] += 2 * lambda_reg * W[:-1, :]")
print("     → Add L2 regularization gradient (exclude bias)")

print("\n" + "="*60)
print("STEP 4: GRADIENT DESCENT OPTIMIZER")
print("="*60)

def gradient_descent(X_train, y_train, X_cv, y_cv, W_init, learning_rate, lambda_reg, num_iterations):
    """
    Train softmax classifier using gradient descent
    
    Inputs:
        X_train, y_train: training data
        X_cv, y_cv: CV data for evaluation
        W_init: initial weights
        learning_rate: step size for updates
        lambda_reg: regularization strength
        num_iterations: number of training iterations
    
    Outputs:
        W: trained weights
        train_losses: list of training losses per iteration
        cv_accuracies: list of CV accuracies per iteration
    """
    W = W_init.copy()
    train_losses = []
    cv_accuracies = []
    
    for iteration in range(num_iterations):
        # Forward pass on training data
        scores_train = X_train @ W
        probs_train = softmax(scores_train)
        
        # Compute loss on training data
        train_loss = cross_entropy_loss(probs_train, y_train, W, lambda_reg)
        train_losses.append(train_loss)
        
        # Compute gradient
        dW = softmax_gradient(X_train, probs_train, y_train, W, lambda_reg)
        
        # Update weights using gradient descent
        W = W - learning_rate * dW
        
        # Evaluate on CV set every 10 iterations (to speed up training)
        if (iteration + 1) % 10 == 0 or iteration == 0 or iteration == num_iterations - 1:
            scores_cv = X_cv @ W
            probs_cv = softmax(scores_cv)
            predictions_cv = np.argmax(probs_cv, axis=1)
            cv_accuracy = np.mean(predictions_cv == y_cv)
            cv_accuracies.append(cv_accuracy)
            print(f"  Iter {iteration+1:3d}: Train Loss = {train_loss:.4f}, CV Accuracy = {cv_accuracy:.4f}")
        else:
            cv_accuracies.append(np.nan)  # Placeholder for non-evaluated iterations
    
    return W, train_losses, cv_accuracies

print("\n✓ Gradient Descent optimizer implemented!")
print("\nOptimizer breakdown:")
print("  1. Forward pass: scores = X @ W, probs = softmax(scores)")
print("  2. Compute loss: loss = cross_entropy_loss(...)")
print("  3. Compute gradient: dW = softmax_gradient(...)")
print("  4. Update weights: W = W - learning_rate * dW")
print("  5. Evaluate on CV set and record accuracy")
print("  6. Repeat for num_iterations")

print("\n" + "="*60)
print("STEP 5: STOCHASTIC GRADIENT DESCENT (SGD) OPTIMIZER")
print("="*60)

def stochastic_gradient_descent(X_train, y_train, X_cv, y_cv, W_init, learning_rate, lambda_reg, num_epochs, batch_size=256, max_iterations=None):
    """
    Train softmax classifier using SGD with mini-batches
    
    Inputs:
        X_train, y_train: training data
        X_cv, y_cv: CV data for evaluation
        W_init: initial weights
        learning_rate: step size for updates
        lambda_reg: regularization strength
        num_epochs: number of passes through training data
        batch_size: mini-batch size (default 256)
        max_iterations: maximum number of mini-batch updates (optional, for fair comparison)
    
    Outputs:
        W: trained weights
        train_losses: list of average training losses per epoch
        cv_accuracies: list of CV accuracies per epoch
    """
    W = W_init.copy()
    train_losses = []
    cv_accuracies = []
    
    num_batches = len(X_train) // batch_size
    total_iterations = 0
    
    for epoch in range(num_epochs):
        # Shuffle training data each epoch
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
        
        epoch_loss = 0
        batches_processed = 0
        
        # Mini-batch training
        for batch in range(num_batches):
            # Check if we've hit max iterations limit
            if max_iterations is not None and total_iterations >= max_iterations:
                break
                
            # Extract mini-batch
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass on mini-batch
            scores_batch = X_batch @ W
            probs_batch = softmax(scores_batch)
            
            # Compute loss on mini-batch
            batch_loss = cross_entropy_loss(probs_batch, y_batch, W, lambda_reg)
            epoch_loss += batch_loss
            batches_processed += 1
            total_iterations += 1
            
            # Compute gradient on mini-batch
            dW = softmax_gradient(X_batch, probs_batch, y_batch, W, lambda_reg)
            
            # Update weights
            W = W - learning_rate * dW
        
        # Average loss for epoch
        if batches_processed > 0:
            avg_epoch_loss = epoch_loss / batches_processed
            train_losses.append(avg_epoch_loss)
        
        # Evaluate on full CV set
        scores_cv = X_cv @ W
        probs_cv = softmax(scores_cv)
        predictions_cv = np.argmax(probs_cv, axis=1)
        cv_accuracy = np.mean(predictions_cv == y_cv)
        cv_accuracies.append(cv_accuracy)
        
        # Print progress
        print(f"  Epoch {epoch+1:3d} (Iteration {total_iterations:4d}): Train Loss = {avg_epoch_loss:.4f}, CV Accuracy = {cv_accuracy:.4f}")
        
        # Break if max iterations reached
        if max_iterations is not None and total_iterations >= max_iterations:
            break
    
    return W, train_losses, cv_accuracies

print("\n✓ SGD optimizer implemented!")
print("\nOptimizer breakdown:")
print("  1. Shuffle training data")
print("  2. Split into mini-batches of 256 samples")
print("  3. For each mini-batch:")
print("     - Forward pass: compute softmax")
print("     - Compute gradient on mini-batch only")
print("     - Update weights: W = W - learning_rate * dW")
print("  4. After each epoch, evaluate on full CV set")
print("  5. Repeat for num_epochs")
print("\nKey differences from GD:")
print("  - Faster: 256 samples per update vs 30k")
print("  - More updates: ~117 per epoch vs 1")
print("  - Noisier: but often converges better!")

print("\n" + "="*60)
print("STEP 5B: SGD WITH MOMENTUM OPTIMIZER")
print("="*60)

print("""
SGD with Momentum = SGD + Momentum

Key Ideas:
  1. Accumulate velocity: Keep exponential moving average of gradients
  2. Use velocity for updates: W = W - lr * velocity
  3. Momentum parameter: Typically 0.9 (how much to keep previous velocity)
  
Why it helps:
  - Accelerates convergence in consistent directions
  - Dampens oscillations in inconsistent directions (zig-zagging)
  - Often converges faster than vanilla SGD
  - Simpler than Adam (no adaptive learning rates)

Parameters:
  - momentum = 0.9: Exponential decay for velocity
  - learning_rate: Step size

Update rule:
  v = momentum * v + gradient
  W = W - learning_rate * v
""")

def sgd_momentum(X_train, y_train, X_cv, y_cv, W_init, learning_rate, lambda_reg, num_epochs, batch_size=256,
                 momentum=0.9, max_iterations=None):
    """
    Train softmax classifier using SGD with Momentum
    
    Inputs:
        X_train, y_train: training data
        X_cv, y_cv: CV data for evaluation
        W_init: initial weights
        learning_rate: step size for updates
        lambda_reg: regularization strength
        num_epochs: number of passes through training data
        batch_size: mini-batch size (default 256)
        momentum: momentum coefficient (default 0.9)
        max_iterations: maximum number of mini-batch updates (optional)
    
    Outputs:
        W: trained weights
        train_losses: list of average training losses per epoch
        cv_accuracies: list of CV accuracies per epoch
    """
    W = W_init.copy()
    
    # Initialize velocity (accumulator for momentum)
    v = np.zeros_like(W)
    
    train_losses = []
    cv_accuracies = []
    
    num_batches = len(X_train) // batch_size
    total_iterations = 0
    
    for epoch in range(num_epochs):
        # Shuffle training data each epoch
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
        
        epoch_loss = 0
        batches_processed = 0
        
        # Mini-batch training
        for batch in range(num_batches):
            # Check if we've hit max iterations limit
            if max_iterations is not None and total_iterations >= max_iterations:
                break
                
            # Extract mini-batch
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass on mini-batch
            scores_batch = X_batch @ W
            probs_batch = softmax(scores_batch)
            
            # Compute loss on mini-batch
            batch_loss = cross_entropy_loss(probs_batch, y_batch, W, lambda_reg)
            epoch_loss += batch_loss
            batches_processed += 1
            total_iterations += 1
            
            # Compute gradient on mini-batch
            dW = softmax_gradient(X_batch, probs_batch, y_batch, W, lambda_reg)
            
            # SGD with Momentum update
            # Accumulate velocity: v = momentum * v + gradient
            v = momentum * v + dW
            
            # Update weights using velocity
            W = W - learning_rate * v
        
        # Average loss for epoch
        if batches_processed > 0:
            avg_epoch_loss = epoch_loss / batches_processed
            train_losses.append(avg_epoch_loss)
        
        # Evaluate on full CV set
        scores_cv = X_cv @ W
        probs_cv = softmax(scores_cv)
        predictions_cv = np.argmax(probs_cv, axis=1)
        cv_accuracy = np.mean(predictions_cv == y_cv)
        cv_accuracies.append(cv_accuracy)
        
        # Print progress
        print(f"  Epoch {epoch+1:3d} (Iteration {total_iterations:4d}): Train Loss = {avg_epoch_loss:.4f}, CV Accuracy = {cv_accuracy:.4f}")
        
        # Break if max iterations reached
        if max_iterations is not None and total_iterations >= max_iterations:
            break
    
    return W, train_losses, cv_accuracies

print("\n✓ SGD with Momentum optimizer implemented!")
print("\nOptimizer breakdown:")
print("  1. Initialize: v = 0 (velocity/momentum accumulator)")
print("  2. For each mini-batch:")
print("     - Compute gradient: dW = softmax_gradient(...)")
print("     - Accumulate velocity: v = momentum*v + dW")
print("     - Update weights: W = W - lr * v")
print("  3. Repeat for num_epochs")
print("\nComparison with SGD:")
print("  - SGD: Each update independent of previous")
print("  - SGD+Momentum: Each update carries velocity from past")
print("  - Momentum helps accelerate in consistent directions")
print("  - Momentum dampens oscillations (less zig-zagging)")
print("\nComparison with Adam:")
print("  - SGD+Momentum: Same learning rate for all parameters")
print("  - Adam: Adaptive learning rate per parameter")
print("  - SGD+Momentum: Usually converges slower than Adam")
print("  - But SGD+Momentum: Simpler, fewer hyperparameters")

print("\n" + "="*60)
print("STEP 6: ADAM OPTIMIZER (ADAPTIVE MOMENT ESTIMATION)")
print("="*60)

print("""
Adam = Adaptive Moment Estimation

Key Ideas:
  1. Momentum: Keep exponential moving average of gradients (first moment, m)
  2. Adaptive Learning Rates: Keep exponential moving average of squared gradients (second moment, v)
  3. Bias correction: Correct for initial zeros in m and v
  4. Update rule: W = W - lr * m_corrected / (sqrt(v_corrected) + epsilon)

Parameters:
  - beta1 = 0.9: Exponential decay for first moment (momentum)
  - beta2 = 0.999: Exponential decay for second moment
  - epsilon = 1e-8: Small constant to prevent division by zero

Advantages:
  - Adaptive learning rate per parameter
  - Works well with mini-batch training
  - Converges faster than vanilla GD/SGD
  - Less sensitive to learning rate
""")

def adam(X_train, y_train, X_cv, y_cv, W_init, learning_rate, lambda_reg, num_epochs, batch_size=256, 
         beta1=0.9, beta2=0.999, epsilon=1e-8, max_iterations=None):
    """
    Train softmax classifier using Adam optimizer
    
    Inputs:
        X_train, y_train: training data
        X_cv, y_cv: CV data for evaluation
        W_init: initial weights
        learning_rate: step size (usually 0.001)
        lambda_reg: regularization strength
        num_epochs: number of passes through training data
        batch_size: mini-batch size (default 256)
        beta1: exponential decay for first moment (default 0.9)
        beta2: exponential decay for second moment (default 0.999)
        epsilon: small constant for numerical stability (default 1e-8)
        max_iterations: maximum number of mini-batch updates (optional)
    
    Outputs:
        W: trained weights
        train_losses: list of average training losses per epoch
        cv_accuracies: list of CV accuracies per epoch
    """
    W = W_init.copy()
    
    # Initialize first and second moment estimates (same shape as W)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    
    train_losses = []
    cv_accuracies = []
    
    num_batches = len(X_train) // batch_size
    total_iterations = 0
    
    for epoch in range(num_epochs):
        # Shuffle training data each epoch
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
        
        epoch_loss = 0
        batches_processed = 0
        
        # Mini-batch training
        for batch in range(num_batches):
            # Check if we've hit max iterations limit
            if max_iterations is not None and total_iterations >= max_iterations:
                break
                
            # Extract mini-batch
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass on mini-batch
            scores_batch = X_batch @ W
            probs_batch = softmax(scores_batch)
            
            # Compute loss on mini-batch
            batch_loss = cross_entropy_loss(probs_batch, y_batch, W, lambda_reg)
            epoch_loss += batch_loss
            batches_processed += 1
            total_iterations += 1
            
            # Compute gradient on mini-batch
            dW = softmax_gradient(X_batch, probs_batch, y_batch, W, lambda_reg)
            
            # Adam update
            # First moment estimate (exponential moving average of gradient)
            m = beta1 * m + (1 - beta1) * dW
            
            # Second moment estimate (exponential moving average of squared gradient)
            v = beta2 * v + (1 - beta2) * (dW ** 2)
            
            # Bias-corrected first moment estimate
            m_corrected = m / (1 - beta1 ** total_iterations)
            
            # Bias-corrected second moment estimate
            v_corrected = v / (1 - beta2 ** total_iterations)
            
            # Update weights
            W = W - learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
        
        # Average loss for epoch
        if batches_processed > 0:
            avg_epoch_loss = epoch_loss / batches_processed
            train_losses.append(avg_epoch_loss)
        
        # Evaluate on full CV set
        scores_cv = X_cv @ W
        probs_cv = softmax(scores_cv)
        predictions_cv = np.argmax(probs_cv, axis=1)
        cv_accuracy = np.mean(predictions_cv == y_cv)
        cv_accuracies.append(cv_accuracy)
        
        # Print progress
        print(f"  Epoch {epoch+1:3d} (Iteration {total_iterations:4d}): Train Loss = {avg_epoch_loss:.4f}, CV Accuracy = {cv_accuracy:.4f}")
        
        # Break if max iterations reached
        if max_iterations is not None and total_iterations >= max_iterations:
            break
    
    return W, train_losses, cv_accuracies

print("\n✓ Adam optimizer implemented!")
print("\nOptimizer breakdown:")
print("  1. Initialize: m = 0, v = 0 (first and second moment estimates)")
print("  2. For each mini-batch:")
print("     - Compute gradient: dW = softmax_gradient(...)")
print("     - Update first moment: m = beta1*m + (1-beta1)*dW")
print("     - Update second moment: v = beta2*v + (1-beta2)*dW²")
print("     - Bias correction: m_hat = m / (1-beta1^t), v_hat = v / (1-beta2^t)")
print("     - Update weights: W = W - lr * m_hat / (sqrt(v_hat) + eps)")
print("  3. Repeat for num_epochs")
print("\nKey differences from SGD:")
print("  - Keeps momentum: helps accelerate convergence")
print("  - Adaptive per-parameter: scales learning rate by gradient history")
print("  - Usually converges faster with less tuning")

print("\n" + "="*60)
print("STEP 7: ADAMW OPTIMIZER (ADAM WITH WEIGHT DECAY)")
print("="*60)

print("""
AdamW = Adam with Weight Decay

Key Difference from Adam + L2 Regularization:
  - Standard Adam + L2: Regularization applied to gradient (coupled)
  - AdamW: Weight decay applied directly to weights (decoupled)
  
Why it matters:
  - L2 regularization scales with adaptive learning rate (smaller for parameters with large histories)
  - Weight decay is constant regardless of learning rate history
  - Often works better in practice!

Formula difference:
  - Adam + L2: gradient_with_penalty = gradient + 2*lambda*W, then update
  - AdamW: update W first, then apply: W = W * (1 - weight_decay)

This prevents regularization from interfering with adaptive learning rates.
""")

def adamw(X_train, y_train, X_cv, y_cv, W_init, learning_rate, weight_decay, num_epochs, batch_size=256,
          beta1=0.9, beta2=0.999, epsilon=1e-8, max_iterations=None):
    """
    Train softmax classifier using AdamW optimizer
    
    Inputs:
        X_train, y_train: training data
        X_cv, y_cv: CV data for evaluation
        W_init: initial weights
        learning_rate: step size (usually 0.001)
        weight_decay: regularization strength (decoupled from gradient)
        num_epochs: number of passes through training data
        batch_size: mini-batch size (default 256)
        beta1: exponential decay for first moment (default 0.9)
        beta2: exponential decay for second moment (default 0.999)
        epsilon: small constant for numerical stability (default 1e-8)
        max_iterations: maximum number of mini-batch updates (optional)
    
    Outputs:
        W: trained weights
        train_losses: list of average training losses per epoch
        cv_accuracies: list of CV accuracies per epoch
    """
    W = W_init.copy()
    
    # Initialize first and second moment estimates
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    
    train_losses = []
    cv_accuracies = []
    
    num_batches = len(X_train) // batch_size
    total_iterations = 0
    
    for epoch in range(num_epochs):
        # Shuffle training data each epoch
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
        
        epoch_loss = 0
        batches_processed = 0
        
        # Mini-batch training
        for batch in range(num_batches):
            # Check if we've hit max iterations limit
            if max_iterations is not None and total_iterations >= max_iterations:
                break
                
            # Extract mini-batch
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass on mini-batch
            scores_batch = X_batch @ W
            probs_batch = softmax(scores_batch)
            
            # Compute loss on mini-batch (WITHOUT regularization in loss calculation)
            N = len(y_batch)
            true_class_probs = probs_batch[np.arange(N), y_batch]
            ce_loss = -np.mean(np.log(true_class_probs))
            epoch_loss += ce_loss
            batches_processed += 1
            total_iterations += 1
            
            # Compute gradient on mini-batch (WITHOUT regularization term)
            target = probs_batch.copy()
            target[np.arange(N), y_batch] -= 1
            dW = X_batch.T @ target / N
            # Note: NO regularization gradient added here
            
            # Adam update (same as before)
            m = beta1 * m + (1 - beta1) * dW
            v = beta2 * v + (1 - beta2) * (dW ** 2)
            
            # Bias-corrected estimates
            m_corrected = m / (1 - beta1 ** total_iterations)
            v_corrected = v / (1 - beta2 ** total_iterations)
            
            # Update weights with Adam
            W = W - learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
            
            # DECOUPLED weight decay (applied after gradient update)
            # Only apply to pixel weights, not bias
            W[:-1, :] = W[:-1, :] * (1 - weight_decay)
        
        # Average loss for epoch
        if batches_processed > 0:
            avg_epoch_loss = epoch_loss / batches_processed
            train_losses.append(avg_epoch_loss)
        
        # Evaluate on full CV set
        scores_cv = X_cv @ W
        probs_cv = softmax(scores_cv)
        predictions_cv = np.argmax(probs_cv, axis=1)
        cv_accuracy = np.mean(predictions_cv == y_cv)
        cv_accuracies.append(cv_accuracy)
        
        # Print progress
        print(f"  Epoch {epoch+1:3d} (Iteration {total_iterations:4d}): Train Loss = {avg_epoch_loss:.4f}, CV Accuracy = {cv_accuracy:.4f}")
        
        # Break if max iterations reached
        if max_iterations is not None and total_iterations >= max_iterations:
            break
    
    return W, train_losses, cv_accuracies

print("\n✓ AdamW optimizer implemented!")
print("\nKey difference from Adam:")
print("  - Adam + L2: Regularization term in gradient (coupled)")
print("  - AdamW: Weight decay applied separately after update (decoupled)")
print("  - AdamW usually: Better generalization, less sensitive to learning rate")
print("\nImplementation:")
print("  1. Same as Adam for moment estimates and bias correction")
print("  2. Same weight update: W = W - lr * m_hat / (sqrt(v_hat) + eps)")
print("  3. NEW: After update, apply weight decay: W[:-1,:] *= (1 - weight_decay)")
print("  4. Note: Bias (W[-1,:]) not regularized")

print("\n" + "="*60)
print("STEP 8: RMSPROP OPTIMIZER (ROOT MEAN SQUARE PROPAGATION)")
print("="*60)

print("""
RMSprop = Root Mean Square Propagation

Key Ideas:
  1. Adaptive Learning Rates: Keep exponential moving average of squared gradients
  2. Simpler than Adam: Only uses second moment (not first moment/momentum)
  3. Update rule: W = W - lr * gradient / (sqrt(v) + epsilon)

Parameters:
  - rho = 0.99: Exponential decay for second moment (different from beta2 in Adam)
  - epsilon = 1e-8: Small constant to prevent division by zero

Advantages:
  - Simpler than Adam (less hyperparameters to tune)
  - Still adaptive (good for mini-batch training)
  - Good middle ground between SGD and Adam

Disadvantages:
  - No momentum (slower convergence than Adam in some cases)
  - Adam usually better, but RMSprop still solid choice
""")

def rmsprop(X_train, y_train, X_cv, y_cv, W_init, learning_rate, lambda_reg, num_epochs, batch_size=256,
            rho=0.99, epsilon=1e-8, max_iterations=None):
    """
    Train softmax classifier using RMSprop optimizer
    
    Inputs:
        X_train, y_train: training data
        X_cv, y_cv: CV data for evaluation
        W_init: initial weights
        learning_rate: step size (usually 0.001)
        lambda_reg: regularization strength (L2)
        num_epochs: number of passes through training data
        batch_size: mini-batch size (default 256)
        rho: exponential decay for second moment (default 0.99)
        epsilon: small constant for numerical stability (default 1e-8)
        max_iterations: maximum number of mini-batch updates (optional)
    
    Outputs:
        W: trained weights
        train_losses: list of average training losses per epoch
        cv_accuracies: list of CV accuracies per epoch
    """
    W = W_init.copy()
    
    # Initialize second moment estimate (no first moment in RMSprop)
    v = np.zeros_like(W)
    
    train_losses = []
    cv_accuracies = []
    
    num_batches = len(X_train) // batch_size
    total_iterations = 0
    
    for epoch in range(num_epochs):
        # Shuffle training data each epoch
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
        
        epoch_loss = 0
        batches_processed = 0
        
        # Mini-batch training
        for batch in range(num_batches):
            # Check if we've hit max iterations limit
            if max_iterations is not None and total_iterations >= max_iterations:
                break
                
            # Extract mini-batch
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass on mini-batch
            scores_batch = X_batch @ W
            probs_batch = softmax(scores_batch)
            
            # Compute loss on mini-batch
            batch_loss = cross_entropy_loss(probs_batch, y_batch, W, lambda_reg)
            epoch_loss += batch_loss
            batches_processed += 1
            total_iterations += 1
            
            # Compute gradient on mini-batch
            dW = softmax_gradient(X_batch, probs_batch, y_batch, W, lambda_reg)
            
            # RMSprop: Update second moment estimate
            # v = rho * v + (1 - rho) * dW^2
            v = rho * v + (1 - rho) * (dW ** 2)
            
            # Update weights using adaptive learning rate
            # W = W - lr * gradient / (sqrt(v) + epsilon)
            W = W - learning_rate * dW / (np.sqrt(v) + epsilon)
        
        # Average loss for epoch
        if batches_processed > 0:
            avg_epoch_loss = epoch_loss / batches_processed
            train_losses.append(avg_epoch_loss)
        
        # Evaluate on full CV set
        scores_cv = X_cv @ W
        probs_cv = softmax(scores_cv)
        predictions_cv = np.argmax(probs_cv, axis=1)
        cv_accuracy = np.mean(predictions_cv == y_cv)
        cv_accuracies.append(cv_accuracy)
        
        # Print progress
        print(f"  Epoch {epoch+1:3d} (Iteration {total_iterations:4d}): Train Loss = {avg_epoch_loss:.4f}, CV Accuracy = {cv_accuracy:.4f}")
        
        # Break if max iterations reached
        if max_iterations is not None and total_iterations >= max_iterations:
            break
    
    return W, train_losses, cv_accuracies

print("\n✓ RMSprop optimizer implemented!")
print("\nOptimizer breakdown:")
print("  1. Initialize: v = 0 (second moment estimate)")
print("  2. For each mini-batch:")
print("     - Compute gradient: dW = softmax_gradient(...)")
print("     - Update second moment: v = rho*v + (1-rho)*dW²")
print("     - Update weights: W = W - lr * dW / (sqrt(v) + eps)")
print("  3. Repeat for num_epochs")
print("\nComparison with Adam:")
print("  - Simpler: No momentum, no bias correction")
print("  - Fewer hyperparameters to tune")
print("  - Often slower than Adam but still effective")

# Use optimal hyperparameters from regularization comparison
learning_rate = 0.001
lambda_reg = 0.1

print(f"\nTraining Configuration (from L2 Ridge comparison):")
print(f"  Learning rate: {learning_rate}")
print(f"  Lambda: {lambda_reg}")

print("\n" + "="*60)
print("TRAINING COMPARISON - ALL OPTIMIZERS")
print("="*60)

print(f"\nFair Comparison Setup (Equal Computational Work):")
print(f"  All methods: 500 weight updates")
print(f"  GD: 500 full-batch iterations")
print(f"  SGD/SGD+Mom/RMSprop/Adam/AdamW: 500 mini-batch iterations (~4.3 epochs)")
print(f"  Key: Same number of weight updates = same computational effort")

# ============================================================
# TRAIN WITH GRADIENT DESCENT
# ============================================================

print("\n" + "-"*60)
print("METHOD 1: FULL-BATCH GRADIENT DESCENT (500 iterations)")
print("-"*60)

np.random.seed(42)
W_gd_init = np.random.randn(W.shape[0], W.shape[1]) * 0.01

W_gd, train_losses_gd, cv_accuracies_gd = gradient_descent(
    X_tr, y_tr, X_cv, y_cv, W_gd_init, 
    learning_rate=learning_rate, 
    lambda_reg=lambda_reg, 
    num_iterations=500
)

best_cv_gd = np.nanmax(cv_accuracies_gd)
best_iter_gd = np.nanargmax(cv_accuracies_gd)

print(f"\nGradient Descent Results:")
print(f"  Best CV Accuracy: {best_cv_gd:.4f} (at iteration {best_iter_gd+1})")
print(f"  Final CV Accuracy: {cv_accuracies_gd[-1]:.4f}")

# ============================================================
# TRAIN WITH SGD
# ============================================================

print("\n" + "-"*60)
print("METHOD 2: STOCHASTIC GRADIENT DESCENT (500 iterations, batch_size=256)")
print("-"*60)

np.random.seed(42)
W_sgd_init = np.random.randn(W.shape[0], W.shape[1]) * 0.01

W_sgd, train_losses_sgd, cv_accuracies_sgd = stochastic_gradient_descent(
    X_tr, y_tr, X_cv, y_cv, W_sgd_init, 
    learning_rate=learning_rate, 
    lambda_reg=lambda_reg, 
    num_epochs=100,
    batch_size=256,
    max_iterations=500
)

best_cv_sgd = np.max(cv_accuracies_sgd)
best_epoch_sgd = np.argmax(cv_accuracies_sgd)

print(f"\nSGD Results:")
print(f"  Best CV Accuracy: {best_cv_sgd:.4f} (at epoch {best_epoch_sgd+1})")
print(f"  Final CV Accuracy: {cv_accuracies_sgd[-1]:.4f}")

# ============================================================
# TRAIN WITH SGD + MOMENTUM
# ============================================================

print("\n" + "-"*60)
print("METHOD 3: SGD WITH MOMENTUM (500 iterations, momentum=0.9)")
print("-"*60)

np.random.seed(42)
W_sgd_mom_init = np.random.randn(W.shape[0], W.shape[1]) * 0.01

W_sgd_mom, train_losses_sgd_mom, cv_accuracies_sgd_mom = sgd_momentum(
    X_tr, y_tr, X_cv, y_cv, W_sgd_mom_init, 
    learning_rate=learning_rate, 
    lambda_reg=lambda_reg, 
    num_epochs=100,
    batch_size=256,
    momentum=0.9,
    max_iterations=500
)

best_cv_sgd_mom = np.max(cv_accuracies_sgd_mom)
best_epoch_sgd_mom = np.argmax(cv_accuracies_sgd_mom)

print(f"\nSGD + Momentum Results:")
print(f"  Best CV Accuracy: {best_cv_sgd_mom:.4f} (at epoch {best_epoch_sgd_mom+1})")
print(f"  Final CV Accuracy: {cv_accuracies_sgd_mom[-1]:.4f}")

# ============================================================
# TRAIN WITH RMSPROP
# ============================================================

print("\n" + "-"*60)
print("METHOD 4: RMSPROP (500 iterations, rho=0.99)")
print("-"*60)

np.random.seed(42)
W_rmsprop_init = np.random.randn(W.shape[0], W.shape[1]) * 0.01

W_rmsprop, train_losses_rmsprop, cv_accuracies_rmsprop = rmsprop(
    X_tr, y_tr, X_cv, y_cv, W_rmsprop_init, 
    learning_rate=learning_rate, 
    lambda_reg=lambda_reg, 
    num_epochs=100,
    batch_size=256,
    rho=0.99,
    max_iterations=500
)

best_cv_rmsprop = np.max(cv_accuracies_rmsprop)
best_epoch_rmsprop = np.argmax(cv_accuracies_rmsprop)

print(f"\nRMSprop Results:")
print(f"  Best CV Accuracy: {best_cv_rmsprop:.4f} (at epoch {best_epoch_rmsprop+1})")
print(f"  Final CV Accuracy: {cv_accuracies_rmsprop[-1]:.4f}")

# ============================================================
# TRAIN WITH ADAM
# ============================================================

print("\n" + "-"*60)
print("METHOD 5: ADAM (500 iterations, beta1=0.9, beta2=0.999)")
print("-"*60)

np.random.seed(42)
W_adam_init = np.random.randn(W.shape[0], W.shape[1]) * 0.01

W_adam, train_losses_adam, cv_accuracies_adam = adam(
    X_tr, y_tr, X_cv, y_cv, W_adam_init, 
    learning_rate=learning_rate, 
    lambda_reg=lambda_reg, 
    num_epochs=100,
    batch_size=256,
    beta1=0.9,
    beta2=0.999,
    max_iterations=500
)

best_cv_adam = np.max(cv_accuracies_adam)
best_epoch_adam = np.argmax(cv_accuracies_adam)

print(f"\nAdam Results:")
print(f"  Best CV Accuracy: {best_cv_adam:.4f} (at epoch {best_epoch_adam+1})")
print(f"  Final CV Accuracy: {cv_accuracies_adam[-1]:.4f}")

# ============================================================
# TRAIN WITH ADAMW
# ============================================================

print("\n" + "-"*60)
print("METHOD 6: ADAMW (500 iterations, beta1=0.9, beta2=0.999, weight_decay=0.0001)")
print("-"*60)

np.random.seed(42)
W_adamw_init = np.random.randn(W.shape[0], W.shape[1]) * 0.01

W_adamw, train_losses_adamw, cv_accuracies_adamw = adamw(
    X_tr, y_tr, X_cv, y_cv, W_adamw_init, 
    learning_rate=learning_rate, 
    weight_decay=0.0001,  # Decoupled weight decay instead of L2
    num_epochs=100,
    batch_size=256,
    beta1=0.9,
    beta2=0.999,
    max_iterations=500
)

best_cv_adamw = np.max(cv_accuracies_adamw)
best_epoch_adamw = np.argmax(cv_accuracies_adamw)

print(f"\nAdamW Results:")
print(f"  Best CV Accuracy: {best_cv_adamw:.4f} (at epoch {best_epoch_adamw+1})")
print(f"  Final CV Accuracy: {cv_accuracies_adamw[-1]:.4f}")

# ============================================================
# COMPARISON SUMMARY
# ============================================================

print("\n" + "="*60)
print("FINAL OPTIMIZER COMPARISON")
print("="*60)

results = [
    ("Gradient Descent", best_cv_gd, cv_accuracies_gd[-1]),
    ("SGD", best_cv_sgd, cv_accuracies_sgd[-1]),
    ("SGD + Momentum", best_cv_sgd_mom, cv_accuracies_sgd_mom[-1]),
    ("RMSprop", best_cv_rmsprop, cv_accuracies_rmsprop[-1]),
    ("Adam", best_cv_adam, cv_accuracies_adam[-1]),
    ("AdamW", best_cv_adamw, cv_accuracies_adamw[-1])
]

# Sort by best CV accuracy
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

print("\nRanking by Best CV Accuracy:")
for i, (name, best, final) in enumerate(results_sorted, 1):
    print(f"  {i}. {name:20s}: Best = {best:.4f}, Final = {final:.4f}")

print(f"\nL2 Ridge Baseline (for reference):")
print(f"  CV Accuracy: {metadata['l2_ridge']['cv_accuracy']:.4f}")
print(f"  Test Accuracy: {metadata['l2_ridge']['test_accuracy']:.4f}")

print(f"\nWinner: {results_sorted[0][0]}")
print(f"  Best CV Accuracy: {results_sorted[0][1]:.4f}")
print(f"  Final CV Accuracy: {results_sorted[0][2]:.4f}")

# ============================================================
# TEST SET EVALUATION
# ============================================================

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

# Evaluate best model (SGD + Momentum) on test set
scores_test = X_test @ W_sgd_mom
probs_test = softmax(scores_test)
predictions_test = np.argmax(probs_test, axis=1)
test_accuracy = np.mean(predictions_test == y_test)

print(f"\nSGD + Momentum (Best Model) Test Results:")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Improvement over L2 Ridge baseline: {(test_accuracy - metadata['l2_ridge']['test_accuracy']):.4f}")
print(f"  Percentage improvement: {((test_accuracy - metadata['l2_ridge']['test_accuracy']) / metadata['l2_ridge']['test_accuracy'] * 100):.2f}%")

# Also evaluate all other models on test set for comparison
print("\n" + "-"*60)
print("All Models - Test Set Accuracy:")
print("-"*60)

test_results = []

# GD
scores_test_gd = X_test @ W_gd
probs_test_gd = softmax(scores_test_gd)
preds_test_gd = np.argmax(probs_test_gd, axis=1)
acc_test_gd = np.mean(preds_test_gd == y_test)
test_results.append(("Gradient Descent", acc_test_gd))
print(f"  Gradient Descent:  {acc_test_gd:.4f}")

# SGD
scores_test_sgd = X_test @ W_sgd
probs_test_sgd = softmax(scores_test_sgd)
preds_test_sgd = np.argmax(probs_test_sgd, axis=1)
acc_test_sgd = np.mean(preds_test_sgd == y_test)
test_results.append(("SGD", acc_test_sgd))
print(f"  SGD:               {acc_test_sgd:.4f}")

# SGD + Momentum
test_results.append(("SGD + Momentum", test_accuracy))
print(f"  SGD + Momentum:    {test_accuracy:.4f} ← BEST")

# RMSprop
scores_test_rmsprop = X_test @ W_rmsprop
probs_test_rmsprop = softmax(scores_test_rmsprop)
preds_test_rmsprop = np.argmax(probs_test_rmsprop, axis=1)
acc_test_rmsprop = np.mean(preds_test_rmsprop == y_test)
test_results.append(("RMSprop", acc_test_rmsprop))
print(f"  RMSprop:           {acc_test_rmsprop:.4f}")

# Adam
scores_test_adam = X_test @ W_adam
probs_test_adam = softmax(scores_test_adam)
preds_test_adam = np.argmax(probs_test_adam, axis=1)
acc_test_adam = np.mean(preds_test_adam == y_test)
test_results.append(("Adam", acc_test_adam))
print(f"  Adam:              {acc_test_adam:.4f}")

# AdamW
scores_test_adamw = X_test @ W_adamw
probs_test_adamw = softmax(scores_test_adamw)
preds_test_adamw = np.argmax(probs_test_adamw, axis=1)
acc_test_adamw = np.mean(preds_test_adamw == y_test)
test_results.append(("AdamW", acc_test_adamw))
print(f"  AdamW:             {acc_test_adamw:.4f}")

print(f"\n  L2 Ridge Baseline: {metadata['l2_ridge']['test_accuracy']:.4f}")

# ============================================================
# VISUALIZATION - TRAINING CURVES
# ============================================================

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: CV Accuracy Curves
ax1 = axes[0]
epochs_gd = np.arange(1, len(cv_accuracies_gd) + 1)
epochs_sgd = np.arange(1, len(cv_accuracies_sgd) + 1)
epochs_sgd_mom = np.arange(1, len(cv_accuracies_sgd_mom) + 1)
epochs_rmsprop = np.arange(1, len(cv_accuracies_rmsprop) + 1)
epochs_adam = np.arange(1, len(cv_accuracies_adam) + 1)
epochs_adamw = np.arange(1, len(cv_accuracies_adamw) + 1)

# Plot only evaluated iterations for GD (every 10 iterations)
gd_valid_idx = ~np.isnan(cv_accuracies_gd)
ax1.plot(epochs_gd[gd_valid_idx], np.array(cv_accuracies_gd)[gd_valid_idx], 'o-', label='Gradient Descent', linewidth=2, markersize=4)

ax1.plot(epochs_sgd, cv_accuracies_sgd, 's-', label='SGD', linewidth=2, markersize=4)
ax1.plot(epochs_sgd_mom, cv_accuracies_sgd_mom, '^-', label='SGD + Momentum', linewidth=2, markersize=4, color='green')
ax1.plot(epochs_rmsprop, cv_accuracies_rmsprop, 'd-', label='RMSprop', linewidth=2, markersize=4)
ax1.plot(epochs_adam, cv_accuracies_adam, 'v-', label='Adam', linewidth=2, markersize=4)
ax1.plot(epochs_adamw, cv_accuracies_adamw, 'p-', label='AdamW', linewidth=2, markersize=4, color='purple')

ax1.axhline(y=metadata['l2_ridge']['cv_accuracy'], color='red', linestyle='--', linewidth=2, label='L2 Ridge Baseline')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('CV Accuracy', fontsize=12)
ax1.set_title('Optimizer Comparison: CV Accuracy Over Training', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.25, 0.42])

# Plot 2: Training Loss Curves
ax2 = axes[1]

# Only plot GD training loss every 50 iterations to avoid clutter
gd_loss_thin = train_losses_gd[::50]
gd_iter_thin = np.arange(0, len(train_losses_gd), 50)
ax2.plot(gd_iter_thin, gd_loss_thin, 'o-', label='Gradient Descent', linewidth=2, markersize=6)

# For SGD methods, convert mini-batch iterations to epoch-like scale
# ~156 mini-batches per epoch
sgd_iters = np.arange(len(train_losses_sgd)) + 1
ax2.plot(sgd_iters * (500 / len(train_losses_sgd)), train_losses_sgd, 's-', label='SGD', linewidth=2, markersize=6)
ax2.plot(sgd_iters * (500 / len(train_losses_sgd_mom)), train_losses_sgd_mom, '^-', label='SGD + Momentum', linewidth=2, markersize=6, color='green')
ax2.plot(sgd_iters * (500 / len(train_losses_rmsprop)), train_losses_rmsprop, 'd-', label='RMSprop', linewidth=2, markersize=6)
ax2.plot(sgd_iters * (500 / len(train_losses_adam)), train_losses_adam, 'v-', label='Adam', linewidth=2, markersize=6)
ax2.plot(sgd_iters * (500 / len(train_losses_adamw)), train_losses_adamw, 'p-', label='AdamW', linewidth=2, markersize=6, color='purple')

ax2.set_xlabel('Iteration (scaled to 500 updates)', fontsize=12)
ax2.set_ylabel('Training Loss', fontsize=12)
ax2.set_title('Optimizer Comparison: Training Loss Over Iterations', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/optimizer_comparison_curves.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization: results/optimizer_comparison_curves.png")
plt.close()

# Create a summary comparison plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

optimizer_names = [name for name, _ in test_results]
cv_accs = [best_cv_gd, best_cv_sgd, best_cv_sgd_mom, best_cv_rmsprop, best_cv_adam, best_cv_adamw]
test_accs = [acc for _, acc in test_results]

x = np.arange(len(optimizer_names))
width = 0.35

bars1 = ax.bar(x - width/2, cv_accs, width, label='Best CV Accuracy', alpha=0.8)
bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)

# Add baseline line
ax.axhline(y=metadata['l2_ridge']['cv_accuracy'], color='red', linestyle='--', linewidth=2, 
           label=f"L2 Ridge Baseline (CV: {metadata['l2_ridge']['cv_accuracy']:.4f})")
ax.axhline(y=metadata['l2_ridge']['test_accuracy'], color='orange', linestyle='--', linewidth=2,
           label=f"L2 Ridge Baseline (Test: {metadata['l2_ridge']['test_accuracy']:.4f})")

ax.set_xlabel('Optimizer', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Final Optimizer Comparison: CV vs Test Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(optimizer_names, rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.3, 0.42])

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/optimizer_comparison_summary.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization: results/optimizer_comparison_summary.png")
plt.close()

print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)
print("\nGenerated plots:")
print("  1. optimizer_comparison_curves.png")
print("     - CV Accuracy curves for all optimizers")
print("     - Training Loss curves for all optimizers")
print("  2. optimizer_comparison_summary.png")
print("     - Side-by-side comparison of CV vs Test accuracy")
print("     - Baseline comparisons included")
