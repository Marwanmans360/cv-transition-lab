
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# ============================================================
# 0) PATHS (EDIT THESE FOR YOUR VM)
# ============================================================
# Your project root folder that contains src/...
PROJECT_ROOT = r"/path/to/cv-transition-lab"  # e.g. "/home/ubuntu/cv-transition-lab"
# CIFAR-10 python batches folder containing data_batch_1..5, test_batch
# CIFAR_PATH = r"/path/to/cifar-10-batches-py"  # e.g. "/home/ubuntu/data/cifar-10-batches-py"
CIFAR_PATH = "/content/drive/MyDrive/datasets/cifar-10-batches-py"
# Add project root to Python path so imports work
sys.path.append(PROJECT_ROOT)

# Your loader (as you already have)
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10

# ============================================================
# 1) LOAD DATA
# ============================================================
(X_train, y_train), (X_test, y_test) = load_cifar10(CIFAR_PATH)

# Flatten if needed (if loader returns N x 32 x 32 x 3)
if X_train.ndim == 4:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

print("Raw shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

# ============================================================
# 2) TRAIN / CV SPLIT
# ============================================================
SEED = 42
rng = np.random.default_rng(SEED)
idx = rng.permutation(X_train.shape[0])

N_cv = 10000
cv_idx = idx[:N_cv]
tr_idx = idx[N_cv:]

X_cv = X_train[cv_idx]
y_cv = y_train[cv_idx]
X_tr = X_train[tr_idx]
y_tr = y_train[tr_idx]

print("\nSplit shapes (before optional subsampling):")
print("Train:", X_tr.shape, y_tr.shape)
print("CV   :", X_cv.shape, y_cv.shape)

# ============================================================
# 3) OPTIONAL SUBSAMPLING (set to None for full sets)
# ============================================================
TRAIN_SUBSET = None   # e.g. 5000 or None
CV_SUBSET    = None   # e.g. 2000 or None

if TRAIN_SUBSET is not None:
    X_tr = X_tr[:TRAIN_SUBSET]
    y_tr = y_tr[:TRAIN_SUBSET]

if CV_SUBSET is not None:
    X_cv = X_cv[:CV_SUBSET]
    y_cv = y_cv[:CV_SUBSET]

print("\nAfter optional subsampling:")
print("Train:", X_tr.shape, y_tr.shape)
print("CV   :", X_cv.shape, y_cv.shape)

# ============================================================
# 4) PREPROCESSING (mean subtraction + feature scaling)
# ============================================================
X_tr   = X_tr.astype(np.float32)
X_cv   = X_cv.astype(np.float32)
X_test = X_test.astype(np.float32)

mean_img = X_tr.mean(axis=0, keepdims=True)
X_tr   -= mean_img
X_cv   -= mean_img
X_test -= mean_img

std_img = X_tr.std(axis=0, keepdims=True) + 1e-8
X_tr   /= std_img
X_cv   /= std_img
X_test /= std_img

print("\nAfter preprocessing:")
print(f"Train min={X_tr.min():.3f}, max={X_tr.max():.3f}, mean={X_tr.mean():.3f}, std={X_tr.std():.3f}")

# ============================================================
# 5) CORE LAYERS: AFFINE, ACTIVATIONS, SOFTMAX LOSS
# ============================================================

def affine_forward(X, W, b):
    out = X @ W + b
    cache = (X, W, b)
    return out, cache

def affine_backward(dout, cache):
    X, W, b = cache
    dX = dout @ W.T
    dW = X.T @ dout
    db = np.sum(dout, axis=0)
    return dX, dW, db

# ---- Activations ----
def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    x = cache
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx

def leaky_relu_forward(x, alpha=0.01):
    out = np.where(x > 0, x, alpha * x)
    cache = (x, alpha)
    return out, cache

def leaky_relu_backward(dout, cache):
    x, alpha = cache
    dx = dout * np.where(x > 0, 1.0, alpha)
    return dx

def tanh_forward(x):
    out = np.tanh(x)
    cache = out  # tanh(x)
    return out, cache

def tanh_backward(dout, cache):
    t = cache
    dx = dout * (1.0 - t**2)
    return dx

def sigmoid_forward(x):
    # numerically stable sigmoid
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    cache = out  # sigmoid(x)
    return out, cache

def sigmoid_backward(dout, cache):
    s = cache
    dx = dout * s * (1.0 - s)
    return dx

def gelu_forward(x):
    # GELU tanh approximation:
    # 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * x**3)
    t = np.tanh(u)
    out = 0.5 * x * (1.0 + t)
    cache = (x, u, t, c)
    return out, cache

def gelu_backward(dout, cache):
    x, u, t, c = cache
    du_dx = c * (1.0 + 3.0 * 0.044715 * x**2)
    dt_dx = (1.0 - t**2) * du_dx
    dx = dout * (0.5 * (1.0 + t) + 0.5 * x * dt_dx)
    return dx

ACTIVATIONS = {
    "relu":       (relu_forward, relu_backward),
    "leaky_relu": (leaky_relu_forward, leaky_relu_backward),
    "tanh":       (tanh_forward, tanh_backward),
    "sigmoid":    (sigmoid_forward, sigmoid_backward),
    "gelu":       (gelu_forward, gelu_backward),
}

def softmax_loss(scores, y):
    # stability
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    N = scores.shape[0]
    loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    dscores = probs
    dscores[np.arange(N), y] -= 1
    dscores /= N
    return loss, dscores

# ============================================================
# 6) TWO-LAYER NET (Affine -> Act -> Affine -> Softmax)
# ============================================================

def init_weights(D, H, C, init="he", seed=42):
    """
    init options:
      - "small": 1e-2 * N(0,1)
      - "xavier": N(0, 2/(fan_in+fan_out))
      - "he": N(0, 2/fan_in)
    """
    rng = np.random.default_rng(seed)

    if init == "small":
        W1 = 1e-2 * rng.standard_normal((D, H))
        W2 = 1e-2 * rng.standard_normal((H, C))
    elif init == "xavier":
        W1 = rng.standard_normal((D, H)) * np.sqrt(2.0 / (D + H))
        W2 = rng.standard_normal((H, C)) * np.sqrt(2.0 / (H + C))
    elif init == "he":
        W1 = rng.standard_normal((D, H)) * np.sqrt(2.0 / D)
        W2 = rng.standard_normal((H, C)) * np.sqrt(2.0 / H)
    else:
        raise ValueError("Unknown init. Use 'small', 'xavier', or 'he'.")

    b1 = np.zeros(H)
    b2 = np.zeros(C)
    return W1.astype(np.float32), b1.astype(np.float32), W2.astype(np.float32), b2.astype(np.float32)

class TwoLayerNet:
    def __init__(self, D, H, C, activation="relu", init="he", reg=1e-3, seed=42):
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(ACTIVATIONS.keys())}")

        self.act_fwd, self.act_bwd = ACTIVATIONS[activation]
        self.activation = activation
        self.reg = reg

        W1, b1, W2, b2 = init_weights(D, H, C, init=init, seed=seed)
        self.params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def forward(self, X):
        z1, fc1_cache = affine_forward(X, self.params["W1"], self.params["b1"])
        a1, act_cache = self.act_fwd(z1)
        scores, fc2_cache = affine_forward(a1, self.params["W2"], self.params["b2"])
        cache = (fc1_cache, act_cache, fc2_cache)
        return scores, cache

    def loss_and_grads(self, X, y):
        W1, W2 = self.params["W1"], self.params["W2"]

        scores, cache = self.forward(X)
        fc1_cache, act_cache, fc2_cache = cache

        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss

        # Backprop
        da1, dW2, db2 = affine_backward(dscores, fc2_cache)
        dW2 += self.reg * W2

        dz1 = self.act_bwd(da1, act_cache)

        dX, dW1, db1 = affine_backward(dz1, fc1_cache)
        dW1 += self.reg * W1

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return loss, grads

    def predict(self, X):
        scores, _ = self.forward(X)
        return np.argmax(scores, axis=1)

# ============================================================
# 7) FULL-BATCH TRAINING (Final CV after ALL iterations)
# ============================================================

def train_full_batch(
    net,
    X_tr, y_tr,
    X_cv, y_cv,
    lr=5e-2,
    lr_decay=0.9,
    num_iters=300,
    decay_every=100,
    log_every=50
):
    """
    Full-batch GD: each iteration uses ALL training data.
    Returns:
      - history: for plotting (loss, logged cv_acc points)
      - final_cv_acc: CV accuracy AFTER all iterations
    """
    history = {"loss": [], "iters": [], "cv_acc": []}

    for it in range(1, num_iters + 1):
        t0 = time.time()

        loss, grads = net.loss_and_grads(X_tr, y_tr)
        for k in net.params:
            net.params[k] -= lr * grads[k]

        history["loss"].append(loss)

        # Log occasionally (doesn't affect training)
        if it == 1 or it % log_every == 0 or it == num_iters:
            cv_acc = np.mean(net.predict(X_cv) == y_cv)
            history["iters"].append(it)
            history["cv_acc"].append(cv_acc)
            dt = time.time() - t0
            print(f"[{net.activation:10s}] iter {it:04d}/{num_iters} | loss={loss:.4f} | cv_acc={cv_acc:.4f} | lr={lr:.5f} | step={dt:.3f}s")

        # Step LR schedule (optional)
        if it % decay_every == 0:
            lr *= lr_decay

    final_cv_acc = np.mean(net.predict(X_cv) == y_cv)
    return history, final_cv_acc

# ============================================================
# 8) EXPERIMENT RUNNER: TWO MODES
# ============================================================

def make_shared_initial_params(D, H, C, init="small", seed=42):
    """
    Generate ONE shared set of initial parameters. Used for controlled comparisons.
    """
    W1, b1, W2, b2 = init_weights(D, H, C, init=init, seed=seed)
    return {"W1": W1.copy(), "b1": b1.copy(), "W2": W2.copy(), "b2": b2.copy()}

def run_all_activations(
    X_tr, y_tr,
    X_cv, y_cv,
    activations=("relu", "leaky_relu", "tanh", "sigmoid", "gelu"),
    H=200,
    reg=1e-3,
    lr=5e-2,
    num_iters=300,
    lr_decay=0.9,
    decay_every=100,
    log_every=50,
    seed=42,
    mode="activation_only",   # "activation_only" or "best_practice"
    shared_init="small",      # used only in activation_only
    plot=True
):
    """
    Compare FINAL CV accuracy AFTER all iterations for multiple activations.

    mode:
      - "activation_only": identical starting weights for all activations (controlled)
      - "best_practice": init matched to activation (He for ReLU-like, Xavier for tanh/sigmoid)
    """
    D = X_tr.shape[1]
    C = 10

    if mode not in ["activation_only", "best_practice"]:
        raise ValueError("mode must be 'activation_only' or 'best_practice'")

    shared_params = None
    if mode == "activation_only":
        shared_params = make_shared_initial_params(D, H, C, init=shared_init, seed=seed)

    results = {}  # act -> {final_cv, history, init_used}

    for act in activations:
        if mode == "best_practice":
            init_used = "he" if act in ["relu", "leaky_relu", "gelu"] else "xavier"
        else:
            init_used = shared_init

        print("\n" + "="*95)
        print(f"Mode={mode} | Activation='{act}' | init='{init_used}' | H={H} | reg={reg} | lr={lr} | iters={num_iters} | seed={seed}")
        print("="*95)

        net = TwoLayerNet(D=D, H=H, C=C, activation=act, init=init_used, reg=reg, seed=seed)

        # Overwrite with identical shared params (controlled)
        if mode == "activation_only":
            net.params = {k: v.copy() for k, v in shared_params.items()}

        history, final_cv = train_full_batch(
            net,
            X_tr, y_tr,
            X_cv, y_cv,
            lr=lr,
            lr_decay=lr_decay,
            num_iters=num_iters,
            decay_every=decay_every,
            log_every=log_every
        )

        results[act] = {"final_cv": final_cv, "history": history, "init_used": init_used}
        print(f"--> FINAL CV accuracy for {act}: {final_cv:.4f}")

    # Summary sorted by FINAL CV
    print("\n" + "#"*95)
    print(f"FINAL CV ACCURACY (after all iterations) | mode={mode} | seed={seed}")
    print("#"*95)
    sorted_acts = sorted(results.keys(), key=lambda a: results[a]["final_cv"], reverse=True)
    for act in sorted_acts:
        r = results[act]
        print(f"{act:10s} | init={r['init_used']:10s} | FINAL_CV={r['final_cv']:.4f}")

    # Plot logged CV curves
    if plot:
        plt.figure(figsize=(12, 6))
        for act in sorted_acts:
            h = results[act]["history"]
            plt.plot(h["iters"], h["cv_acc"], marker="o",
                     label=f"{act} (final={results[act]['final_cv']:.3f})")
        plt.title(f"CV Accuracy vs Iteration (Full-batch GD) — mode={mode}, seed={seed}")
        plt.xlabel("Iteration")
        plt.ylabel("CV Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results

# ============================================================
# 9) RUN BOTH MODES (YOU CAN TURN ONE OFF IF YOU WANT)
# ============================================================

activations_to_test = ["relu", "leaky_relu", "tanh", "sigmoid", "gelu"]

# Hyperparameters (keep same across activations for fair comparison)
H = 200
reg = 1e-3
lr = 5e-2
num_iters = 300
lr_decay = 0.9
decay_every = 100
log_every = 50

print("\n\n======================== MODE A: ACTIVATION-ONLY (CONTROLLED) ========================")
results_activation_only = run_all_activations(
    X_tr, y_tr,
    X_cv, y_cv,
    activations=activations_to_test,
    H=H,
    reg=reg,
    lr=lr,
    num_iters=num_iters,
    lr_decay=lr_decay,
    decay_every=decay_every,
    log_every=log_every,
    seed=SEED,
    mode="activation_only",
    shared_init="small",   # most neutral for controlled comparison
    plot=True
)

print("\n\n======================== MODE B: BEST-PRACTICE (HE/XAVIER) ==========================")
results_best_practice = run_all_activations(
    X_tr, y_tr,
    X_cv, y_cv,
    activations=activations_to_test,
    H=H,
    reg=reg,
    lr=lr,
    num_iters=num_iters,
    lr_decay=lr_decay,
    decay_every=decay_every,
    log_every=log_every,
    seed=SEED,
    mode="best_practice",
    plot=True
)
