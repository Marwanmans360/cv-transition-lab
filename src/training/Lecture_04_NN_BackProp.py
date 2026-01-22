
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# ============================================================
# 0) DATA LOADING (your loader)
# ============================================================

# EDIT THESE PATHS FOR YOUR VM
sys.path.append(r"C:\Users\user\OneDrive - TechnoVal\Desktop\Scripts\ML\cv-transition-lab")
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10

path = r"C:\Users\user\OneDrive - TechnoVal\Desktop\Scripts\ML\cv-transition-lab\data\cifar-10-batches-py\\"
(X_train, y_train), (X_test, y_test) = load_cifar10(path)

# Flatten if needed (if loader returns N x 32 x 32 x 3)
if X_train.ndim == 4:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

print("Raw shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

# ============================================================
# 1) TRAIN / CV SPLIT
# ============================================================

rng = np.random.default_rng(42)
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
# 2) OPTIONAL: SUBSAMPLE (set to None to use full sets)
#    Since you're on a VM, you can likely set these to None.
# ============================================================

TRAIN_SUBSET = None  # e.g., 5000 or None
CV_SUBSET    = None  # e.g., 2000 or None

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
# 3) PREPROCESSING (mean subtraction + feature scaling)
# ============================================================

X_tr = X_tr.astype(np.float32)
X_cv = X_cv.astype(np.float32)
X_test = X_test.astype(np.float32)

mean_img = X_tr.mean(axis=0, keepdims=True)
X_tr -= mean_img
X_cv -= mean_img
X_test -= mean_img

std_img = X_tr.std(axis=0, keepdims=True) + 1e-8
X_tr /= std_img
X_cv /= std_img
X_test /= std_img

print("\nAfter preprocessing:")
print(f"Train min={X_tr.min():.3f}, max={X_tr.max():.3f}, mean={X_tr.mean():.3f}, std={X_tr.std():.3f}")

# ============================================================
# 4) LAYERS: Affine, Activations, Softmax Loss
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

# ---------- Activations ----------
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
    cache = out  # store tanh(x)
    return out, cache

def tanh_backward(dout, cache):
    t = cache
    dx = dout * (1.0 - t**2)
    return dx

def sigmoid_forward(x):
    # stable sigmoid
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    cache = out  # store sigmoid(x)
    return out, cache

def sigmoid_backward(dout, cache):
    s = cache
    dx = dout * s * (1.0 - s)
    return dx

def gelu_forward(x):
    # GELU tanh approximation
    # gelu(x)=0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
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
    "relu": (relu_forward, relu_backward),
    "leaky_relu": (leaky_relu_forward, leaky_relu_backward),
    "tanh": (tanh_forward, tanh_backward),
    "sigmoid": (sigmoid_forward, sigmoid_backward),
    "gelu": (gelu_forward, gelu_backward),
}

def softmax_loss(scores, y):
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
# 5) TWO-LAYER NET (Affine -> Act -> Affine -> Softmax)
# ============================================================

def init_weights(D, H, C, init="he", seed=42):
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
# 6) FULL-BATCH TRAINING (Final CV accuracy after all iterations)
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
    Full-batch GD: each iteration uses all training data.
    Returns history and FINAL (end-of-training) CV accuracy.
    """
    history = {
        "loss": [],
        "iters": [],
        "cv_acc": [],   # logged during training (optional)
    }

    for it in range(1, num_iters + 1):
        t0 = time.time()

        loss, grads = net.loss_and_grads(X_tr, y_tr)
        for k in net.params:
            net.params[k] -= lr * grads[k]

        history["loss"].append(loss)

        if (it % log_every == 0) or (it == 1) or (it == num_iters):
            cv_acc = np.mean(net.predict(X_cv) == y_cv)
            history["iters"].append(it)
            history["cv_acc"].append(cv_acc)

            dt = time.time() - t0
            print(f"[{net.activation:10s}] iter {it:04d}/{num_iters} | loss={loss:.4f} | cv_acc={cv_acc:.4f} | lr={lr:.5f} | step={dt:.3f}s")

        if it % decay_every == 0:
            lr *= lr_decay

    final_cv_acc = np.mean(net.predict(X_cv) == y_cv)
    return history, final_cv_acc

# ============================================================
# 7) RUN EXPERIMENTS FOR ALL ACTIVATIONS + COMPARE FINAL CV
# ============================================================

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
    plot=True
):
    D = X_tr.shape[1]
    C = 10

    results = {}  # act -> {final_cv, history, init}

    for act in activations:
        init = "he" if act in ["relu", "leaky_relu", "gelu"] else "xavier"
        print("\n" + "="*90)
        print(f"Training activation='{act}' | init='{init}' | H={H} | reg={reg} | lr={lr} | iters={num_iters}")
        print("="*90)

        net = TwoLayerNet(D=D, H=H, C=C, activation=act, init=init, reg=reg, seed=seed)

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

        results[act] = {
            "final_cv": final_cv,
            "history": history,
            "init": init
        }

        print(f"--> FINAL CV accuracy for {act}: {final_cv:.4f}")

    # Summary sorted by FINAL CV (your requested metric)
    print("\n" + "#"*90)
    print("FINAL CV ACCURACY COMPARISON (after all iterations)")
    print("#"*90)
    sorted_acts = sorted(results.keys(), key=lambda a: results[a]["final_cv"], reverse=True)
    for act in sorted_acts:
        print(f"{act:10s} | init={results[act]['init']:6s} | FINAL_CV={results[act]['final_cv']:.4f}")

    # Optional plot: logged CV curves
    if plot:
        plt.figure(figsize=(12, 6))
        for act in sorted_acts:
            h = results[act]["history"]
            plt.plot(h["iters"], h["cv_acc"], marker="o",
                     label=f"{act} (final={results[act]['final_cv']:.3f})")
        plt.title("CV Accuracy vs Iteration (Full-batch GD) — Activation Comparison")
        plt.xlabel("Iteration")
        plt.ylabel("CV Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results

# ============================================================
# 8) CONFIG + RUN
# ============================================================

# These are reasonable starting points on a VM (adjust as you like):
H = 200
reg = 1e-3
lr = 5e-2
num_iters = 300
lr_decay = 0.9
decay_every = 100
log_every = 50

activations_to_test = ["relu", "leaky_relu", "tanh", "sigmoid", "gelu"]

results = run_all_activations(
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
    seed=42,
    plot=True
)
