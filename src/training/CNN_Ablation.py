import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# SEED + DEVICE
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CIFAR-10 LOADING
# ============================================================

def load_cifar10_batch(batch_file_path):
    with open(batch_file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch


def load_cifar10_from_folder(cifar10_folder):
    X_train_list = []
    y_train_list = []

    for i in range(1, 6):
        batch_path = os.path.join(cifar10_folder, f"data_batch_{i}")
        batch = load_cifar10_batch(batch_path)
        X_train_list.append(batch[b"data"])
        y_train_list.append(np.array(batch[b"labels"], dtype=np.int64))

    X_train_flat = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    test_path = os.path.join(cifar10_folder, "test_batch")
    test_batch = load_cifar10_batch(test_path)
    X_test_flat = test_batch[b"data"]
    y_test = np.array(test_batch[b"labels"], dtype=np.int64)

    return (X_train_flat, y_train), (X_test_flat, y_test)


def cifar10_flat_to_hwc_uint8(X_flat):
    X_flat = np.asarray(X_flat)
    assert X_flat.ndim == 2 and X_flat.shape[1] == 3072, f"Expected (N, 3072). Got {X_flat.shape}"
    X_hwc = X_flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X_hwc.astype(np.uint8)


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def show_image_hwc_uint8(img_hwc_uint8, title=""):
    img = np.asarray(img_hwc_uint8)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    plt.imshow(img, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.show()


def unnormalize_chw_to_hwc_uint8(img_chw_norm, mean, std):
    # img_norm = (img/255 - mean) / std
    # => img/255 = img_norm*std + mean
    mean_t = torch.tensor(mean, dtype=img_chw_norm.dtype, device=img_chw_norm.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=img_chw_norm.dtype, device=img_chw_norm.device).view(3, 1, 1)

    img_01 = img_chw_norm * std_t + mean_t
    img_255 = torch.round(img_01 * 255.0).clamp(0, 255).to(torch.uint8)
    return img_255.permute(1, 2, 0).cpu().numpy()


def show_conv1_filters(model, number_to_show=16):
    W = model.block1.conv1.weight.data.cpu()  # (32,3,3,3)
    n = min(number_to_show, W.shape[0])
    grid_size = int(np.ceil(np.sqrt(n)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(7, 7))
    axes = np.array(axes).reshape(-1)

    for i in range(grid_size * grid_size):
        axes[i].axis("off")
        if i >= n:
            continue

        k = W[i]  # (3,3,3)
        k = (k - k.min()) / (k.max() - k.min() + 1e-8)
        k_hwc = k.permute(1, 2, 0)
        axes[i].imshow(k_hwc)
        axes[i].set_title(f"F{i}", fontsize=8)

    plt.tight_layout()
    plt.show()


# ============================================================
# MODEL: MINI-VGG STYLE (your 3 blocks)
# ============================================================

class FirstBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        return x


class SecondBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        return x


class ThirdBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        return x


class MiniVGG(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.block1 = FirstBlock()
        self.block2 = SecondBlock()
        self.block3 = ThirdBlock()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ============================================================
# TRAINING UTILS
# ============================================================

def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

    return total_loss / total, total_correct / total


def overfit_one_batch(model, train_loader, device, steps=300, lr=0.01):  # Reduced learning rate
    print("\n=== Overfit 1 batch test (debugging) ===")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    Xb, yb = next(iter(train_loader))
    Xb, yb = Xb.to(device), yb.to(device)

    for t in range(1, steps + 1):
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)

        if torch.isnan(loss):
            print("NaN detected in loss!")
            break

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if t % 50 == 0:
            acc = accuracy_from_logits(logits, yb)
            print(f"[Overfit] step {t:03d} | loss {loss.item():.4f} | acc {acc:.3f}")

            # Debugging gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad max {param.grad.abs().max()}")

    print("If acc does NOT rise high (e.g., >0.9), something is wrong.\n")


# ============================================================
# WEIGHT INITIALIZATION
# ============================================================

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(42)
    device = get_device()
    print("Device:", device)

    cifar_folder = r"C:\Users\user\OneDrive - TechnoVal\Desktop\Scripts\ML\cv-transition-lab\data\cifar-10-batches-py"

    class_names = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

    # 1) Load CIFAR flat
    (X_train_flat, y_train), (X_test_flat, y_test) = load_cifar10_from_folder(cifar_folder)

    # 2) Convert to HWC uint8
    X_train_original = cifar10_flat_to_hwc_uint8(X_train_flat)  # (50000,32,32,3) uint8
    X_test_original  = cifar10_flat_to_hwc_uint8(X_test_flat)   # (10000,32,32,3) uint8

    # 3) Compute mean/std in [0..1]
    X_train_01 = X_train_original.astype(np.float32) / 255.0
    mean = X_train_01.mean(axis=(0, 1, 2))
    std  = X_train_01.std(axis=(0, 1, 2))
    print("mean:", mean, "std:", std)

    # 4) Normalize
    X_train_norm = (X_train_01 - mean) / std
    X_test_norm = ((X_test_original.astype(np.float32) / 255.0) - mean) / std

    # 5) Split train/val
    rng = np.random.default_rng(42)
    idx = rng.permutation(X_train_norm.shape[0])
    val_size = 10000
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    X_tr, y_tr = X_train_norm[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train_norm[val_idx], y_train[val_idx]

    # 6) Torch tensors (NCHW)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).permute(0, 3, 1, 2)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # 7) DataLoaders
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

    # 8) Quick sanity visualization (optional)
    # show one raw image
    idx_vis = 1680
    label = int(y_train[idx_vis])
    show_image_hwc_uint8(X_train_original[idx_vis], title=f"Raw image | {class_names[label]}")

    # show one normalized->unnormalized from loader
    Xb, yb = next(iter(train_loader))
    img_hwc_uint8 = unnormalize_chw_to_hwc_uint8(Xb[0], mean, std)
    show_image_hwc_uint8(img_hwc_uint8, title=f"From loader | label={class_names[int(yb[0].item())]}")

    # 9) Build model
    model = MiniVGG(dropout_p=0.0).to(device)
    initialize_weights(model)

    # sanity forward
    dummy = torch.randn(8, 3, 32, 32).to(device)
    out = model(dummy)
    print("Dummy forward output shape:", tuple(out.shape))  # (8,10)

    # show random conv1 filters (pre-training)
    show_conv1_filters(model, number_to_show=16)

    # 10) Overfit-one-batch test (debug)
    overfit_one_batch(model, train_loader, device, steps=300, lr=0.01)

    # 11) Baseline training (Run 0)
    print("=== Training baseline (Run 0) ===")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)

    best_val = 0.0
    best_state = None

    epochs = 35
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer=optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step()

        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d} | train acc {tr_acc:.3f} loss {tr_loss:.3f} "
              f"| val acc {va_acc:.3f} loss {va_loss:.3f}")

    print(f"\nBest validation accuracy: {best_val:.3f}")

    # 12) Test evaluation using best model
    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc = run_epoch(model, test_loader, device, optimizer=None)
    print(f"Test accuracy: {te_acc:.3f} | test loss: {te_loss:.3f}")


if __name__ == "__main__":
    main()