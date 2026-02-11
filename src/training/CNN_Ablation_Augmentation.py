import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


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


def visualize_augmentations(img_hwc_uint8, augmentation_types, mean, std):
    """
    Show how different augmentation levels transform a single image.
    
    Args:
        img_hwc_uint8: single image (H, W, C) in uint8 format
        augmentation_types: list of augmentation type names
        mean: normalization mean
        std: normalization std
    """
    # Convert to PIL for transforms
    from PIL import Image
    img_pil = Image.fromarray(img_hwc_uint8)
    
    num_types = len(augmentation_types)
    fig, axes = plt.subplots(1, num_types, figsize=(4*num_types, 4))
    if num_types == 1:
        axes = [axes]
    
    for idx, aug_type in enumerate(augmentation_types):
        transform = get_augmentation_transform(aug_type, mean, std)
        
        # Apply transform
        img_transformed = transform(img_pil)
        
        # Denormalize for visualization
        if aug_type == 'none':
            # For 'none' case, we still need to denormalize
            img_denorm = unnormalize_chw_to_hwc_uint8(img_transformed, mean, std)
        else:
            img_denorm = unnormalize_chw_to_hwc_uint8(img_transformed, mean, std)
        
        axes[idx].imshow(img_denorm)
        axes[idx].set_title(f"{aug_type.upper()}", fontsize=12, fontweight='bold')
        axes[idx].axis("off")
    
    plt.tight_layout()
    filename = "augmentation_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.show()


def visualize_augmentation_gridstyles(img_hwc_uint8, mean, std):
    """
    Create a grid visualization showing each augmentation applied multiple times
    to demonstrate the stochastic nature of each augmentation level.
    
    This helps understand:
    - How much variation each augmentation introduces
    - Whether augmentations preserve semantic content
    - The intensity/severity of each augmentation level
    """
    from PIL import Image
    img_pil = Image.fromarray(img_hwc_uint8)
    
    augmentation_types = ['none', 'crop_flip', 'color_jitter', 'cutout', 'advanced']
    num_samples = 4  # Show 4 different random applications
    
    fig, axes = plt.subplots(len(augmentation_types), num_samples, figsize=(12, 15))
    
    for aug_idx, aug_type in enumerate(augmentation_types):
        transform = get_augmentation_transform(aug_type, mean, std)
        
        for sample_idx in range(num_samples):
            # Apply transform multiple times to show stochasticity
            img_transformed = transform(img_pil)
            img_denorm = unnormalize_chw_to_hwc_uint8(img_transformed, mean, std)
            
            ax = axes[aug_idx, sample_idx]
            ax.imshow(img_denorm)
            
            if sample_idx == 0:
                ax.set_ylabel(aug_type.upper(), fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.suptitle("Data Augmentation Levels (each row shows different random applications)", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    filename = "augmentation_grid.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.show()


def save_augmented_batch_samples(data_loader, mean, std, augmentation_type, num_images=8):
    """
    Save a grid of augmented images from the training set to visualize
    what the model actually sees during training.
    
    Args:
        data_loader: DataLoader with augmentation applied
        mean: normalization mean for denormalization
        std: normalization std for denormalization
        augmentation_type: name of augmentation for filename
        num_images: number of images to show
    """
    # Get one batch
    batch_X, batch_y = next(iter(data_loader))
    
    # Visualize first num_images
    n = min(num_images, batch_X.size(0))
    grid_size = int(np.ceil(np.sqrt(n)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten() if grid_size > 1 else [axes]
    
    for i in range(grid_size * grid_size):
        axes[i].axis('off')
        if i < n:
            img_denorm = unnormalize_chw_to_hwc_uint8(batch_X[i], mean, std)
            axes[i].imshow(img_denorm)
            axes[i].set_title(f"Label: {batch_y[i].item()}", fontsize=8)
    
    fig.suptitle(f"Augmented Training Samples: {augmentation_type.upper()}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"augmented_batch_{augmentation_type}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Saved batch visualization: {filename}")
    plt.show()


# ============================================================
# AUGMENTATION TRANSFORMATIONS
# ============================================================

class RandomErasing(nn.Module):
    """Random Erasing / Cutout - randomly mask a rectangle in the image."""
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
        
        C, H, W = img.shape
        area = H * W
        
        log_ratio = torch.log(torch.tensor(self.ratio))
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1]))
        
        w = int(torch.sqrt(torch.tensor(area / aspect_ratio)).item())
        h = int(torch.sqrt(torch.tensor(area * aspect_ratio)).item())
        
        w = min(w, W)
        h = min(h, H)
        
        x = torch.randint(0, W - w + 1, (1,)).item() if w < W else 0
        y = torch.randint(0, H - h + 1, (1,)).item() if h < H else 0
        
        img[:, y:y+h, x:x+w] = self.value
        return img


def get_augmentation_transform(augmentation_type, mean, std):
    """
    Returns a torchvision transform pipeline based on augmentation type.
    Progressive augmentation levels:
    1. 'none' - baseline (no augmentation)
    2. 'crop_flip' - pad + random crop + horizontal flip
    3. 'color_jitter' - crop_flip + color jitter
    4. 'cutout' - crop_flip + color_jitter + random erasing
    5. 'advanced' - all of above (future: MixUp/CutMix)
    
    Args:
        augmentation_type: string identifying the augmentation strategy
        mean: normalization mean
        std: normalization std
    """
    # Base normalization transform (always applied)
    normalize = transforms.Normalize(mean=mean, std=std)
    
    if augmentation_type == 'none':
        # Level 0: No augmentation, only normalization
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation_type == 'crop_flip':
        # Level 1: Classic CIFAR recipe - pad 32→40, random crop 32×32, horizontal flip
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),  # Pads 32→40, then crops back to 32
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation_type == 'color_jitter':
        # Level 2: Add color jitter (brightness, contrast, saturation, hue)
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation_type == 'cutout':
        # Level 3: Add Random Erasing / Cutout (occlusion-style)
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            RandomErasing(p=0.5, scale=(0.02, 0.33)),
            normalize
        ])
    
    elif augmentation_type == 'advanced':
        # Level 4: All augmentations (future: add MixUp/CutMix)
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            RandomErasing(p=0.5, scale=(0.02, 0.33)),
            normalize
        ])
    
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


# ============================================================
# MODEL: MINI-VGG STYLE
# ============================================================

class FirstBlock(nn.Module):
    def __init__(self, norm_type='batch', activation_type='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.norm1 = self._get_norm(norm_type, 32)
        self.activation1 = self._get_activation(activation_type)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.norm2 = self._get_norm(norm_type, 32)
        self.activation2 = self._get_activation(activation_type)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _get_norm(self, norm_type, channels):
        if norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'group':
            return nn.GroupNorm(16, channels)
        else:
            return nn.Identity()

    def _get_activation(self, activation_type):
        if activation_type == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.pool(x)
        return x


class SecondBlock(nn.Module):
    def __init__(self, norm_type='batch', activation_type='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm1 = self._get_norm(norm_type, 64)
        self.activation1 = self._get_activation(activation_type)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm2 = self._get_norm(norm_type, 64)
        self.activation2 = self._get_activation(activation_type)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _get_norm(self, norm_type, channels):
        if norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'group':
            return nn.GroupNorm(16, channels)
        else:
            return nn.Identity()

    def _get_activation(self, activation_type):
        if activation_type == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.pool(x)
        return x


class ThirdBlock(nn.Module):
    def __init__(self, norm_type='batch', activation_type='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm1 = self._get_norm(norm_type, 128)
        self.activation1 = self._get_activation(activation_type)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm2 = self._get_norm(norm_type, 128)
        self.activation2 = self._get_activation(activation_type)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _get_norm(self, norm_type, channels):
        if norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'group':
            return nn.GroupNorm(16, channels)
        else:
            return nn.Identity()

    def _get_activation(self, activation_type):
        if activation_type == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.pool(x)
        return x


class MiniVGG(nn.Module):
    def __init__(self, dropout_p=0.5, norm_type='batch', activation_type='relu'):
        super().__init__()
        self.block1 = FirstBlock(norm_type=norm_type, activation_type=activation_type)
        self.block2 = SecondBlock(norm_type=norm_type, activation_type=activation_type)
        self.block3 = ThirdBlock(norm_type=norm_type, activation_type=activation_type)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            self._get_activation(activation_type),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(256, 10),
        )

    def _get_activation(self, activation_type):
        if activation_type == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()

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
    """
    Run one epoch of training or evaluation.
    
    Args:
        model: PyTorch model to train/evaluate
        loader: DataLoader providing batches
        device: cuda or cpu
        optimizer: if provided, runs training; otherwise evaluation
    
    Returns:
        avg_loss: average loss over epoch
        avg_acc: average accuracy over epoch
    """
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

            # Forward pass
            logits = model(X)
            loss = criterion(logits, y)

            if train:
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * X.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

    return total_loss / total, total_correct / total


def overfit_one_batch(model, train_loader, device, steps=300, lr=0.01):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if t % 50 == 0:
            acc = accuracy_from_logits(logits, yb)
            print(f"[Overfit] step {t:03d} | loss {loss.item():.4f} | acc {acc:.3f}")

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

    # Toggle between local and Colab paths
    use_colab = True  # Set to True for Google Colab
    
    if use_colab:
        cifar_folder = "/content/drive/MyDrive/datasets/cifar-10-batches-py"
    else:
        cifar_folder = r"C:\Users\user\OneDrive - TechnoVal\Desktop\Scripts\ML\cv-transition-lab\data\cifar-10-batches-py"
    
    print(f"Using CIFAR path: {cifar_folder}")
    print(f"Environment: {'Google Colab' if use_colab else 'Local'}\n")

    class_names = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

    # 1) Load CIFAR flat
    (X_train_flat, y_train), (X_test_flat, y_test) = load_cifar10_from_folder(cifar_folder)

    # 2) Convert to HWC uint8
    X_train_original = cifar10_flat_to_hwc_uint8(X_train_flat)
    X_test_original  = cifar10_flat_to_hwc_uint8(X_test_flat)

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

    # 7) DataLoaders with augmentation - will be applied per augmentation type
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

    # 8) Quick sanity visualization (optional)
    idx_vis = 1680
    label = int(y_train[idx_vis])
    show_image_hwc_uint8(X_train_original[idx_vis], title=f"Raw image | {class_names[label]}")

    Xb, yb = next(iter(val_loader))
    img_hwc_uint8 = unnormalize_chw_to_hwc_uint8(Xb[0], mean, std)
    show_image_hwc_uint8(img_hwc_uint8, title=f"From loader | label={class_names[int(yb[0].item())]}")

    # Fixed hyperparameters for augmentation study
    augmentation_types = ['none', 'crop_flip', 'color_jitter', 'cutout', 'advanced']
    norm_type = 'batch'      # Fixed to BatchNorm
    activation_type = 'relu' # Fixed to ReLU
    dropout_p = 0.5          # Fixed to optimal dropout rate
    results = {}

    print(f"\n{'='*70}")
    print("         DATA AUGMENTATION ABLATION STUDY")
    print(f"{'='*70}")
    print(f"\n📋 Fixed Hyperparameters:")
    print(f"   • Normalization: BatchNorm")
    print(f"   • Activation: ReLU")
    print(f"   • Dropout: 0.5")
    print(f"   • Optimizer: SGD (lr=0.01, momentum=0.9, weight_decay=1e-4)")
    print(f"   • Scheduler: MultiStepLR (milestones=[20, 30], gamma=0.1)")
    print(f"   • Epochs: 35")
    print(f"\n🎯 Progressive Augmentation Levels:")
    print(f"   Level 1: 'none' - Baseline (no augmentation)")
    print(f"            → Just normalization")
    print(f"   Level 2: 'crop_flip' - Classic CIFAR recipe")
    print(f"            → Pad 32→40, random crop to 32×32")
    print(f"            → Random horizontal flip (p=0.5)")
    print(f"   Level 3: 'color_jitter' - Add color variations")
    print(f"            → crop_flip augmentations")
    print(f"            → Brightness, contrast, saturation, hue jitter")
    print(f"   Level 4: 'cutout' - Add occlusion robustness")
    print(f"            → color_jitter augmentations")
    print(f"            → Random erasing (cutout) with p=0.5")
    print(f"   Level 5: 'advanced' - Maximum augmentation")
    print(f"            → All of the above (future: MixUp/CutMix)")
    print(f"\n{'='*70}\n")
    
    # Visualize augmentations on a sample image BEFORE training
    print("=== STEP 1: AUGMENTATION VISUALIZATION ===")
    print("📊 Visualizing how each augmentation level transforms a sample image...")
    sample_img = X_train_original[1680]  # Same image as before for consistency
    sample_label = class_names[int(y_train[1680])]
    print(f"   Sample image: {sample_label.upper()} (index 1680)\n")
    
    visualize_augmentations(sample_img, augmentation_types, mean, std)
    visualize_augmentation_gridstyles(sample_img, mean, std)
    print("✅ Augmentation visualizations complete!\n")

    print("\n=== STEP 2: TRAINING WITH DIFFERENT AUGMENTATION LEVELS ===\n")
    
    # Track training time for each augmentation
    import time
    
    for run_id, augmentation_type in enumerate(augmentation_types):
        print(f"\n{'='*70}")
        print(f"   RUN {run_id + 1}/{len(augmentation_types)}: {augmentation_type.upper()} Augmentation")
        print(f"{'='*70}")
        print(f"📦 Config: BatchNorm + ReLU + Dropout={dropout_p}")
        print(f"🎨 Augmentation: {augmentation_type}")
        
        run_start_time = time.time()

        # Create augmentation transform
        train_transform = get_augmentation_transform(augmentation_type, mean, std)
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Create augmented datasets
        from torchvision.datasets import VisionDataset

        class AugmentedTensorDataset(VisionDataset):
            def __init__(self, X, y, transform=None):
                self.X = X
                self.y = y
                self.transform = transform

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                img = self.X[idx]
                label = self.y[idx]
                
                if self.transform:
                    img = transforms.ToPILImage()(img.numpy().astype(np.uint8))
                    img = self.transform(img)
                
                return img, label

        # Create augmented training dataset
        print(f"\n📊 Creating dataset with {augmentation_type} augmentation...")
        train_dataset = AugmentedTensorDataset(X_train_original[tr_idx], y_tr, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")
        
        # Save augmented batch samples for this augmentation type
        print(f"\n📸 Saving augmented batch samples...")
        save_augmented_batch_samples(train_loader, mean, std, augmentation_type, num_images=8)

        # Build model
        print(f"\n🏗️  Building MiniVGG model...")
        model = MiniVGG(dropout_p=dropout_p, norm_type=norm_type, activation_type=activation_type).to(device)
        initialize_weights(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        # Sanity check forward pass
        dummy = torch.randn(8, 3, 32, 32).to(device)
        out = model(dummy)
        print(f"   Output shape: {tuple(out.shape)} ✓")

        # Optional: Show random conv1 filters (commented out to reduce clutter)
        # show_conv1_filters(model, number_to_show=16)

        # Overfit-one-batch test (sanity check)
        print(f"\n🔍 Running overfit-one-batch test (sanity check)...")
        overfit_one_batch(model, train_loader, device, steps=300, lr=0.01)

        # Main training loop
        print(f"\n{'='*70}")
        print(f"   TRAINING: {augmentation_type.upper()} (Run {run_id + 1}/{len(augmentation_types)})")
        print(f"{'='*70}\n")
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)

        best_val = 0.0
        best_epoch = 0
        best_state = None
        
        train_accs, train_losses = [], []
        val_accs, val_losses = [], []

        epochs = 35
        epoch_start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Run training and validation
            tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer=optimizer)
            va_loss, va_acc = run_epoch(model, val_loader, device, optimizer=None)
            scheduler.step()

            # Track metrics
            train_accs.append(tr_acc)
            train_losses.append(tr_loss)
            val_accs.append(va_acc)
            val_losses.append(va_loss)

            # Update best model
            if va_acc > best_val:
                best_val = va_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
            # Calculate time per epoch
            epoch_time = time.time() - epoch_start_time
            epoch_start_time = time.time()
            
            # Detailed logging with progress indicators
            train_val_gap = tr_acc - va_acc
            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Train: acc={tr_acc:.4f} loss={tr_loss:.4f} | "
                  f"Val: acc={va_acc:.4f} loss={va_loss:.4f} | "
                  f"Gap: {train_val_gap:+.4f} | "
                  f"Time: {epoch_time:.1f}s", end="")
            
            # Mark best epoch
            if epoch == best_epoch:
                print(" ⭐ BEST", end="")
            print()
            
            # Periodic detailed reports
            if epoch % 10 == 0 or epoch == epochs:
                print(f"   └─ [Checkpoint] Best Val: {best_val:.4f} at epoch {best_epoch}")

        print(f"\n✅ Training complete!")
        print(f"   Best validation accuracy: {best_val:.4f} (epoch {best_epoch})")

        # Test evaluation using best model
        print(f"\n📊 Evaluating on test set with best model (epoch {best_epoch})...")
        if best_state is not None:
            model.load_state_dict(best_state)

        te_loss, te_acc = run_epoch(model, test_loader, device, optimizer=None)
        print(f"   Test accuracy: {te_acc:.4f}")
        print(f"   Test loss: {te_loss:.4f}")
        
        # Calculate final train-val gap at best epoch
        final_train_val_gap = train_accs[best_epoch-1] - val_accs[best_epoch-1]
        print(f"   Train-Val gap at best epoch: {final_train_val_gap:+.4f}")
        
        # Calculate total training time
        run_time = time.time() - run_start_time
        print(f"   Total training time: {run_time/60:.1f} minutes")
        
        # Save model weights
        model_save_path = f"model_augmentation_{augmentation_type}.pth"
        torch.save({
            'model_state_dict': best_state,
            'augmentation_type': augmentation_type,
            'norm_type': norm_type,
            'activation_type': activation_type,
            'dropout_p': dropout_p,
            'best_val_acc': best_val,
            'best_epoch': best_epoch,
            'test_acc': te_acc,
            'test_loss': te_loss,
            'training_time': run_time
        }, model_save_path)
        print(f"   💾 Saved model: {model_save_path}")

        # Store results for comparison
        results[f"Aug_{augmentation_type}"] = {
            "best_val_acc": best_val,
            "best_epoch": best_epoch,
            "test_acc": te_acc,
            "test_loss": te_loss,
            "train_accs": train_accs,
            "train_losses": train_losses,
            "val_accs": val_accs,
            "val_losses": val_losses,
            "final_train_val_gap": final_train_val_gap,
            "training_time": run_time
        }
        
        print(f"\n{'='*70}\n")

    # Print comprehensive summary
    print(f"\n\n{'='*70}")
    print("            FINAL RESULTS: AUGMENTATION ABLATION STUDY")
    print(f"{'='*70}\n")
    
    print("📊 Performance Summary (BatchNorm, ReLU, Dropout=0.5):\n")
    print(f"{'Augmentation':<15} {'Val Acc':<10} {'Test Acc':<10} {'Test Loss':<10} {'Gap':<10} {'Best Epoch':<12} {'Time (min)':<12}")
    print("-" * 85)
    
    for run_name, metrics in results.items():
        aug_type = run_name.split("_")[1]
        print(f"{aug_type.upper():<15} "
              f"{metrics['best_val_acc']:<10.4f} "
              f"{metrics['test_acc']:<10.4f} "
              f"{metrics['test_loss']:<10.4f} "
              f"{metrics['final_train_val_gap']:+<10.4f} "
              f"{metrics['best_epoch']:<12} "
              f"{metrics['training_time']/60:<12.1f}")
    
    print("\n📈 Key Observations:")
    
    # Find best and worst performers
    best_test_acc = max(results.values(), key=lambda x: x['test_acc'])
    worst_test_acc = min(results.values(), key=lambda x: x['test_acc'])
    best_gap = min(results.values(), key=lambda x: abs(x['final_train_val_gap']))
    
    best_test_name = [k.split('_')[1] for k, v in results.items() if v == best_test_acc][0]
    worst_test_name = [k.split('_')[1] for k, v in results.items() if v == worst_test_acc][0]
    best_gap_name = [k.split('_')[1] for k, v in results.items() if v == best_gap][0]
    
    print(f"   • Best test accuracy: {best_test_name.upper()} ({best_test_acc['test_acc']:.4f})")
    print(f"   • Worst test accuracy: {worst_test_name.upper()} ({worst_test_acc['test_acc']:.4f})")
    print(f"   • Smallest train-val gap: {best_gap_name.upper()} ({best_gap['final_train_val_gap']:+.4f})")
    print(f"   • Improvement from baseline: {(best_test_acc['test_acc'] - results['Aug_none']['test_acc'])*100:.2f}%")
    
    # =========================================================================
    # VISUALIZATION 1: Training and Validation Curves
    # Shows accuracy and loss progression over epochs for each augmentation level
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("            GENERATING COMPARISON VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    print("📊 Creating training curves (accuracy & loss)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot accuracy curves
    for run_name, metrics in results.items():
        aug_type = run_name.split("_")[1]
        epochs_range = range(1, len(metrics['train_accs']) + 1)
        
        # Training curves with solid lines
        axes[0].plot(epochs_range, metrics['train_accs'], '-', 
                    label=f'Train ({aug_type})', linewidth=2.5, alpha=0.7)
        # Validation curves with dashed lines  
        axes[0].plot(epochs_range, metrics['val_accs'], '--', 
                    label=f'Val ({aug_type})', linewidth=2.5, alpha=0.8)
    
    axes[0].set_xlabel("Epoch", fontsize=13)
    axes[0].set_ylabel("Accuracy", fontsize=13)
    axes[0].set_title("Training vs Validation Accuracy", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Plot loss curves
    for run_name, metrics in results.items():
        aug_type = run_name.split("_")[1]
        epochs_range = range(1, len(metrics['train_losses']) + 1)
        
        axes[1].plot(epochs_range, metrics['train_losses'], '-', 
                    label=f'Train ({aug_type})', linewidth=2.5, alpha=0.7)
        axes[1].plot(epochs_range, metrics['val_losses'], '--', 
                    label=f'Val ({aug_type})', linewidth=2.5, alpha=0.8)
    
    axes[1].set_xlabel("Epoch", fontsize=13)
    axes[1].set_ylabel("Loss", fontsize=13)
    axes[1].set_title("Training vs Validation Loss", fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Impact of Data Augmentation on Training Dynamics", 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("training_curves_augmentation.png", dpi=150, bbox_inches='tight')
    print("   ✅ Saved: training_curves_augmentation.png")
    plt.show()
    
    # =========================================================================
    # VISUALIZATION 2: Final Test Accuracy Comparison Bar Chart
    # Compares final test performance across all augmentation strategies
    # Shows which augmentation provides best generalization
    # =========================================================================
    print("\n📊 Creating test accuracy comparison bar chart...")
    
    aug_labels = [run.split("_")[1] for run in results.keys()]
    test_accs = [results[run]['test_acc'] for run in results.keys()]
    
    # Create color gradient from red (low) to green (high)
    norm = plt.Normalize(vmin=min(test_accs), vmax=max(test_accs))
    colors = plt.cm.RdYlGn(norm(test_accs))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(aug_labels, test_accs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels on top of each bar
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel("Augmentation Type", fontsize=13, fontweight='bold')
    plt.ylabel("Test Accuracy", fontsize=13, fontweight='bold')
    plt.title("Final Test Accuracy by Augmentation Strategy", 
             fontsize=14, fontweight='bold', pad=15)
    plt.ylim([min(test_accs) - 0.02, max(test_accs) + 0.03])
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("test_accuracy_comparison_augmentation.png", dpi=150, bbox_inches='tight')
    print("   ✅ Saved: test_accuracy_comparison_augmentation.png")
    plt.show()
    
    # =========================================================================
    # VISUALIZATION 3: Train-Val Accuracy Gap Evolution
    # Shows overfitting tendency over training for each augmentation level
    # Lower gap = better generalization (augmentation working well)
    # =========================================================================
    print("\n📊 Creating train-val gap evolution plot...")
    
    # =========================================================================
    # VISUALIZATION 3: Train-Val Accuracy Gap Evolution
    # Shows overfitting tendency over training for each augmentation level
    # Lower gap = better generalization (augmentation working well)
    # =========================================================================
    print("\n📊 Creating train-val gap evolution plot...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for run_name, metrics in results.items():
        aug_type = run_name.split("_")[1]
        epochs_range = range(1, len(metrics['train_accs']) + 1)
        # Calculate gap: positive = overfitting, negative = underfitting
        gap = [t - v for t, v in zip(metrics['train_accs'], metrics['val_accs'])]
        
        ax.plot(epochs_range, gap, 'o-', label=f'{aug_type}', linewidth=2.5, markersize=5)
    
    ax.set_xlabel("Epoch", fontsize=13, fontweight='bold')
    ax.set_ylabel("Train - Val Accuracy Gap", fontsize=13, fontweight='bold')
    ax.set_title("Overfitting Reduction by Augmentation Strategy", 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    # Zero line: gap=0 means perfect train-val balance
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect Balance')
    
    plt.tight_layout()
    plt.savefig("augmentation_gap.png", dpi=150, bbox_inches='tight')
    print("   ✅ Saved: augmentation_gap.png")
    plt.show()
    
    # =========================================================================
    # KEY INSIGHTS AND TAKEAWAYS
    # =========================================================================
    print(f"\n{'='*70}")
    print("                        🎓 KEY INSIGHTS")
    print(f"{'='*70}")
    print("✓ Data augmentation reduces overfitting by creating realistic training variations")
    print("✓ Progressive augmentation levels balance regularization vs information preservation")
    print("✓ Train-val gap is a key metric: smaller gap = better generalization")
    print("✓ Test accuracy reveals which augmentation best transfers to unseen data")
    print("✓ Cutout/Random Erasing helps model learn robust features despite occlusions")
    print("✓ Color jitter improves robustness to lighting and color variations")
    print(f"{'='*70}\n")
    
    print(f"\n{'='*70}")
    print("                    🎉 AUGMENTATION STUDY COMPLETE 🎉")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
