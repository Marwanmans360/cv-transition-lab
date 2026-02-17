"""
Plain Deep CNN vs ResNet Comparison
====================================

Systematic comparison of plain deep CNNs vs ResNets on CIFAR-10.

Key Experiments:
1. Plain Deep CNN (16 conv layers, no residuals)
   - Expected: harder optimization, slower convergence, potentially worse accuracy
   
2. ResNet with skip connections (same depth)
   - Expected: faster convergence, better final accuracy, more stable training

Using best augmentation strategy: crop_flip (RandomCrop + RandomHorizontalFlip)
Using same hyperparameters: lr=0.01, momentum=0.9, weight_decay=1e-4

Author: Deep Architecture Study
Date: February 2026
"""

import os
import sys
import time
import pickle
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# ============================================================================
# Reproducibility & Device Setup
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n{'='*70}")
        print(f"🚀 GPU DETECTED")
        print(f"{'='*70}")
        print(f"   Device: {gpu_name}")
        print(f"   Memory: {gpu_memory:.2f} GB")
        print(f"{'='*70}\n")
    else:
        device = torch.device('cpu')
        print(f"\n{'='*70}")
        print(f"⚠️  WARNING: Running on CPU")
        print(f"{'='*70}\n")
    return device


# ============================================================================
# CIFAR-10 Data Loading
# ============================================================================

def load_cifar10_batch(batch_file_path):
    """Load a single CIFAR-10 batch file."""
    with open(batch_file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch


def load_cifar10_from_folder(cifar10_folder):
    """Load all CIFAR-10 training and test data."""
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
    """Convert CIFAR-10 flat format to HWC uint8 format."""
    X_flat = np.asarray(X_flat)
    assert X_flat.ndim == 2 and X_flat.shape[1] == 3072
    X_hwc = X_flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X_hwc.astype(np.uint8)


# ============================================================================
# Dataset Class
# ============================================================================

class CIFAR10Dataset(Dataset):
    """Custom CIFAR-10 dataset with transforms."""
    def __init__(self, X, y, transform=None):
        self.X = X  # HWC uint8 format
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        
        return img, label


# ============================================================================
# LESSON 1: Basic Residual Block
# ============================================================================

class BasicBlock(nn.Module):
    """
    Basic Residual Block - The fundamental building block of ResNet
    
    CONCEPT:
    --------
    Traditional: output = F(x)
    Residual:    output = F(x) + x  (then ReLU)
    
    Where F(x) is: Conv → BN → ReLU → Conv → BN
    
    WHY IT WORKS:
    -------------
    1. The skip connection (x) provides a direct path for gradients
    2. If the optimal mapping is identity, F(x) can learn to be zero
    3. Makes optimization easier for deep networks
    
    IMPLEMENTATION DETAILS:
    -----------------------
    - When input/output channels differ, we need a projection shortcut
    - When spatial size changes (stride=2), shortcut must also downsample
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first conv (1 for same size, 2 for downsample)
        """
        super(BasicBlock, self).__init__()
        
        # ===== RESIDUAL PATH F(x) =====
        # First conv: may downsample via stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second conv: always stride=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # ===== SKIP CONNECTION (SHORTCUT) =====
        # If dimensions change, we need a projection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Use 1x1 conv to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        STEP BY STEP:
        1. Save input for skip connection: identity = x
        2. Apply residual function: out = F(x) through conv-bn-relu-conv-bn
        3. Add skip connection: out = out + identity
        4. Apply final ReLU: out = ReLU(out)
        """
        # Save input for skip connection
        identity = x
        
        # Residual function F(x)
        out = self.conv1(x)      # Conv with potential downsampling
        out = self.bn1(out)      # Batch normalization
        out = self.relu(out)     # First ReLU
        
        out = self.conv2(out)    # Second conv
        out = self.bn2(out)      # Batch normalization
        
        # Skip connection (project if needed)
        identity = self.shortcut(identity)
        
        # THE KEY STEP: Add residual
        out = out + identity     # Element-wise addition
        
        # Final activation AFTER addition
        out = self.relu(out)
        
        return out


# ============================================================================
# LESSON 2: ResNet Architecture (Same Depth as Plain CNN)
# ============================================================================

class ResNet16(nn.Module):
    """
    ResNet with 16 convolutional layers (same as PlainDeepCNN for fair comparison)
    
    ARCHITECTURE DESIGN:
    --------------------
    - Uses BasicBlock (2 convs per block)
    - 4 stages, 2 blocks per stage → 2 * 2 * 4 = 16 conv layers
    - Each stage doubles channels and halves spatial size
    - Skip connections in every BasicBlock
    
    COMPARISON WITH PLAIN CNN:
    --------------------------
    Plain CNN:  Conv→BN→ReLU → Conv→BN→ReLU → Conv→BN→ReLU → ...
    ResNet:     [Conv→BN→ReLU → Conv→BN] + skip → ReLU → ...
                 \_________F(x)_________/    |
                                             x
    """
    
    def __init__(self, num_classes=10):
        super(ResNet16, self).__init__()
        
        # Initial conv layer (not counted in the 16)
        # This matches common ResNet designs
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1: 16 channels, 32x32 spatial size
        # 2 BasicBlocks = 4 conv layers
        self.stage1 = self._make_stage(16, 16, num_blocks=2, stride=1)
        
        # Stage 2: 32 channels, 16x16 spatial size
        # 2 BasicBlocks = 4 conv layers
        self.stage2 = self._make_stage(16, 32, num_blocks=2, stride=2)
        
        # Stage 3: 64 channels, 8x8 spatial size
        # 2 BasicBlocks = 4 conv layers
        self.stage3 = self._make_stage(32, 64, num_blocks=2, stride=2)
        
        # Stage 4: 128 channels, 4x4 spatial size
        # 2 BasicBlocks = 4 conv layers
        self.stage4 = self._make_stage(64, 128, num_blocks=2, stride=2)
        
        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
        # Total: 16 conv layers in residual blocks (same as PlainDeepCNN)
        self.total_conv_layers = 16
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a stage with multiple BasicBlocks.
        
        Args:
            in_channels: Input channels for this stage
            out_channels: Output channels for this stage
            num_blocks: Number of BasicBlocks in this stage
            stride: Stride for first block (1 or 2)
        
        Note: First block may downsample (stride=2), rest have stride=1
        """
        layers = []
        
        # First block: may change dimensions
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # Remaining blocks: same dimensions
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through ResNet.
        
        GRADIENT FLOW:
        --------------
        With skip connections, gradients can flow DIRECTLY from output
        back to early layers, avoiding the vanishing gradient problem.
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual stages (with skip connections)
        x = self.stage1(x)  # Each stage has multiple skip connections
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ============================================================================
# Plain Deep CNN Architecture (No Residuals)
# ============================================================================

class PlainDeepCNN(nn.Module):
    """
    Plain Deep CNN with 16 convolutional layers (no skip connections).
    
    Architecture:
    - 16 conv layers (3x3 kernel) organized in 4 stages
    - Each stage has 4 conv layers with same channel count
    - Downsampling via stride=2 between stages
    - BatchNorm + ReLU after each conv
    - Expected to suffer from optimization difficulties
    """
    def __init__(self, num_classes=10):
        super(PlainDeepCNN, self).__init__()
        
        # Stage 1: 16 channels, 32x32 spatial size
        self.stage1 = self._make_plain_stage(3, 16, num_layers=4, first_stride=1)
        
        # Stage 2: 32 channels, 16x16 spatial size
        self.stage2 = self._make_plain_stage(16, 32, num_layers=4, first_stride=2)
        
        # Stage 3: 64 channels, 8x8 spatial size
        self.stage3 = self._make_plain_stage(32, 64, num_layers=4, first_stride=2)
        
        # Stage 4: 128 channels, 4x4 spatial size
        self.stage4 = self._make_plain_stage(64, 128, num_layers=4, first_stride=2)
        
        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
        # Count total conv layers
        self.total_conv_layers = 16
        
    def _make_plain_stage(self, in_channels, out_channels, num_layers, first_stride):
        """Create a plain stage with multiple conv-bn-relu layers."""
        layers = []
        
        # First layer may downsample
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        
        # Remaining layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                         stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += X.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * X.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += X.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def initialize_weights(model):
    """Initialize model weights using He initialization."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)


# ============================================================================
# Main Training Loop
# ============================================================================

def train_model(model, train_loader, val_loader, config, device):
    """
    Train a model with the given configuration.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Dictionary with training configuration
        device: cuda or cpu
    
    Returns:
        Dictionary with training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['milestones'],
        gamma=config['gamma']
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    
    print(f"\n{'='*70}")
    print(f"Training {config['name']}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
    
    print(f"\n✅ Training complete! Best Val Acc: {best_val_acc:.4f}\n")
    
    return history


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_comparison(results, save_path=None):
    """
    Plot training curves comparing plain deep CNN vs ResNet.
    
    EDUCATIONAL PURPOSE:
    This visualization helps you SEE the difference that skip connections make:
    - Smoother loss curves (better optimization)
    - Higher accuracy (better performance)
    - More stable training (less variance)
    
    Args:
        results: Dictionary with training histories for each model
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    colors = {'Plain Deep CNN': '#e74c3c', 'ResNet': '#27ae60', 'ResNet-16': '#27ae60'}
    markers = {'Plain Deep CNN': 'o', 'ResNet': 's', 'ResNet-16': 's'}
    
    # Plot 1: Training Loss - Shows optimization difficulty
    ax = axes[0, 0]
    for name, history in results.items():
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], label=name, 
                color=colors.get(name, 'blue'), linewidth=2.5, alpha=0.8,
                marker=markers.get(name, 'o'), markersize=4, markevery=5)
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training Loss: ResNet Optimizes More Smoothly', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, '👀 Look for: Smoother curve = better optimization',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot 2: Validation Accuracy - Shows generalization
    ax = axes[0, 1]
    for name, history in results.items():
        epochs = range(1, len(history['val_acc']) + 1)
        ax.plot(epochs, history['val_acc'], label=name,
                color=colors.get(name, 'blue'), linewidth=2.5, alpha=0.8,
                marker=markers.get(name, 'o'), markersize=4, markevery=5)
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Validation Accuracy: ResNet Achieves Higher Performance',
                 fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, '👀 Look for: Higher peak = better model',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Plot 3: Train vs Val Loss - Shows overfitting
    ax = axes[1, 0]
    for name, history in results.items():
        epochs = range(1, len(history['train_loss']) + 1)
        color = colors.get(name, 'blue')
        ax.plot(epochs, history['train_loss'], '--', 
                label=f'{name} (train)', color=color, linewidth=2, alpha=0.6)
        ax.plot(epochs, history['val_loss'], '-',
                label=f'{name} (val)', color=color, linewidth=2.5, alpha=0.9)
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Train vs Val Loss: Gap Shows Overfitting',
                 fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, '👀 Look for: Smaller gap = less overfitting',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Plot 4: Training Accuracy - Shows learning speed
    ax = axes[1, 1]
    for name, history in results.items():
        epochs = range(1, len(history['train_acc']) + 1)
        ax.plot(epochs, history['train_acc'], label=name,
                color=colors.get(name, 'blue'), linewidth=2.5, alpha=0.8,
                marker=markers.get(name, 'o'), markersize=4, markevery=5)
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Training Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Training Accuracy: Convergence Speed',
                 fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, '👀 Look for: Faster rise = quicker learning',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    # Add overall title
    fig.suptitle('Plain Deep CNN vs ResNet: The Power of Skip Connections',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Saved training comparison plot to {save_path}")
    
    plt.show()


def print_summary_table(results):
    """Print a summary table of results."""
    print(f"\n{'='*70}")
    print("SUMMARY RESULTS")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<20} | {'Best Val Acc':<12} | {'Final Val Acc':<12} | {'Final Train Acc':<12}")
    print("-" * 70)
    
    for name, history in results.items():
        best_val = max(history['val_acc'])
        final_val = history['val_acc'][-1]
        final_train = history['train_acc'][-1]
        print(f"{name:<20} | {best_val:<12.4f} | {final_val:<12.4f} | {final_train:<12.4f}")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# Educational Visualization
# ============================================================================

def print_architecture_comparison():
    """Print a visual comparison of Plain CNN vs ResNet architectures."""
    print("\n" + "="*70)
    print("ARCHITECTURE COMPARISON: Plain CNN vs ResNet")
    print("="*70 + "\n")
    
    print("PLAIN DEEP CNN (Traditional):")
    print("-" * 70)
    print("""
    Input
      ↓
    Conv → BN → ReLU
      ↓
    Conv → BN → ReLU
      ↓
    Conv → BN → ReLU
      ↓
    Conv → BN → ReLU
      ↓
      ...
      ↓
    Output
    
    Problem: Gradients get weaker as they backpropagate through many layers
    Result:  Hard to train, may suffer degradation
    """)
    
    print("\n" + "-" * 70)
    print("RESNET (With Skip Connections):")
    print("-" * 70)
    print("""
    Input (x)
      ↓─────────┐                ← Skip connection!
    Conv → BN → ReLU             ← F(x) path
      ↓         │
    Conv → BN   │
      ↓         │
      +─────────┘                ← Add: F(x) + x
      ↓
    ReLU
      ↓
      ╫─────────┐                ← Another skip!
    Conv → BN → ReLU
      ↓         │
    Conv → BN   │
      ↓         │
      +─────────┘
      ↓
    ReLU
      ↓
    Output
    
    Key Ideas:
    1. y = F(x) + x    (residual learning)
    2. Gradient can flow directly through skip (↓ path)
    3. If identity is optimal, F(x) can learn to be ~0
    4. Much easier to optimize!
    """)
    print("="*70 + "\n")


def print_residual_math_explanation():
    """Explain the mathematical intuition behind residual learning."""
    print("\n" + "="*70)
    print("MATHEMATICAL INTUITION: Why Residuals Work")
    print("="*70 + "\n")
    
    print("Traditional Learning:")
    print("-" * 70)
    print("Goal: Learn mapping H(x) directly")
    print("  H(x) = desired output")
    print("  Network must learn the complete transformation\n")
    
    print("Residual Learning:")
    print("-" * 70)
    print("Goal: Learn residual F(x) = H(x) - x")
    print("  Then: H(x) = F(x) + x")
    print("  Network only learns the 'difference' from identity\n")
    
    print("Why is this easier?")
    print("-" * 70)
    print("Example: If optimal mapping is close to identity (H(x) ≈ x)")
    print("  • Traditional: Must learn H(x) = x precisely")
    print("  • Residual:    Only learn F(x) ≈ 0 (much easier!)\n")
    
    print("Gradient Flow:")
    print("-" * 70)
    print("  ∂Loss/∂x = ∂Loss/∂y × (∂F/∂x + 1)")
    print("                        ^^^^^^^^^")
    print("            The '+1' ensures gradient always flows!")
    print("            Even if ∂F/∂x vanishes, gradient still propagates\n")
    
    print("="*70 + "\n")


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    """Run the Plain Deep CNN vs ResNet comparison experiment."""
    
    print("\n" + "="*70)
    print("PLAIN DEEP CNN vs RESNET COMPARISON")
    print("="*70 + "\n")
    
    # Educational explanations
    print_architecture_comparison()
    print_residual_math_explanation()
    
    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    
    # Configuration
    CIFAR10_PATH = 'data/cifar-10-batches-py'
    RESULTS_DIR = 'results/plain_vs_resnet'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Training configuration (same as best augmentation experiment)
    config = {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'milestones': [20, 30],
        'gamma': 0.1,
        'epochs': 35,
        'batch_size': 64
    }
    
    print("📋 Training Configuration:")
    print(f"   Learning Rate: {config['lr']}")
    print(f"   Momentum: {config['momentum']}")
    print(f"   Weight Decay: {config['weight_decay']}")
    print(f"   LR Milestones: {config['milestones']}")
    print(f"   LR Gamma: {config['gamma']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch Size: {config['batch_size']}\n")
    
    # Load CIFAR-10
    print("📦 Loading CIFAR-10 dataset...")
    (X_train_flat, y_train), (X_test_flat, y_test) = load_cifar10_from_folder(CIFAR10_PATH)
    X_train_hwc = cifar10_flat_to_hwc_uint8(X_train_flat)
    X_test_hwc = cifar10_flat_to_hwc_uint8(X_test_flat)
    
    # Split into train/val
    n_train = 45000
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, 50000))
    
    X_train = X_train_hwc[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train_hwc[val_indices]
    y_val = y_train[val_indices]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test_hwc)}\n")
    
    # CIFAR-10 normalization statistics
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    # Data augmentation: crop_flip (best performing strategy)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    print("🎨 Data Augmentation: crop_flip (RandomCrop + RandomHorizontalFlip)")
    print(f"   Normalization: mean={mean}, std={std}\n")
    
    # Create datasets and loaders
    train_dataset = CIFAR10Dataset(X_train, y_train_split, transform=train_transform)
    val_dataset = CIFAR10Dataset(X_val, y_val, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=2)
    
    # ========================================================================
    # Experiment 1: Plain Deep CNN (16 layers, no residuals)
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: PLAIN DEEP CNN (16 conv layers, NO residuals)")
    print("="*70 + "\n")
    
    print("📚 THEORY: Plain Deep Networks")
    print("-" * 70)
    print("As we stack more layers, we expect:")
    print("  ✗ Optimization becomes harder (vanishing gradients)")
    print("  ✗ Training loss may not decrease smoothly")
    print("  ✗ May underperform shallower networks (degradation problem)")
    print("-" * 70 + "\n")
    
    model_plain = PlainDeepCNN(num_classes=10).to(device)
    initialize_weights(model_plain)
    
    # Count parameters
    total_params_plain = sum(p.numel() for p in model_plain.parameters())
    trainable_params_plain = sum(p.numel() for p in model_plain.parameters() if p.requires_grad)
    
    print(f"🏗️  Model Architecture:")
    print(f"   Total Conv Layers: {model_plain.total_conv_layers}")
    print(f"   Total Parameters: {total_params_plain:,}")
    print(f"   Trainable Parameters: {trainable_params_plain:,}\n")
    
    # Sanity check
    dummy = torch.randn(2, 3, 32, 32).to(device)
    out = model_plain(dummy)
    print(f"   Output shape: {tuple(out.shape)} ✓\n")
    
    # Train Plain Deep CNN
    config['name'] = 'Plain Deep CNN'
    history_plain = train_model(model_plain, train_loader, val_loader, config, device)
    
    # ========================================================================
    # Experiment 2: ResNet-16 (16 layers, WITH residuals)
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: RESNET-16 (16 conv layers, WITH residuals)")
    print("="*70 + "\n")
    
    print("📚 THEORY: Residual Learning")
    print("-" * 70)
    print("With skip connections (y = F(x) + x):")
    print("  ✓ Gradients flow directly through shortcuts")
    print("  ✓ Easier to optimize (learning residual F(x) vs full mapping)")
    print("  ✓ No degradation - can at least learn identity")
    print("  ✓ Training loss should decrease more smoothly")
    print("-" * 70 + "\n")
    
    model_resnet = ResNet16(num_classes=10).to(device)
    initialize_weights(model_resnet)
    
    # Count parameters
    total_params_resnet = sum(p.numel() for p in model_resnet.parameters())
    trainable_params_resnet = sum(p.numel() for p in model_resnet.parameters() if p.requires_grad)
    
    print(f"🏗️  Model Architecture:")
    print(f"   Total Conv Layers: {model_resnet.total_conv_layers}")
    print(f"   Total Parameters: {total_params_resnet:,}")
    print(f"   Trainable Parameters: {trainable_params_resnet:,}")
    print(f"   Skip Connections: 8 BasicBlocks (2 per stage × 4 stages)\n")
    
    # Sanity check
    out = model_resnet(dummy)
    print(f"   Output shape: {tuple(out.shape)} ✓\n")
    
    # Train ResNet
    config['name'] = 'ResNet-16'
    history_resnet = train_model(model_resnet, train_loader, val_loader, config, device)
    
    # ========================================================================
    # Comparison Analysis
    # ========================================================================
    
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70 + "\n")
    
    # Combine results
    results = {
        'Plain Deep CNN': history_plain,
        'ResNet': history_resnet
    }
    
    # Plot comparison
    plot_training_comparison(results, 
                            save_path=os.path.join(RESULTS_DIR, 'plain_vs_resnet_comparison.png'))
    
    # Print summary
    print_summary_table(results)
    
    # Additional analysis
    print("\n📊 DETAILED COMPARISON:\n")
    print(f"{'Metric':<30} | {'Plain CNN':<15} | {'ResNet-16':<15} | {'Winner':<10}")
    print("-" * 75)
    
    best_val_plain = max(history_plain['val_acc'])
    best_val_resnet = max(history_resnet['val_acc'])
    print(f"{'Best Validation Accuracy':<30} | {best_val_plain:<15.4f} | {best_val_resnet:<15.4f} | "
          f"{'ResNet' if best_val_resnet > best_val_plain else 'Plain':<10}")
    
    final_val_plain = history_plain['val_acc'][-1]
    final_val_resnet = history_resnet['val_acc'][-1]
    print(f"{'Final Validation Accuracy':<30} | {final_val_plain:<15.4f} | {final_val_resnet:<15.4f} | "
          f"{'ResNet' if final_val_resnet > final_val_plain else 'Plain':<10}")
    
    final_train_plain = history_plain['train_acc'][-1]
    final_train_resnet = history_resnet['train_acc'][-1]
    print(f"{'Final Training Accuracy':<30} | {final_train_plain:<15.4f} | {final_train_resnet:<15.4f} | "
          f"{'ResNet' if final_train_resnet > final_train_plain else 'Plain':<10}")
    
    # Training speed (epochs to reach certain accuracy)
    threshold = 0.70
    epochs_to_threshold_plain = next((i+1 for i, acc in enumerate(history_plain['val_acc']) 
                                     if acc >= threshold), config['epochs'])
    epochs_to_threshold_resnet = next((i+1 for i, acc in enumerate(history_resnet['val_acc'])
                                      if acc >= threshold), config['epochs'])
    print(f"{'Epochs to 70% Val Acc':<30} | {epochs_to_threshold_plain:<15} | {epochs_to_threshold_resnet:<15} | "
          f"{'ResNet' if epochs_to_threshold_resnet < epochs_to_threshold_plain else 'Plain':<10}")
    
    print()
    
    # Parameter comparison
    print(f"{'Total Parameters':<30} | {total_params_plain:<15,} | {total_params_resnet:<15,} | "
          f"{'Similar':<10}")
    
    print("\n" + "="*75 + "\n")
    
    # Key observations
    print("🔍 KEY OBSERVATIONS:\n")
    print("1. Training Loss Convergence:")
    print("   - Plain CNN: Training may be unstable, loss may plateau")
    print("   - ResNet: Smoother convergence, more reliable optimization\n")
    
    print("2. Validation Accuracy:")
    if best_val_resnet > best_val_plain:
        improvement = (best_val_resnet - best_val_plain) * 100
        print(f"   - ResNet achieves {improvement:.2f}% higher accuracy")
        print("   - Skip connections enable better generalization\n")
    else:
        print("   - Comparable performance (both architectures work)\n")
    
    print("3. Optimization Stability:")
    print("   - ResNet should show more stable training curves")
    print("   - Plain CNN may show more variance in validation accuracy\n")
    
    print("4. What Skip Connections Changed:")
    print("   ✓ Gradient Flow: Direct paths avoid vanishing gradients")
    print("   ✓ Optimization: Easier to learn residuals F(x) than full H(x)")
    print("   ✓ Stability: Training is more reliable and predictable")
    print("   ✓ Depth: Can go deeper without degradation\n")
    
    # Save all results
    results_file = os.path.join(RESULTS_DIR, 'comparison_results.json')
    comparison_data = {
        'plain_cnn': history_plain,
        'resnet': history_resnet,
        'summary': {
            'plain_best_val': float(best_val_plain),
            'resnet_best_val': float(best_val_resnet),
            'plain_params': total_params_plain,
            'resnet_params': total_params_resnet,
            'epochs_to_70_plain': int(epochs_to_threshold_plain),
            'epochs_to_70_resnet': int(epochs_to_threshold_resnet)
        }
    }
    with open(results_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"💾 Saved comparison results to {results_file}\n")
    
    print("="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("\n✅ You now understand how skip connections transform deep learning!\n")
    

if __name__ == "__main__":
    main()
