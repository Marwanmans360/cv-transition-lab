"""
CNN Augmentation Advanced Experiments
======================================

Systematic study of data augmentation techniques with proper methodology:

Step 0: Multi-seed baseline anchoring (3 seeds, crop_flip)
Step 1: Clean train accuracy evaluation (measure true train-val gap)
Step 2: Cutout tuning grid (p, length, dropout combinations)
Step 3: RandAugment integration (compare vs baseline)
Step 4: MixUp/CutMix experiments (mixing augmentations)
Step 5: Extended training schedule (200 epochs with cosine annealing)

Author: Advanced augmentation study
Date: February 2026
"""

import os
import sys
import time
import pickle
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
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
    """Get the best available device with detailed GPU information."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"\n{'='*70}")
        print(f"🚀 GPU DETECTED AND ENABLED")
        print(f"{'='*70}")
        print(f"   Device: {gpu_name}")
        print(f"   Total Memory: {gpu_memory:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        print(f"{'='*70}\n")
    else:
        device = torch.device('cpu')
        print(f"\n{'='*70}")
        print(f"⚠️  WARNING: GPU NOT AVAILABLE")
        print(f"{'='*70}")
        print("   Running on CPU - Training will be significantly slower!")
        print("   For Google Colab: Runtime > Change runtime type > GPU")
        print(f"{'='*70}\n")
    return device


def print_gpu_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "CPU mode"


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for experiments."""
    # Paths (update for your environment)
    DATA_PATH = '/content/drive/MyDrive/datasets/cifar-10-batches-py'
    RESULTS_DIR = 'results_advanced_aug'
    
    # Dataset
    NUM_CLASSES = 10
    IMG_SIZE = 32
    
    # Training schedules
    SCHEDULE_QUICK = {
        'epochs': 35,
        'lr': 0.01,
        'scheduler': 'multistep',
        'milestones': [20, 30],
        'gamma': 0.1
    }
    
    SCHEDULE_STANDARD = {
        'epochs': 200,
        'lr': 0.1,
        'scheduler': 'cosine',
        'milestones': None,
        'gamma': None
    }
    
    # Training hyperparameters
    BATCH_SIZE = 64
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    
    # Multi-seed
    SEEDS = [42, 123, 456]
    
    # Model
    DROPOUT = 0.5  # Default dropout
    
    # Augmentation
    NORMALIZE_MEAN = [0.4914, 0.4822, 0.4465]
    NORMALIZE_STD = [0.2470, 0.2435, 0.2616]


# ============================================================================
# CIFAR-10 Data Loading
# ============================================================================

def unpickle(file):
    """Load CIFAR-10 batch file."""
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data


def load_cifar10_data(data_path):
    """
    Load CIFAR-10 dataset from pickle files.
    
    Returns:
        X_train: (50000, 32, 32, 3) uint8
        y_train: (50000,) int64
        X_test: (10000, 32, 32, 3) uint8
        y_test: (10000,) int64
    """
    # Load training batches
    X_train_batches = []
    y_train_batches = []
    
    for i in range(1, 6):
        batch_file = os.path.join(data_path, f'data_batch_{i}')
        batch_dict = unpickle(batch_file)
        X_batch = batch_dict[b'data']  # (10000, 3072)
        y_batch = batch_dict[b'labels']  # list of 10000 labels
        
        # Reshape to (10000, 3, 32, 32) then transpose to (10000, 32, 32, 3)
        X_batch = X_batch.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        X_train_batches.append(X_batch)
        y_train_batches.append(y_batch)
    
    X_train = np.concatenate(X_train_batches, axis=0)  # (50000, 32, 32, 3)
    y_train = np.concatenate(y_train_batches, axis=0)  # (50000,)
    
    # Load test batch
    test_file = os.path.join(data_path, 'test_batch')
    test_dict = unpickle(test_file)
    X_test = test_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_dict[b'labels'])
    
    print(f"Loaded CIFAR-10:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def split_train_val(X_train, y_train, val_size=10000, seed=42):
    """Split training data into train and validation sets."""
    np.random.seed(seed)
    n_total = len(X_train)
    indices = np.random.permutation(n_total)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_tr = X_train[train_indices]
    y_tr = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    
    print(f"Split: Train={len(X_tr)}, Val={len(X_val)}")
    
    return X_tr, y_tr, X_val, y_val


# ============================================================================
# Custom Dataset with Augmentation
# ============================================================================

class AugmentedTensorDataset(Dataset):
    """Dataset that applies transforms on-the-fly to pre-loaded numpy arrays."""
    
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: numpy array of shape (N, H, W, C), dtype uint8
            y: numpy array of shape (N,), dtype int64
            transform: torchvision transforms to apply
        """
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = self.X[idx]  # (32, 32, 3) uint8 numpy array
        label = int(self.y[idx])
        
        # Convert numpy array to PIL Image (required by torchvision transforms)
        img = Image.fromarray(img)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            # Default: just convert to tensor and normalize
            img = transforms.ToTensor()(img)
        
        return img, label


# ============================================================================
# Augmentation Transforms
# ============================================================================

class CutoutFixed:
    """Cutout augmentation with fixed patch size."""
    
    def __init__(self, length=16, p=0.5):
        """
        Args:
            length: side length of the square cutout patch
            p: probability of applying cutout
        """
        self.length = length
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            img: Tensor of shape (C, H, W)
        
        Returns:
            img with cutout applied
        """
        if np.random.rand() > self.p:
            return img
        
        h, w = img.size(1), img.size(2)
        
        # Random center
        cy = np.random.randint(0, h)
        cx = np.random.randint(0, w)
        
        # Compute cutout box
        y1 = np.clip(cy - self.length // 2, 0, h)
        y2 = np.clip(cy + self.length // 2, 0, h)
        x1 = np.clip(cx - self.length // 2, 0, w)
        x2 = np.clip(cx + self.length // 2, 0, w)
        
        # Fill with zeros (or mean)
        img[:, y1:y2, x1:x2] = 0.0
        
        return img


def get_augmentation_transform(aug_type='crop_flip', cutout_p=0.5, cutout_length=16):
    """
    Get augmentation pipeline.
    
    Args:
        aug_type: 'none', 'crop_flip', 'color_jitter', 'cutout', 'advanced', 'randaugment'
        cutout_p: probability of applying cutout
        cutout_length: size of cutout patch
    
    Returns:
        torchvision.transforms.Compose
    """
    normalize = transforms.Normalize(
        mean=Config.NORMALIZE_MEAN,
        std=Config.NORMALIZE_STD
    )
    
    if aug_type == 'none':
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    elif aug_type == 'crop_flip':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
    
    elif aug_type == 'color_jitter':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
    
    elif aug_type == 'cutout':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
            CutoutFixed(length=cutout_length, p=cutout_p)
        ])
    
    elif aug_type == 'advanced':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
            CutoutFixed(length=cutout_length, p=cutout_p)
        ])
    
    elif aug_type == 'randaugment':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize
        ])
    
    else:
        raise ValueError(f"Unknown aug_type: {aug_type}")
    
    return transform


def get_test_transform():
    """Get test/validation transform (no augmentation)."""
    normalize = transforms.Normalize(
        mean=Config.NORMALIZE_MEAN,
        std=Config.NORMALIZE_STD
    )
    return transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])


# ============================================================================
# MixUp and CutMix
# ============================================================================

def mix_up_data(X, y, alpha=0.2):
    """
    Apply MixUp augmentation.
    
    Args:
        X: batch of images (B, C, H, W)
        y: batch of labels (B,)
        alpha: mixup interpolation strength
    
    Returns:
        mixed_X: mixed images
        y_a: labels for first component
        y_b: labels for second component
        lam: mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = X.size(0)
    index = torch.randperm(batch_size).to(X.device)
    
    mixed_X = lam * X + (1 - lam) * X[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_X, y_a, y_b, lam


def cutmix_data(X, y, alpha=1.0):
    """
    Apply CutMix augmentation.
    
    Args:
        X: batch of images (B, C, H, W)
        y: batch of labels (B,)
        alpha: cutmix beta distribution parameter
    
    Returns:
        mixed_X: images with cutmix applied
        y_a: labels for first component
        y_b: labels for second component
        lam: mixing coefficient (area ratio)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = X.size(0)
    index = torch.randperm(batch_size).to(X.device)
    
    # Get image dimensions
    _, _, H, W = X.size()
    
    # Random box
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    X[:, :, bby1:bby2, bbx1:bbx2] = X[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return X, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# MiniVGG Model
# ============================================================================

class MiniVGG(nn.Module):
    """
    Simplified VGG-style CNN for CIFAR-10.
    
    Architecture:
        Conv Block 1: 32 filters
        Conv Block 2: 64 filters
        Conv Block 3: 128 filters
        FC: 256 units
        Output: 10 classes
    
    Each conv block: Conv-BN-ReLU-Conv-BN-ReLU-MaxPool-Dropout
    """
    
    def __init__(self, num_classes=10, dropout=0.5):
        super(MiniVGG, self).__init__()
        
        # Conv Block 1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        self.dropout1 = nn.Dropout2d(p=dropout)
        
        # Conv Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        self.dropout2 = nn.Dropout2d(p=dropout)
        
        # Conv Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        self.dropout3 = nn.Dropout2d(p=dropout)
        
        # Fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Training Functions
# ============================================================================

def run_epoch(model, loader, criterion, optimizer, device, 
              is_training=True, grad_clip=None, mixing_aug=None, mixing_alpha=None):
    """
    Run one epoch of training or evaluation.
    
    Args:
        model: PyTorch model
        loader: DataLoader
        criterion: loss function
        optimizer: optimizer (only used if is_training=True)
        device: torch device
        is_training: whether to train or evaluate
        grad_clip: gradient clipping threshold
        mixing_aug: 'mixup' or 'cutmix' or None
        mixing_alpha: alpha parameter for mixing augmentations
    
    Returns:
        avg_loss: average loss
        avg_acc: average accuracy
    """
    if is_training:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Verify GPU usage on first batch (one-time check)
    first_batch = True
    
    with torch.set_grad_enabled(is_training):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            
            # One-time GPU verification
            if first_batch and is_training:
                if torch.cuda.is_available():
                    assert X.is_cuda, "ERROR: Data not on GPU!"
                    assert next(model.parameters()).is_cuda, "ERROR: Model not on GPU!"
                first_batch = False
            
            # Apply mixing augmentation if specified
            if is_training and mixing_aug == 'mixup' and mixing_alpha is not None:
                X, y_a, y_b, lam = mix_up_data(X, y, mixing_alpha)
                outputs = model(X)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            elif is_training and mixing_aug == 'cutmix' and mixing_alpha is not None:
                X, y_a, y_b, lam = cutmix_data(X, y, mixing_alpha)
                outputs = model(X)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(X)
                loss = criterion(outputs, y)
            
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            total_loss += loss.item() * X.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    avg_loss = total_loss / total
    avg_acc = correct / total
    
    return avg_loss, avg_acc


def train_model(model, train_loader, val_loader, test_loader, 
                device, config_schedule, 
                mixing_aug=None, mixing_alpha=None,
                train_clean_loader=None,
                exp_name='experiment'):
    """
    Train model with specified schedule.
    
    Args:
        model: PyTorch model
        train_loader: training DataLoader (potentially augmented)
        val_loader: validation DataLoader (clean)
        test_loader: test DataLoader (clean)
        device: torch device
        config_schedule: dictionary with training schedule config
        mixing_aug: 'mixup', 'cutmix', or None
        mixing_alpha: alpha for mixing augmentation
        train_clean_loader: optional clean training DataLoader for gap measurement
        exp_name: experiment name for saving checkpoints
    
    Returns:
        history: dictionary with training history
        best_model_state: state dict of best model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config_schedule['lr'],
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Scheduler
    if config_schedule['scheduler'] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config_schedule['milestones'],
            gamma=config_schedule['gamma']
        )
    elif config_schedule['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config_schedule['epochs']
        )
    else:
        scheduler = None
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
        'train_clean_loss': [],
        'train_clean_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    
    num_epochs = config_schedule['epochs']
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            is_training=True,
            grad_clip=Config.GRAD_CLIP,
            mixing_aug=mixing_aug,
            mixing_alpha=mixing_alpha
        )
        
        # Validate
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device,
            is_training=False
        )
        
        # Test
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, optimizer, device,
            is_training=False
        )
        
        # Clean train evaluation (if provided)
        if train_clean_loader is not None:
            train_clean_loss, train_clean_acc = run_epoch(
                model, train_clean_loader, criterion, optimizer, device,
                is_training=False
            )
        else:
            train_clean_loss, train_clean_acc = train_loss, train_acc
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['train_clean_loss'].append(train_clean_loss)
        history['train_clean_acc'].append(train_clean_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Detailed logging with progress indicators
        train_val_gap = train_clean_acc - val_acc
        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train: acc={train_acc:.4f} loss={train_loss:.4f} (clean: {train_clean_acc:.4f}) | "
              f"Val: acc={val_acc:.4f} loss={val_loss:.4f} | "
              f"Gap: {train_val_gap:+.4f} | "
              f"Time: {epoch_time:.1f}s", end="")
        
        # Mark best epoch
        if epoch + 1 == best_epoch:
            print(" ⭐ BEST", end="")
        print()
        
        # Periodic checkpoint logging
        if (epoch + 1) % 10 == 0 or epoch + 1 == num_epochs:
            gpu_info = f" | {print_gpu_memory_usage()}" if torch.cuda.is_available() else ""
            print(f"   └─ [Checkpoint] Best Val: {best_val_acc:.4f} at epoch {best_epoch}{gpu_info}")
    
    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    if torch.cuda.is_available():
        print(f"   Peak GPU Memory: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    
    # Save model checkpoint
    checkpoint_path = os.path.join(Config.RESULTS_DIR, f'{exp_name}_model.pth')
    torch.save({
        'model_state_dict': best_model_state,
        'exp_name': exp_name,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'config': config_schedule,
        'history': history
    }, checkpoint_path)
    print(f"   💾 Saved model: {checkpoint_path}")
    
    return history, best_model_state


# ============================================================================
# Experiment Running Functions
# ============================================================================

def run_single_experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                          device, config_schedule,
                          aug_type='crop_flip', dropout=0.5,
                          cutout_p=0.5, cutout_length=16,
                          mixing_aug=None, mixing_alpha=None,
                          seed=42,
                          evaluate_clean_train=False,
                          exp_name='experiment'):
    """
    Run a single experiment with given configuration.
    
    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        X_test, y_test: test data
        device: torch device
        config_schedule: training schedule config
        aug_type: augmentation type
        dropout: dropout rate
        cutout_p: cutout probability
        cutout_length: cutout patch size
        mixing_aug: mixing augmentation type
        mixing_alpha: mixing alpha parameter
        seed: random seed
        evaluate_clean_train: whether to evaluate on clean train data
        exp_name: experiment name for logging
    
    Returns:
        results: dictionary with final metrics
        history: training history
    """
    set_seed(seed)
    
    # Create transforms
    train_transform = get_augmentation_transform(
        aug_type=aug_type,
        cutout_p=cutout_p,
        cutout_length=cutout_length
    )
    test_transform = get_test_transform()
    
    # Create datasets
    train_dataset = AugmentedTensorDataset(X_train, y_train, transform=train_transform)
    val_dataset = AugmentedTensorDataset(X_val, y_val, transform=test_transform)
    test_dataset = AugmentedTensorDataset(X_test, y_test, transform=test_transform)
    
    # Create loaders (pin_memory for faster GPU transfer)
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=pin_mem)
    
    # Create clean train loader if needed
    train_clean_loader = None
    if evaluate_clean_train:
        train_clean_dataset = AugmentedTensorDataset(X_train, y_train, transform=test_transform)
        train_clean_loader = DataLoader(train_clean_dataset, batch_size=Config.BATCH_SIZE,
                                       shuffle=False, num_workers=0, pin_memory=pin_mem)
    
    # Create model
    model = MiniVGG(num_classes=Config.NUM_CLASSES, dropout=dropout).to(device)
    num_params = count_parameters(model)
    print(f"   Model parameters: {num_params:,}")
    print(f"   Device: {next(model.parameters()).device}")
    if torch.cuda.is_available():
        print(f"   {print_gpu_memory_usage()}")
    
    # Train
    run_start = time.time()
    history, best_model_state = train_model(
        model, train_loader, val_loader, test_loader,
        device, config_schedule,
        mixing_aug=mixing_aug,
        mixing_alpha=mixing_alpha,
        train_clean_loader=train_clean_loader,
        exp_name=exp_name
    )
    training_time = time.time() - run_start
    
    # Load best model and evaluate
    print(f"\n📊 Evaluating on test set with best model...")
    model.load_state_dict(best_model_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = None  # Not needed for evaluation
    
    val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, is_training=False)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, is_training=False)
    
    if train_clean_loader is not None:
        train_clean_loss, train_clean_acc = run_epoch(model, train_clean_loader, criterion, optimizer, device, is_training=False)
    else:
        train_clean_loss = history['train_loss'][-1]
        train_clean_acc = history['train_acc'][-1]
    
    print(f"   Test accuracy: {test_acc:.4f}")
    print(f"   Test loss: {test_loss:.4f}")
    print(f"   Train-Val gap: {train_clean_acc - val_acc:+.4f}")
    print(f"   Total training time: {training_time/60:.1f} minutes")
    
    results = {
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_clean_acc': train_clean_acc,
        'train_val_gap': train_clean_acc - val_acc,
        'training_time': training_time,
        'seed': seed,
        'aug_type': aug_type,
        'dropout': dropout,
        'cutout_p': cutout_p,
        'cutout_length': cutout_length,
        'mixing_aug': mixing_aug,
        'mixing_alpha': mixing_alpha
    }
    
    return results, history


def run_multi_seed_experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                               device, config_schedule,
                               aug_type='crop_flip', dropout=0.5,
                               cutout_p=0.5, cutout_length=16,
                               mixing_aug=None, mixing_alpha=None,
                               seeds=None,
                               evaluate_clean_train=False,
                               exp_name_prefix='experiment'):
    """
    Run experiment across multiple seeds and aggregate results.
    
    Returns:
        agg_results: aggregated statistics
        all_results: list of individual run results
        all_histories: list of individual run histories
    """
    if seeds is None:
        seeds = Config.SEEDS
    
    all_results = []
    all_histories = []
    
    print(f"\n{'='*70}")
    print(f"Multi-seed experiment: {aug_type}")
    if mixing_aug:
        print(f"Mixing augmentation: {mixing_aug} (alpha={mixing_alpha})")
    print(f"Seeds: {seeds}")
    print(f"{'='*70}")
    
    for i, seed in enumerate(seeds):
        print(f"\n[Seed {i+1}/{len(seeds)}] Running with seed={seed}")
        
        exp_name = f"{exp_name_prefix}_seed{seed}"
        results, history = run_single_experiment(
            X_train, y_train, X_val, y_val, X_test, y_test,
            device, config_schedule,
            aug_type=aug_type,
            dropout=dropout,
            cutout_p=cutout_p,
            cutout_length=cutout_length,
            mixing_aug=mixing_aug,
            mixing_alpha=mixing_alpha,
            seed=seed,
            evaluate_clean_train=evaluate_clean_train,
            exp_name=exp_name
        )
        
        all_results.append(results)
        all_histories.append(history)
        
        print(f"  ✓ Seed {seed}: Test Acc: {results['test_acc']:.4f}, "
              f"Train-Val Gap: {results['train_val_gap']:+.4f}")
    
    # Aggregate
    test_accs = [r['test_acc'] for r in all_results]
    val_accs = [r['val_acc'] for r in all_results]
    gaps = [r['train_val_gap'] for r in all_results]
    
    agg_results = {
        'aug_type': aug_type,
        'test_acc_mean': np.mean(test_accs),
        'test_acc_std': np.std(test_accs),
        'val_acc_mean': np.mean(val_accs),
        'val_acc_std': np.std(val_accs),
        'gap_mean': np.mean(gaps),
        'gap_std': np.std(gaps),
        'test_accs': test_accs,
        'dropout': dropout,
        'cutout_p': cutout_p,
        'cutout_length': cutout_length,
        'mixing_aug': mixing_aug,
        'mixing_alpha': mixing_alpha
    }
    
    print(f"\nAggregated Results:")
    print(f"  Test Acc: {agg_results['test_acc_mean']:.4f} ± {agg_results['test_acc_std']:.4f}")
    print(f"  Val Acc:  {agg_results['val_acc_mean']:.4f} ± {agg_results['val_acc_std']:.4f}")
    print(f"  Gap:      {agg_results['gap_mean']:+.4f} ± {agg_results['gap_std']:.4f}")
    
    return agg_results, all_results, all_histories


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_multi_seed_comparison(all_agg_results, save_path=None):
    """
    Plot comparison of different augmentation strategies with error bars.
    
    Args:
        all_agg_results: list of aggregated results dictionaries
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    aug_types = [r['aug_type'] for r in all_agg_results]
    test_means = [r['test_acc_mean'] for r in all_agg_results]
    test_stds = [r['test_acc_std'] for r in all_agg_results]
    gap_means = [r['gap_mean'] for r in all_agg_results]
    gap_stds = [r['gap_std'] for r in all_agg_results]
    
    x = np.arange(len(aug_types))
    
    # Test accuracy with color gradient
    norm = plt.Normalize(vmin=min(test_means), vmax=max(test_means))
    colors = plt.cm.RdYlGn(norm(test_means))
    
    bars = axes[0].bar(x, test_means, yerr=test_stds, capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, test_means, test_stds)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                    f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    axes[0].set_xlabel('Augmentation Strategy', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Test Accuracy Comparison (Mean ± Std)', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(aug_types, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Train-val gap
    gap_colors = ['green' if g < 0.01 else 'orange' if g < 0.05 else 'red' for g in gap_means]
    axes[1].bar(x, gap_means, yerr=gap_stds, capsize=5, alpha=0.7, color=gap_colors, edgecolor='black', linewidth=1.5)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Balance')
    axes[1].set_xlabel('Augmentation Strategy', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Train-Val Gap', fontsize=12, fontweight='bold')
    axes[1].set_title('Generalization Gap (Mean ± Std)', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(aug_types, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend(fontsize=9)
    
    plt.suptitle('Advanced Augmentation Study - Final Results', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
    
    plt.show()


def plot_training_curves(all_histories, strategy_names, save_path=None):
    """
    Plot training curves for multiple strategies.
    
    Args:
        all_histories: list of history dictionaries
        strategy_names: list of strategy names
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot accuracy curves
    for history, name in zip(all_histories, strategy_names):
        epochs_range = range(1, len(history['train_clean_acc']) + 1)
        axes[0].plot(epochs_range, history['train_clean_acc'], '-', 
                    label=f'Train ({name})', linewidth=2.5, alpha=0.7)
        axes[0].plot(epochs_range, history['val_acc'], '--', 
                    label=f'Val ({name})', linewidth=2.5, alpha=0.8)
    
    axes[0].set_xlabel("Epoch", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Accuracy", fontsize=13, fontweight='bold')
    axes[0].set_title("Training vs Validation Accuracy", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Plot loss curves
    for history, name in zip(all_histories, strategy_names):
        epochs_range = range(1, len(history['train_clean_loss']) + 1)
        axes[1].plot(epochs_range, history['train_clean_loss'], '-', 
                    label=f'Train ({name})', linewidth=2.5, alpha=0.7)
        axes[1].plot(epochs_range, history['val_loss'], '--', 
                    label=f'Val ({name})', linewidth=2.5, alpha=0.8)
    
    axes[1].set_xlabel("Epoch", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Loss", fontsize=13, fontweight='bold')
    axes[1].set_title("Training vs Validation Loss", fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Impact of Advanced Augmentation on Training Dynamics", 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
    
    plt.show()


# ============================================================================
# Helper Functions
# ============================================================================

def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


# ============================================================================
# Main Experiment Pipeline
# ============================================================================

def main():
    """Main experiment pipeline."""
    
    print("="*70)
    print("CNN Advanced Augmentation Experiments")
    print("="*70)
    
    # Setup
    device = get_device()
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Load data
    print("\nLoading CIFAR-10 data...")
    X_train_full, y_train_full, X_test, y_test = load_cifar10_data(Config.DATA_PATH)
    X_train, y_train, X_val, y_val = split_train_val(X_train_full, y_train_full, val_size=10000, seed=42)
    
    # ========================================================================
    # STEP 0: Multi-seed baseline anchoring
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 0: Multi-seed baseline anchoring")
    print("="*70)
    
    baseline_agg, baseline_results, baseline_histories = run_multi_seed_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test,
        device, Config.SCHEDULE_QUICK,
        aug_type='crop_flip',
        dropout=0.5,
        seeds=Config.SEEDS,
        evaluate_clean_train=True,
        exp_name_prefix='step0_baseline'
    )
    
    # Save baseline results
    baseline_save = {
        'agg': baseline_agg,
        'results': baseline_results
    }
    with open(os.path.join(Config.RESULTS_DIR, 'step0_baseline.json'), 'w') as f:
        json.dump(convert_to_json_serializable(baseline_save), f, indent=2)
    
    print(f"\n✓ Step 0 complete. Baseline: {baseline_agg['test_acc_mean']:.4f} ± {baseline_agg['test_acc_std']:.4f}")
    
    # ========================================================================
    # STEP 1: Clean train accuracy evaluation (already done in Step 0)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Clean train evaluation (integrated in Step 0)")
    print("="*70)
    print(f"Train-Val Gap: {baseline_agg['gap_mean']:+.4f} ± {baseline_agg['gap_std']:.4f}")
    print("✓ Step 1 complete (train_clean_acc tracked throughout)")
    
    # ========================================================================
    # STEP 2: Cutout tuning grid
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Cutout tuning grid")
    print("="*70)
    
    cutout_configs = [
        {'p': 0.1, 'length': 8, 'dropout': 0.3},
        {'p': 0.1, 'length': 12, 'dropout': 0.3},
        {'p': 0.1, 'length': 16, 'dropout': 0.3},
        {'p': 0.25, 'length': 8, 'dropout': 0.3},
        {'p': 0.25, 'length': 12, 'dropout': 0.3},
        {'p': 0.25, 'length': 16, 'dropout': 0.3},
    ]
    
    cutout_results = []
    cutout_all_results = {}  # Store individual results for best config
    
    for i, cfg in enumerate(cutout_configs, 1):
        print(f"\n[Config {i}/{len(cutout_configs)}] Tuning: p={cfg['p']}, length={cfg['length']}, dropout={cfg['dropout']}")
        
        exp_name = f"step2_cutout_p{cfg['p']}_len{cfg['length']}_drop{cfg['dropout']}"
        agg, results, histories = run_multi_seed_experiment(
            X_train, y_train, X_val, y_val, X_test, y_test,
            device, Config.SCHEDULE_QUICK,
            aug_type='cutout',
            dropout=cfg['dropout'],
            cutout_p=cfg['p'],
            cutout_length=cfg['length'],
            seeds=Config.SEEDS,
            evaluate_clean_train=True,
            exp_name_prefix=exp_name
        )
        
        cutout_results.append(agg)
        # Store the full results for this config
        config_key = f"p{cfg['p']}_len{cfg['length']}"
        cutout_all_results[config_key] = {'agg': agg, 'results': results, 'histories': histories}
    
    # Find best cutout config
    best_cutout = max(cutout_results, key=lambda x: x['test_acc_mean'])
    best_cutout_key = f"p{best_cutout['cutout_p']}_len{best_cutout['cutout_length']}"
    best_cutout_results = cutout_all_results[best_cutout_key]['results']
    print(f"\n✓ Step 2 complete. Best cutout: p={best_cutout['cutout_p']}, "
          f"length={best_cutout['cutout_length']}, "
          f"Test Acc: {best_cutout['test_acc_mean']:.4f} ± {best_cutout['test_acc_std']:.4f}")
    
    # Save cutout results
    with open(os.path.join(Config.RESULTS_DIR, 'step2_cutout_tuning.json'), 'w') as f:
        json.dump(convert_to_json_serializable(cutout_results), f, indent=2)
    
    # ========================================================================
    # STEP 3: RandAugment
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: RandAugment")
    print("="*70)
    
    randaug_agg, randaug_results, randaug_histories = run_multi_seed_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test,
        device, Config.SCHEDULE_QUICK,
        aug_type='randaugment',
        dropout=0.5,
        seeds=Config.SEEDS,
        evaluate_clean_train=True,
        exp_name_prefix='step3_randaugment'
    )
    
    print(f"\n✓ Step 3 complete. RandAugment: {randaug_agg['test_acc_mean']:.4f} ± {randaug_agg['test_acc_std']:.4f}")
    
    # ========================================================================
    # STEP 4: MixUp and CutMix
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: MixUp and CutMix")
    print("="*70)
    
    # MixUp
    print("\nTesting MixUp (alpha=0.2)...")
    mixup_agg, mixup_results, mixup_histories = run_multi_seed_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test,
        device, Config.SCHEDULE_QUICK,
        aug_type='crop_flip',
        dropout=0.5,
        mixing_aug='mixup',
        mixing_alpha=0.2,
        seeds=Config.SEEDS,
        evaluate_clean_train=True,
        exp_name_prefix='step4_mixup'
    )
    
    # CutMix
    print("\nTesting CutMix (alpha=1.0)...")
    cutmix_agg, cutmix_results, cutmix_histories = run_multi_seed_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test,
        device, Config.SCHEDULE_QUICK,
        aug_type='crop_flip',
        dropout=0.5,
        mixing_aug='cutmix',
        mixing_alpha=1.0,
        seeds=Config.SEEDS,
        evaluate_clean_train=True,
        exp_name_prefix='step4_cutmix'
    )
    
    print(f"\n✓ Step 4 complete.")
    print(f"  MixUp:  {mixup_agg['test_acc_mean']:.4f} ± {mixup_agg['test_acc_std']:.4f}")
    print(f"  CutMix: {cutmix_agg['test_acc_mean']:.4f} ± {cutmix_agg['test_acc_std']:.4f}")
    
    # ========================================================================
    # Compare all strategies
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    all_strategies = [
        baseline_agg,
        best_cutout,
        randaug_agg,
        mixup_agg,
        cutmix_agg
    ]
    
    # Rename for clarity
    baseline_agg['aug_type'] = 'baseline_crop_flip'
    best_cutout['aug_type'] = f"cutout_tuned"
    mixup_agg['aug_type'] = 'mixup'
    cutmix_agg['aug_type'] = 'cutmix'
    
    for strategy in all_strategies:
        print(f"{strategy['aug_type']:20s}: "
              f"Test={strategy['test_acc_mean']:.4f}±{strategy['test_acc_std']:.4f}, "
              f"Gap={strategy['gap_mean']:+.4f}±{strategy['gap_std']:.4f}")
    
    # Find best overall
    best_strategy = max(all_strategies, key=lambda x: x['test_acc_mean'])
    print(f"\n🏆 Best strategy: {best_strategy['aug_type']} "
          f"(Test: {best_strategy['test_acc_mean']:.4f} ± {best_strategy['test_acc_std']:.4f})")
    
    # Save summary
    summary = {
        'baseline': baseline_agg,
        'best_cutout': best_cutout,
        'randaugment': randaug_agg,
        'mixup': mixup_agg,
        'cutmix': cutmix_agg,
        'best_overall': best_strategy
    }
    
    with open(os.path.join(Config.RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(convert_to_json_serializable(summary), f, indent=2)
    
    # ========================================================================
    # Visualizations
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n📊 Creating comparison plots...")
    plot_multi_seed_comparison(
        all_strategies,
        save_path=os.path.join(Config.RESULTS_DIR, 'comparison.png')
    )
    
    # Training curves for best strategies
    print("\n📊 Creating training curves for top strategies...")
    selected_histories = [baseline_histories[0], randaug_histories[0], mixup_histories[0]]
    selected_names = ['baseline_crop_flip', 'randaugment', 'mixup']
    plot_training_curves(
        selected_histories,
        selected_names,
        save_path=os.path.join(Config.RESULTS_DIR, 'training_curves.png')
    )
    
    # ========================================================================
    # Comprehensive summary table
    # ========================================================================
    print("\n" + "="*70)
    print("COMPREHENSIVE SUMMARY TABLE")
    print("="*70)
    
    print(f"\n{'Strategy':<20} {'Test Acc':<12} {'Val Acc':<12} {'Gap':<12} {'Time (min)':<12}")
    print("-" * 70)
    
    all_strategies_with_time = [
        (baseline_agg, baseline_results),
        (best_cutout, best_cutout_results),
        (randaug_agg, randaug_results),
        (mixup_agg, mixup_results),
        (cutmix_agg, cutmix_results)
    ]
    
    for strategy, results_list in all_strategies_with_time:
        avg_time = np.mean([r['training_time']/60 for r in results_list]) if results_list else 0
        print(f"{strategy['aug_type']:<20} "
              f"{strategy['test_acc_mean']:.4f}±{strategy['test_acc_std']:.4f}  "
              f"{strategy['val_acc_mean']:.4f}±{strategy['val_acc_std']:.4f}  "
              f"{strategy['gap_mean']:+.4f}±{strategy['gap_std']:.4f}  "
              f"{avg_time:<12.1f}")
    
    # ========================================================================
    # KEY INSIGHTS
    # ========================================================================
    print("\n" + "="*70)
    print("                        🎓 KEY INSIGHTS")
    print("="*70)
    
    print("\n📈 Findings from Advanced Augmentation Study:")
    print("\n   1. Multi-seed Validation:")
    print(f"      • Baseline variance: ±{baseline_agg['test_acc_std']:.4f} ({baseline_agg['test_acc_std']*100:.2f}%)")
    print("      • Multiple seeds essential for reliable comparison")
    
    print("\n   2. Clean Train Evaluation:")
    print(f"      • Baseline gap: {baseline_agg['gap_mean']:+.4f} (true train-val difference)")
    print("      • Negative gaps indicate strong regularization/underfitting")
    
    print("\n   3. Augmentation Effectiveness:")
    best_improvement = (best_strategy['test_acc_mean'] - baseline_agg['test_acc_mean']) * 100
    print(f"      • Best strategy improved baseline by {best_improvement:+.2f}%")
    print(f"      • Winner: {best_strategy['aug_type']}")
    
    print("\n   4. Practical Recommendations:")
    if best_strategy['aug_type'] == 'baseline_crop_flip':
        print("      • Simple crop+flip remains competitive")
        print("      • Advanced augmentations need careful tuning")
    elif 'cutout' in best_strategy['aug_type']:
        print(f"      • Cutout works best with reduced dropout")
        print(f"      • Optimal params: p={best_strategy.get('cutout_p', 'N/A')}, length={best_strategy.get('cutout_length', 'N/A')}")
    elif best_strategy['aug_type'] in ['mixup', 'cutmix']:
        print(f"      • Mixing augmentations effective for this dataset")
        print(f"      • Alpha parameter: {best_strategy.get('mixing_alpha', 'N/A')}")
    elif best_strategy['aug_type'] == 'randaugment':
        print("      • RandAugment provides strong automated augmentation")
        print("      • Reduces need for manual augmentation tuning")
    
    print("\n   5. Training Insights:")
    print("      • Clean train accuracy crucial for measuring true generalization")
    print("      • Extended training (200 epochs) may benefit strong augmentation")
    print("      • Balance augmentation strength with other regularization (dropout)")
    
    print("\n" + "="*70)
    print("✓ All experiments complete!")
    print(f"✓ Results saved to: {Config.RESULTS_DIR}")
    print(f"✓ Model checkpoints: {len(Config.SEEDS) * 5} models saved")
    if torch.cuda.is_available():
        print(f"✓ GPU Training Complete: {torch.cuda.get_device_name(0)}")
        print(f"✓ Total GPU Memory Used: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    else:
        print("✓ CPU Training Complete")
    print("="*70)


if __name__ == '__main__':
    main()
