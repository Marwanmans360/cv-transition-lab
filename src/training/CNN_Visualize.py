"""
Visualization script for trained CNN models.
Loads saved model weights and generates visualizations without retraining.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import visualization and model functions from CNN_Ablation
import sys
sys.path.insert(0, os.path.dirname(__file__))
from CNN_Ablation import (
    set_seed, get_device, load_cifar10_from_folder, 
    cifar10_flat_to_hwc_uint8,
    visualize_layer_filters, visualize_activation_maps, 
    visualize_learned_representations, MiniVGG, initialize_weights
)


def load_model(model_path, device, activation_type='relu', norm_type='batch', dropout_p=0.5):
    """Load model weights from saved checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = MiniVGG(dropout_p=dropout_p, norm_type=norm_type, activation_type=activation_type).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded model from: {model_path}")
    print(f"   Activation: {checkpoint['activation_type'].upper()}")
    print(f"   Best Val Acc: {checkpoint['best_val_acc']:.3f}")
    print(f"   Test Acc: {checkpoint['test_acc']:.3f}")
    
    return model, checkpoint


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

    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    # 1) Load CIFAR
    (X_train_flat, y_train), (X_test_flat, y_test) = load_cifar10_from_folder(cifar_folder)

    # 2) Convert to HWC uint8
    X_train_original = cifar10_flat_to_hwc_uint8(X_train_flat)
    X_test_original = cifar10_flat_to_hwc_uint8(X_test_flat)

    # 3) Compute mean/std
    X_train_01 = X_train_original.astype(np.float32) / 255.0
    mean = X_train_01.mean(axis=(0, 1, 2))
    std  = X_train_01.std(axis=(0, 1, 2))

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

    # Find all saved model files
    model_files = [f for f in os.listdir('.') if f.startswith('model_weights_act_') and f.endswith('.pth')]
    
    if not model_files:
        print("❌ No saved model weights found!")
        print("   Run CNN_Ablation.py first to train and save models.")
        return
    
    print(f"\n📦 Found {len(model_files)} saved model(s):")
    for f in sorted(model_files):
        print(f"   - {f}")
    
    # Visualize each model
    for model_file in sorted(model_files):
        print(f"\n{'='*60}")
        print(f"Visualizing: {model_file}")
        print(f"{'='*60}")
        
        # Load model
        model, checkpoint = load_model(model_file, device)
        activation_type = checkpoint['activation_type']
        
        # Visualize filters (pre-training - random init)
        # Note: We can't show true pre-training since we only saved the trained weights
        # But we can demonstrate by initializing a fresh model
        print("\n=== LEARNED FILTER VISUALIZATION ===")
        print("📊 Visualizing learned filters...")
        visualize_layer_filters(model, 'block1', 1, number_to_show=16, stage="post", activation_type=activation_type)
        visualize_layer_filters(model, 'block2', 2, number_to_show=16, stage="post", activation_type=activation_type)
        visualize_layer_filters(model, 'block3', 3, number_to_show=16, stage="post", activation_type=activation_type)
        
        # Visualize activation maps
        print("\n=== ACTIVATION MAP VISUALIZATION ===")
        print("📊 Visualizing activation maps...")
        visualize_activation_maps(model, val_loader, device, num_images=3, stage="post", activation_type=activation_type)
        
        # Visualize learned representations (t-SNE)
        print("\n=== LEARNED REPRESENTATIONS (t-SNE) ===")
        print("📊 Visualizing t-SNE representations...")
        visualize_learned_representations(model, val_loader, device, num_samples=10, stage="post", activation_type=activation_type)
    
    print(f"\n{'='*60}")
    print("✅ Visualization complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
