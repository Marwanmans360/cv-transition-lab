import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab')
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10
path = 'C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\data\\cifar-10-batches-py\\'
(X_train, y_train), (X_test, y_test) = load_cifar10(path)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")

# Reshape the data to (N, 32, 32, 3)
X_train = X_train.reshape(-1, 32, 32, 3)
X_test = X_test.reshape(-1, 32, 32, 3)

print(f"Reshaped X_train: {X_train.shape}")
print(f"Reshaped X_test: {X_test.shape}")

# ----------------------------------------
# Data Prepocessing: Per Channeel Mean Subtraction and Std. Dev. Normalization
# ----------------------------------------
mean = X_train.mean(axis=(0, 1, 2)) / 255.0  # Normalize to [0, 1] first
std = X_train.std(axis=(0, 1, 2)) / 255.0

print(f"Per-channel mean: {mean}")
print(f"Per-channel std: {std}")

# Normalize the data
X_train = (X_train / 255.0 - mean) / std
X_test = (X_test / 255.0 - mean) / std

# Function to print data values before and after normalization
def print_data_values(X_original, X_normalized, num_samples=5):
    for i in range(num_samples):
        print(f"Sample {i + 1}:")
        print("Before Normalization (First Pixel):", X_original[i, 0, 0])  # First pixel of the image
        print("After Normalization (First Pixel):", X_normalized[i, 0, 0])  # First pixel of the image
        print("-" * 50)

# Call the function to print data values
print_data_values(X_train * 255, X_train)  # Multiply X_train by 255 to get original scale

# Shuffle and split into training and validation sets

rng = np.random.default_rng(42)
idx = rng.permutation(X_train.shape[0])
N = 10000  # CV size
cv_idx = idx[:N]
tr_idx = idx[N:]  # remaining go to train

X_cv = X_train[cv_idx]
y_cv = y_train[cv_idx]
X_tr = X_train[tr_idx]
y_tr = y_train[tr_idx]

# Convert to PyTorch tensors
X_tr = torch.tensor(X_tr, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
y_tr = torch.tensor(y_tr, dtype=torch.long)
X_cv = torch.tensor(X_cv, dtype=torch.float32).permute(0, 3, 1, 2)
y_cv = torch.tensor(y_cv, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders for training, validation, and testing
train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_cv, y_cv), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)


# CNN Block 1 Conv(3 -> 32) -> ReLU -> Conv(32 -> 32) -> ReLU -> MaxPool
class FirstBlock(nn.Module):
    def __init__(self):
        super(FirstBlock, self).__init__()
        self.conv1 = nn.Conv2d(3 , 32, kernel_size=3, padding=1) # Conv Layer 1 (Channels, Filters, Kernel Size, Padding, Stride)
        self.relu1 = nn.ReLU()                     # ReLU Activation 1
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # Conv Layer 2
        self.relu2 = nn.ReLU()                     # ReLU Activation 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling

    def forward(self, x):
        x = self.relu1(self.conv1(x)) # Conv Layer 1 + ReLU
        x = self.relu2(self.conv2(x)) # Conv Layer 2 + ReLU
        x = self.pool(x)              # Max Pooling
        return x

# Invoke the FirstBlock to verify its functionality
first_block = FirstBlock()

# Test the block with a dummy input
dummy_input = torch.randn(64, 3, 32, 32)  # Batch of 64 images, 3 channels, 32x32
output = first_block(dummy_input)
print(f"Output shape after first block: {output.shape}")


# Define the second block of the mini-VGG
class SecondBlock(nn.Module):
    def __init__(self):
        super(SecondBlock, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Conv layer 1
        self.relu1 = nn.ReLU()  # ReLU activation
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Conv layer 2
        self.relu2 = nn.ReLU()  # ReLU activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max-pooling layer

    def forward(self, x):
        x = self.relu1(self.conv1(x))  # Conv1 -> ReLU
        x = self.relu2(self.conv2(x))  # Conv2 -> ReLU
        x = self.pool(x)  # Max-pooling
        return x

# Instantiate the second block
second_block = SecondBlock()

# Test the block with a dummy input
dummy_input = torch.randn(64, 32, 16, 16)  # Batch of 64 images, 32 channels, 16x16
output = second_block(dummy_input)
print(f"Output shape after second block: {output.shape}")

class ThirdBlock(nn.Module):
    def __init__(self):
        super(ThirdBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Conv layer 1
        self.relu1 = nn.ReLU()  # ReLU activation
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # Conv layer 2
        self.relu2 = nn.ReLU()  # ReLU activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max-pooling layer

    def forward(self, x):
        x = self.relu1(self.conv1(x))  # Conv1 -> ReLU
        x = self.relu2(self.conv2(x))  # Conv2 -> ReLU
        x = self.pool(x)  # Max-pooling
        return x

# Test the block with a dummy input
third_block = ThirdBlock()

dummy_input = torch.randn(64, 64, 8, 8)  # Batch of 64 images, 64 channels, 8x8
output = third_block(dummy_input)
print(f"Output shape after third block: {output.shape}")