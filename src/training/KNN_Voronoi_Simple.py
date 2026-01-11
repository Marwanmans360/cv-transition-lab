"""
Simple k-NN Voronoi Visualization on CIFAR-10
Reduced training set and coarse grid for fast visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
sys.path.append('C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab')
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10

print("="*60)
print("K-NN VORONOI VISUALIZATION")
print("="*60)

# Load data
print("\nLoading CIFAR-10...")
path = 'C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\data\\cifar-10-batches-py\\'
(X_train, y_train), (X_test, y_test) = load_cifar10(path)

# Simple k-NN classifier
class KNN:
    def __init__(self, k=1):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X - x)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y[k_indices]
            y_pred.append(np.bincount(k_labels).argmax())
        return np.array(y_pred)

# Reduce to 2D
print("Applying PCA...")
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_train)
print(f"Explained variance: {sum(pca.explained_variance_ratio_):.1%}")

# Use tiny subset
n_samples = 200
idx = np.random.choice(len(X_2d), n_samples, replace=False)
X_subset = X_2d[idx]
y_subset = y_train[idx]
print(f"Using {n_samples} samples")

# Visualize
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
colors = plt.cm.tab10(np.linspace(0, 1, 10))

k_values = [1, 3, 5, 15]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, k in enumerate(k_values):
    print(f"\nk={k}:")
    ax = axes[idx]
    
    # Train
    knn = KNN(k=k)
    knn.fit(X_subset, y_subset)
    
    # Create coarse grid
    margin = 10
    x_min, x_max = X_subset[:, 0].min() - margin, X_subset[:, 0].max() + margin
    y_min, y_max = X_subset[:, 1].min() - margin, X_subset[:, 1].max() + margin
    
    step = 5  # very coarse for speed
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    print(f"  Grid: {xx.shape[0]} x {xx.shape[1]} = {xx.size} points")
    
    # Predict
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points).reshape(xx.shape)
    print(f"  Predicting...")
    
    # Plot regions
    print(f"  Plotting...")
    ax.contourf(xx, yy, Z, levels=np.arange(-0.5, 10.5, 1), 
                colors=colors, alpha=0.4)
    
    # Plot points
    for c in range(10):
        mask = y_subset == c
        ax.scatter(X_subset[mask, 0], X_subset[mask, 1],
                  c=[colors[c]], s=80, edgecolors='black', linewidth=1.5,
                  label=class_names[c])
    
    # Accuracy
    acc = np.mean(knn.predict(X_subset) == y_subset)
    
    ax.set_title(f'k={k} (Train Acc: {acc:.1%})', fontsize=13, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.grid(True, alpha=0.2)

# Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03),
          ncol=10, fontsize=9)

plt.suptitle('k-NN Decision Boundaries (Voronoi Diagram)\nCIFAR-10 with 2D PCA', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

# Save
save_path = 'C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\reports\\knn_voronoi_simple.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n{'='*60}")
print(f"✓ Saved to: {save_path}")
print(f"{'='*60}")

plt.show()

print("""
INTERPRETATION:
- k=1: Sharp, jagged boundaries (each point defines its own region)
- k=3,5: Smoother transitions between classes  
- k=15: Very smooth, generalized boundaries

The colored regions show which class k-NN predicts for any point in space.
""")
