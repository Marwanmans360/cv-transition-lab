import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
sys.path.append('C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab')
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10

print("Loading CIFAR-10 data...")
path = 'C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\data\\cifar-10-batches-py\\'
(X_train, y_train), (X_test, y_test) = load_cifar10(path)

class NearestNeighborClassifier:
    def __init__(self, k=1, distance_metric='L2'):
        self.k = k
        self.distance_metric = distance_metric

    def train_model_one(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

        for i in range(num_test):
            if self.distance_metric == 'L1':
                distances = np.sum(np.abs(self.X_train - X[i,:]), axis=1)
            else:  # L2
                distances = np.sqrt(np.sum((self.X_train - X[i,:])**2, axis=1))
            
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_lables = self.y_train[k_nearest_indices]
            y_pred[i] = np.bincount(k_nearest_lables).argmax()
        
        return y_pred

# ============================================================
# PCA REDUCTION
# ============================================================

print("\nApplying PCA to reduce to 2D...")
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)

print(f"Original shape: {X_train.shape}")
print(f"Reduced shape: {X_train_2d.shape}")
print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

# Use subset for faster training
train_subset_size = 100
train_idx = np.random.choice(len(X_train_2d), size=train_subset_size, replace=False)
X_train_subset = X_train_2d[train_idx]
y_train_subset = y_train[train_idx]

print(f"Using {train_subset_size} training samples for faster visualization")

# ============================================================
# VORONOI VISUALIZATION
# ============================================================

k_values = [1, 3, 5, 15]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
colors = plt.cm.tab10(np.linspace(0, 1, 10))

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.ravel()

for idx, k in enumerate(k_values):
    print(f"\nGenerating Voronoi for k={k}...")
    ax = axes[idx]
    
    # Train classifier
    classifier = NearestNeighborClassifier(k=k, distance_metric='L2')
    classifier.train_model_one(X_train_subset, y_train_subset)
    
    # Create mesh grid
    h = 2.0  # larger step size for reasonable grid
    x_min, x_max = X_train_subset[:, 0].min() - 5, X_train_subset[:, 0].max() + 5
    y_min, y_max = X_train_subset[:, 1].min() - 5, X_train_subset[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    print(f"  Grid size: {xx.shape}")
    
    # Predict on grid (vectorized)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.predict(grid_points).reshape(xx.shape)
    
    # Plot decision regions
    print(f"  Plotting decision regions...")
    contour = ax.contourf(xx, yy, Z, levels=np.arange(-0.5, 10.5, 1), 
                          colors=colors, alpha=0.5)
    
    # Plot training points
    print(f"  Plotting training points...")
    for class_id in range(10):
        mask = y_train_subset == class_id
        ax.scatter(X_train_subset[mask, 0], X_train_subset[mask, 1], 
                  c=[colors[class_id]], label=class_names[class_id], 
                  edgecolors='black', linewidth=1, s=50, alpha=0.9)
    
    # Compute accuracy on subset
    predictions = classifier.predict(X_train_subset)
    accuracy = np.mean(predictions == y_train_subset)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title(f'k-NN Decision Boundaries (k={k})\nTrain Accuracy: {accuracy:.2%}', 
                fontsize=12, fontweight='bold')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

print("\nAdding legend...")
# Add legend to figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), 
          ncol=10, fontsize=10, frameon=True)

plt.suptitle('k-NN Voronoi Decision Boundaries (2D PCA Projection of CIFAR-10)', 
            fontsize=14, fontweight='bold')
plt.tight_layout()

print("Saving figure...")
save_path = 'C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\reports\\knn_voronoi_boundaries.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Voronoi diagram saved to: {save_path}")

plt.show()

print("\n" + "="*60)
print("KEY OBSERVATIONS:")
print("="*60)
print("""
- k=1: Very complex, jagged decision boundaries
  → Each training point acts as its own region (overfitting)
  
- k=3,5: Smoother boundaries with clear regions
  → Good balance between fit and generalization
  
- k=15: Very smooth, large decision regions
  → May oversmooth and lose detail (underfitting)

The Voronoi diagram shows how the input space is partitioned
based on k-NN predictions. Colors represent class labels.
""")

print("\n✓ Visualization complete!")
