import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
sys.path.append('C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab')
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10
path = 'C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\data\\cifar-10-batches-py\\'
(X_train, y_train), (X_test, y_test) = load_cifar10(path)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# The labes represent the following classes:
# 0: airplane
# 1: automobile
# 2: bird
# 3: cat
# 4: deer
# 5: dog
# 6: frog
# 7: horse
# 8: ship
# 9: truck


class NearestNeighborClassifier:
    def __init__(self, k=1, distance_metric='L2'):
        self.k = k
        self.distance_metric = distance_metric

    def train_model_one(self, X, y): #Memorizing Training Data
        """
        Store the training data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Training labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the labels for the input data using the k-NN algorithm.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Input data to classify.

        Returns:
        y_pred : array-like, shape (n_samples,)
            Predicted labels for the input data.
        """
        num_test = X.shape[0]
        # make sure that the output type matches the input type 
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

        for i in range(num_test):
            # Calculate distances based on metric
            if self.distance_metric == 'L1':
                distances = np.sum(np.abs(self.X_train - X[i,:]), axis=1)
            else:  # L2
                distances = np.sqrt(np.sum((self.X_train - X[i,:])**2, axis=1))
            
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_lables = self.y_train[k_nearest_indices]
            y_pred[i] = np.bincount(k_nearest_lables).argmax()
        
        return y_pred


# First lets look at first 10 test examples and their ground truth labels
num_test_samples = 500
# print("Ground truth labels for first 10 test samples: ", y_test[:num_test_samples])

# Create and train the classifier
# knn_classifier = NearestNeighborClassifier()
# knn_classifier.train_model_one(X_train, y_train)
# Predict labels for the first 10 test samples
# predicted_labels = knn_classifier.predict(X_test[:num_test_samples])
# print("Predicted labels for first 10 test samples: ", predicted_labels)

# Explore different k values
print("\n" + "="*50)
print("Testing different k values (on 500 test samples)")
print("="*50)

k_values = [1, 3, 5, 7, 9, 15]
l1_accuracies = []
l2_accuracies = []
num_test_samples = 100

for k in k_values:
    # Test L1
    classifier_l1 = NearestNeighborClassifier(k=k, distance_metric='L1')
    classifier_l1.train_model_one(X_train, y_train)
    predictions_l1 = classifier_l1.predict(X_test[:num_test_samples])
    accuracy_l1 = np.mean(predictions_l1 == y_test[:num_test_samples])
    l1_accuracies.append(accuracy_l1)
    
    # Test L2
    classifier_l2 = NearestNeighborClassifier(k=k, distance_metric='L2')
    classifier_l2.train_model_one(X_train, y_train)
    predictions_l2 = classifier_l2.predict(X_test[:num_test_samples])
    accuracy_l2 = np.mean(predictions_l2 == y_test[:num_test_samples])
    l2_accuracies.append(accuracy_l2)
    
    print(f"k={k:2d} | L1: {accuracy_l1:.4f} | L2: {accuracy_l2:.4f}")

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(k_values, l1_accuracies, marker='o', label='L1 Distance', linewidth=2)
plt.plot(k_values, l2_accuracies, marker='s', label='L2 Distance', linewidth=2)
plt.xlabel('k Value', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('k-NN Accuracy: L1 vs L2 Distance Metrics', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.show()

# ============================================================
# VORONOI-LIKE DECISION BOUNDARY VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("GENERATING VORONOI DECISION BOUNDARIES (2D PCA)")
print("="*60)

# Reduce to 2D using PCA for visualization
print("\nReducing to 2D using PCA...")
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

print(f"Original shape: {X_train.shape}")
print(f"Reduced shape: {X_train_2d.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Create mesh grid for decision boundary
h = 0.1  # step size in mesh
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Test different k values for Voronoi visualization
k_voronoi = [1, 3, 5, 15]
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.ravel()

# Use smaller training subset for faster computation
train_subset_size = 1000
train_idx = np.random.choice(len(X_train_2d), size=train_subset_size, replace=False)
X_train_2d_subset = X_train_2d[train_idx]
y_train_subset = y_train[train_idx]

# Color map for 10 classes
colors = plt.cm.tab10(np.linspace(0, 1, 10))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

for idx, k in enumerate(k_voronoi):
    print(f"\nProcessing k={k}...")
    
    # Create classifier with subset
    classifier = NearestNeighborClassifier(k=k, distance_metric='L2')
    classifier.train_model_one(X_train_2d_subset, y_train_subset)
    
    # Create a finer mesh for visualization
    h = 0.5  # step size in mesh - increase for faster computation
    x_min, x_max = X_train_2d_subset[:, 0].min() - 1, X_train_2d_subset[:, 0].max() + 1
    y_min, y_max = X_train_2d_subset[:, 1].min() - 1, X_train_2d_subset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    print(f"  Grid size: {xx.shape[0]} x {xx.shape[1]} = {xx.shape[0]*xx.shape[1]} points")
    
    # Vectorize prediction for speed
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.predict(grid_points).reshape(xx.shape)
    
    print(f"  Predictions done, plotting...")
    
    # Plot decision boundary
    ax = axes[idx]
    
    # Plot mesh (decision regions) with contourf
    contour = ax.contourf(xx, yy, Z, levels=np.arange(-0.5, 10.5, 1), 
                          colors=colors, alpha=0.4)
    
    # Plot training points
    for class_id in range(10):
        mask = y_train_subset == class_id
        ax.scatter(X_train_2d_subset[mask, 0], X_train_2d_subset[mask, 1], 
                  c=[colors[class_id]], label=class_names[class_id], 
                  edgecolors='black', linewidth=0.5, s=30, alpha=0.8)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title(f'k-NN Decision Boundaries (k={k})', fontsize=12, fontweight='bold')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    # Test accuracy with this k on subset
    predictions = classifier.predict(X_train_2d_subset)
    accuracy = np.mean(predictions == y_train_subset)
    ax.text(0.05, 0.95, f'Train Accuracy: {accuracy:.2%}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
          ncol=10, fontsize=10, frameon=True)

plt.suptitle('k-NN Voronoi Decision Boundaries (2D PCA Projection)', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\reports\\knn_voronoi_boundaries.png', 
           dpi=150, bbox_inches='tight')
print("\n✓ Voronoi diagram saved to: reports/knn_voronoi_boundaries.png")
plt.show()

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)
print("""
Observations from Voronoi diagrams:
- k=1: Very complex, jagged boundaries (overfitting to individual points)
- k=3,5: Smoother boundaries, better generalization
- k=15: Very smooth, may underfit (too much averaging)

The boundaries show which regions belong to which class based on
nearest neighbors. Darker colors indicate confidence in the boundary.
""")
