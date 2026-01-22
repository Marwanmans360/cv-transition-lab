import numpy as np
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
print("Testing different k values")
print("="*50)

k_values = [1, 3, 5, 7, 9, 15]
l1_accuracies = []
l2_accuracies = []

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