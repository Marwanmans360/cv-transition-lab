import numpy as np

class NearestNeighborClassifier:
    def __init__(self):
        pass

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
        Predict the labels for the input data using the 1-NN algorithm.

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
            # find the nearest training example to the i-th test example
            # Using L1 distance (sum of absolute value difference)
            distances = np.sum(np.abs(self.X_train - X[i,:]), axis=1) #becauce you want to compute L1 difference between each pixel value
                                                                      # and sum them up for each training example for each row along coloumn axis=1
                                                                      
            min_index = np.argmin(distances) # get the index with smallest distance
            y_pred[i] = self.y_train[min_index] # predict the label of the nearest example
