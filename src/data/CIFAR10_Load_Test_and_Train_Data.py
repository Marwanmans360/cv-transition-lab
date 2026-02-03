import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


path = 'C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\data\\cifar-10-batches-py\\'

def load_batch(directory):
    """Load a single batch file from CIFAR-10"""
    with open(directory, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def load_cifar10(dir):

    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        batch_file = os.path.join(dir, f'data_batch_{i}')
        batch = load_batch(batch_file)
        X_train_list.append(batch[b'data'])
        y_train_list.append(batch[b'labels'])


    X_train = np.concatenate(X_train_list, axis = 0)
    y_train = np.concatenate(y_train_list,axis = 0)

    test_path = os.path.join(dir, "test_batch")
    test_batch = load_batch(test_path)
    X_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":   
    (X_train, y_train), (X_test, y_test) = load_cifar10(path)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

