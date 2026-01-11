import pickle
import matplotlib.pyplot as plt

filename = 'C:\\Users\\user\\OneDrive - TechnoVal\\Desktop\\Scripts\\ML\\cv-transition-lab\\data\\cifar-10-batches-py\\data_batch_1'

def load_batch(filename):
    """Load a single batch file from CIFAR-10"""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

# Load a training batch
batch1 = load_batch(filename)

# The batch is a dictionary with keys:
# - b'batch_label': name of the batch
# - b'labels': list of label integers (0-9)
# - b'data': numpy array of uint8s, shape (10000, 3072)
# - b'filenames': list of filenames

# To inspect the data:

print(type(batch1))
print(batch1.keys())
print(type(batch1[b'data']), batch1[b'data'].shape, batch1[b'data'].dtype)

# Extract images and labels
# images = batch1[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
# labels = batch1[b'labels'
images = batch1[b'data'].reshape(10000,3,32,32).transpose(0,2,3,1) #reshaping and transposing to get (num_samples, height, width, channels) (10000,32,32,3)
labels = batch1[b'labels'] #list of labels (10000,)


# Quick sanity visualization (To confirm no"inversion" / channel mixups)
idx = 1680
plt.imshow(images[idx])
plt.title(f"Label: {labels[idx]}")
plt.axis('off')
plt.show()