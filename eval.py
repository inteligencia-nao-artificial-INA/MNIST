import numpy as np
from load import load_mnist_images
from load import load_mnist_labels
from train import load_params
from train import make_predictions
from train import get_accuracy
from train import test_prediction
import random

# eval mnist
#test_images_path = "dataset/mnist/t10k-images.idx3-ubyte"
#test_labels_path = "dataset/mnist/t10k-labels.idx1-ubyte"

# eval fashion-mnist
test_images_path = "dataset/fashion-mnist/t10k-images-idx3-ubyte"
test_labels_path = "dataset/fashion-mnist/t10k-labels-idx1-ubyte"

test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# add label to images array
data = np.column_stack((test_labels, test_images))
#print(data.shape)

m, n = data.shape

# splitting dataset in dev set (test set)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

# eval mnist
#W1, b1, W2, b2 = load_params('model/mnist/nn_parameters.npz')

# eval fashion-mnist
W1, b1, W2, b2 = load_params('model/fashion-mnist/nn_parameters.npz')

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(get_accuracy(dev_predictions, Y_dev))

random.seed(1337)
# fashion-mnist labels
# 0 T-shirt/top 1 Trouser 2 Pullover 3 Dress 4 Coat 5 Sandal 6 Shirt 7 Sneaker 8 Bag 9 Ankle boot
for i in range(10):
    a = random.randint(0,1000)    # 10k images in the fashion-mnist
    test_prediction(a, W1, b1, W2, b2)