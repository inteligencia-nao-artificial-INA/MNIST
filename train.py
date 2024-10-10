from load import load_mnist_images
from load import load_mnist_labels
import numpy as np
from matplotlib import pyplot as plt

# eval mnist
#train_images_path = "dataset/mnist/train-images.idx3-ubyte"
#train_labels_path = "dataset/mnist/train-labels.idx1-ubyte"

# eval fashion-mnist
train_images_path = "dataset/fashion-mnist/train-images-idx3-ubyte"
train_labels_path = "dataset/fashion-mnist/train-labels-idx1-ubyte"

train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
#print(train_images.shape)
#print(train_labels.shape)

# add label to images array
data = np.column_stack((train_labels, train_images))
print(data.shape)

m, n = data.shape
#np.random.shuffle(data) # suffle bfore split in dev and training sets

#''' comment to use test data
# splitting dataset in dev and trainining sets
data_dev = data[0:1000].T
Y = data_dev[0]
X = data_dev[1:n]
X = X / 255.
#'''

''' comment to use train data
data_train = data[1000:m].T
Y = data_train[0]
X = data_train[1:n]
X = X / 255.
#_,m_train = X.shape
#print(Y)
'''

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    '''
    X if X > 0 
    0 if X <= 0
    '''
    return np.maximum(Z, 0)

def softmax(Z):
    '''
    convert network output into probability distribution
    '''
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return f'Accuracy: {np.sum(predictions == Y) / Y.size * 100:.4f}%'

def gradient_descent(X, Y, alpha, iterations):
    '''
    optmization function
    '''
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
            #print(f"W1: {W1}, b1: {b1}, W2: {W2}, b2: {b2}")
    
    return W1, b1, W2, b2

# save params to file (using numpy.savez)
def save_params(W1, b1, W2, b2, filename):
    np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"Parameters saved to {filename}")

'''
# adjust learning rate (0.10)
W1, b1, W2, b2 = gradient_descent(X, Y, 0.10, 1000)        # run again to train W and b
#print(f"W1: {W1}, b1: {b1}, W2: {W2}, b2: {b2}")
#save_params(W1, b1, W2, b2, 'model/mnist/nn_parameters.npz')                # run again to update parameters
save_params(W1, b1, W2, b2, 'model/fashion-mnist/nn_parameters.npz') 
'''

def load_params(filename):
    params = np.load(filename)
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    print(f"Parameters loaded from {filename}")
    return W1, b1, W2, b2

'''
Iteration:  490
[9 7 6 ... 2 4 7] [9 9 6 ... 2 4 7]
0.8416949152542373
'''

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X[:, index, None]
    prediction = make_predictions(X[:, index, None], W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

'''
# make predictions 
W1, b1, W2, b2 = load_params('model/nn_parameters.npz')
test_prediction(1, W1, b1, W2, b2)
test_prediction(13, W1, b1, W2, b2)
test_prediction(56, W1, b1, W2, b2)
test_prediction(255, W1, b1, W2, b2)
'''