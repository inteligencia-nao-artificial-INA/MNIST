import load
import numpy as np
from matplotlib import pyplot as plt

def init_params():
    """
    Initializes the parameters (weights and biases) for a two-layer neural network.

    Returns:
        W1 (np.ndarray): Weight matrix for the first layer (shape: 10x784).
        b1 (np.ndarray): Bias vector for the first layer (shape: 10x1).
        W2 (np.ndarray): Weight matrix for the second layer (shape: 10x10).
        b2 (np.ndarray): Bias vector for the second layer (shape: 10x1).
    """
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    """
    Applies the ReLU (Rectified Linear Unit) activation function element-wise.

    Args:
        Z (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Matrix with ReLU applied (element-wise maximum of 0 and Z).
    """
    return np.maximum(Z, 0)

def softmax(Z):
    """
    Applies the softmax activation function to the input, converting it into a probability distribution.

    Args:
        Z (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Probability distribution after applying the softmax function.
    """
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    """
    Performs the forward propagation step in the neural network.

    Args:
        W1 (np.ndarray): Weight matrix for the first layer.
        b1 (np.ndarray): Bias vector for the first layer.
        W2 (np.ndarray): Weight matrix for the second layer.
        b2 (np.ndarray): Bias vector for the second layer.
        X (np.ndarray): Input data matrix.

    Returns:
        Z1 (np.ndarray): Linear transformation at the first layer.
        A1 (np.ndarray): Activation at the first layer.
        Z2 (np.ndarray): Linear transformation at the second layer.
        A2 (np.ndarray): Activation (output probabilities) at the second layer.
    """
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    """
    Computes the derivative of the ReLU activation function.

    Args:
        Z (np.ndarray): Input matrix.

    Returns:
        np.ndarray: A matrix where each element is 1 if Z > 0, otherwise 0.
    """
    return Z > 0

def one_hot(Y):
    """
    Converts class labels into one-hot encoding.

    Args:
        Y (np.ndarray): Array of class labels.

    Returns:
        np.ndarray: One-hot encoded matrix of class labels.
    """
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m):
    """
    Performs the backward propagation step for the neural network.

    Args:
        Z1, A1, Z2, A2: Activations and linear transformations from forward propagation.
        W1, W2: Weight matrices for the first and second layers.
        X (np.ndarray): Input data matrix.
        Y (np.ndarray): Ground truth labels.
        m (int): Number of examples in the dataset.

    Returns:
        dW1, db1, dW2, db2: Gradients of weights and biases for layers 1 and 2.
    """
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Updates the parameters of the neural network using the computed gradients.

    Args:
        W1 (np.ndarray): Weight matrix for the first layer.
        b1 (np.ndarray): Bias vector for the first layer.
        W2 (np.ndarray): Weight matrix for the second layer.
        b2 (np.ndarray): Bias vector for the second layer.
        dW1 (np.ndarray): Gradient of W1.
        db1 (np.ndarray): Gradient of b1.
        dW2 (np.ndarray): Gradient of W2.
        db2 (np.ndarray): Gradient of b2.
        alpha (float): Learning rate.

    Returns:
        W1 (np.ndarray): Updated weight matrix for the first layer.
        b1 (np.ndarray): Updated bias vector for the first layer.
        W2 (np.ndarray): Updated weight matrix for the second layer.
        b2 (np.ndarray): Updated bias vector for the second layer.
    """
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    """
    Retrieves the predicted class labels from the output layer activations.

    Args:
        A2 (np.ndarray): Output layer activations (probabilities).

    Returns:
        np.ndarray: Predicted class labels.
    """
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    """
    Computes the accuracy of the model by comparing predictions with the true labels.

    Args:
        predictions (np.ndarray): Predicted class labels.
        Y (np.ndarray): True class labels.

    Returns:
        str: Accuracy as a percentage string formatted to four decimal places.
    """
    return f'Accuracy: {np.sum(predictions == Y) / Y.size * 100:.4f}%'

def gradient_descent(X, Y, m, alpha, iterations):
    """
    Optimizes the neural network parameters using gradient descent.

    Args:
        X (np.ndarray): Input data matrix.
        Y (np.ndarray): True class labels.
        m (int): Number of examples.
        alpha (float): Learning rate.
        iterations (int): Number of iterations.

    Returns:
        W1, b1, W2, b2: Optimized weight and bias matrices for the neural network.
    """
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            print("Iteration: ", i)
            print(get_accuracy(predictions, Y))
    
    return W1, b1, W2, b2

def save_params(W1, b1, W2, b2, filename):
    """
    Saves the model parameters to a file.

    Args:
        W1 (np.ndarray): Weight matrix for the first layer.
        b1 (np.ndarray): Bias vector for the first layer.
        W2 (np.ndarray): Weight matrix for the second layer.
        b2 (np.ndarray): Bias vector for the second layer.
        filename (str): The file path to save the parameters to.
    """
    np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"Parameters saved to {filename}")

def load_params(filename):
    """
    Loads the model parameters from a file.

    Args:
        filename (str): The file path to load the parameters from.

    Returns:
        W1 (np.ndarray): Weight matrix for the first layer.
        b1 (np.ndarray): Bias vector for the first layer.
        W2 (np.ndarray): Weight matrix for the second layer.
        b2 (np.ndarray): Bias vector for the second layer.
    """
    params = np.load(filename)
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    print(f"Parameters loaded from {filename}")
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    """
    Makes predictions for a given input dataset.

    Args:
        X (np.ndarray): Input data matrix.
        W1 (np.ndarray): Weight matrix for the first layer.
        b1 (np.ndarray): Bias vector for the first layer.
        W2 (np.ndarray): Weight matrix for the second layer.
        b2 (np.ndarray): Bias vector for the second layer.

    Returns:
        np.ndarray: Predicted class labels.
    """
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, X, Y, W1, b1, W2, b2):
    """Tests a prediction by displaying the predicted label and the actual label for a given index.

    Args:
        index (int): Index of the sample to test.
        X (np.ndarray): Input data matrix.
        Y (np.ndarray): Ground truth labels.
        W1 (np.ndarray): Weight matrix for the first layer.
        b1 (np.ndarray): Bias vector for the first layer.
        W2 (np.ndarray): Weight matrix for the second layer.
        b2 (np.ndarray): Bias vector for the second layer.

    Returns:
        None: Displays the prediction and actual label.
    """
    current_image = X[:, index, None]
    prediction = make_predictions(X[:, index, None], W1, b1, W2, b2)
    label = Y[index]
    print("Index: ", index)
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def load_and_prepare_data(images_path, labels_path, split_ratio=0.1):
    """
    Load dataset, normalize, and split into training and validation sets.
    
    Args:
        images_path (str): Path to the images file.
        labels_path (str): Path to the labels file.
        split_ratio (float): Proportion of the data to be used as validation set.
        
    Returns:
        X_train (np.ndarray): Training data.
        Y_train (np.ndarray): Training labels.
        X_eval (np.ndarray): Validation data.
        Y_eval (np.ndarray): Validation labels.
    """

    images = load.load_mnist_images(images_path)          # load images
    labels = load.load_mnist_labels(labels_path)          # load labels
    
    data = np.column_stack((labels, images))              # combine labels and images
    np.random.shuffle(data)
    
    m = data.shape[0]                                     
    eval_size = int(m * split_ratio)                      # split into training and dev sets
    
    data_eval = data[:eval_size].T
    Y_dev = data_eval[0]
    X_dev = data_eval[1:] / 255.                          # normalize
    
    data_train = data[eval_size:].T
    Y_train = data_train[0]
    X_train = data_train[1:] / 255.                       # normalize
    
    return X_train, Y_train, X_dev, Y_dev

def train_neural_network(X_train, Y_train, learning_rate, iterations, model_save_path=None):
    """
    Train the neural network using gradient descent.
    
    Args:
        X_train (np.ndarray): Training data.
        Y_train (np.ndarray): Training labels.
        learning_rate (float): Learning rate for gradient descent.
        iterations (int): Number of iterations for training.
        model_save_path (str): Path to save the trained model parameters.
    
    Returns:
        W1, b1, W2, b2 (np.ndarray): Trained weights and biases.
    """
    m = X_train.shape[1]
    
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, m, learning_rate, iterations)
    
    if model_save_path:
        np.savez(model_save_path, W1=W1, b1=b1, W2=W2, b2=b2)
        print(f"Model parameters saved to {model_save_path}")
    
    return W1, b1, W2, b2
