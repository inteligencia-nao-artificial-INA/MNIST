import train
import model
import config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(prog='visualize.py', description='Visualize neural network activations of trained models', add_help=True)
parser.add_argument('-dataset', dest='dataset', metavar='mnist', help='dataset name', type=str, required=True)
args = parser.parse_args()

dataset = config.configs[args.dataset]

X_train, Y_train, X_dev, Y_dev = train.load_and_prepare_data(dataset['images'], dataset['labels'])
W1, b1, W2, b2 = model.load_params(dataset['model'])

# plot node activation in layer 1 (acho que para todas iterações)
# after computing the forward propagation
Z1, A1, Z2, A2 = model.forward_prop(W1, b1, W2, b2, X_dev)

# A1 - layer 1: activation after first layer (after ReLU)
# A2 - layer 2: activation after softmax 

# visualize activations for the first hidden layer
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
sns.heatmap(A1, cmap='viridis')  # use the activations from Layer 1
plt.title('Activations of Layer 1')
plt.subplot(1,2,2)
sns.heatmap(A2, cmap='viridis')  # use the activations from Layer 2
plt.title('Activations of Layer 2')
plt.show()

# visualize forward prop activation 
Z1, A1, Z2, A2 = model.forward_prop(W1, b1, W2, b2, X_dev)
predictions = model.get_predictions(A2)
accuracy = model.get_accuracy(predictions, Y_dev)

# visualize activations for a correct prediction
correct_idx = np.where(predictions == Y_dev)[0][0]
plt.figure(figsize=(10, 5))
sns.heatmap(A1[:, correct_idx].reshape(1, -1), cmap='coolwarm', annot=True)
plt.title('Activations of Layer 1 for a Correct Prediction')
plt.show()

# features detected by neurons
plt.figure(figsize=(10, 5))
sns.heatmap(W1, cmap='RdYlBu', annot=False)
plt.title('Weights of Layer 1 Neurons')
plt.show()