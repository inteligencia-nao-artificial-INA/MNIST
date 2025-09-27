import model
import config
import argparse

parser = argparse.ArgumentParser(prog='train.py', description='Train neural network using different datasets, learning rates, and iterations', add_help=True)
parser.add_argument('-dataset', dest='dataset', metavar='mnist', help='dataset name', type=str, required=True)
parser.add_argument('-lr', '--learning_rate', dest='learning_rate', metavar=0.1, help='training learning rate', type=float, required=True)
parser.add_argument('-i', '--iterations', dest='iterations', metavar=1000, help='training iterations', type=int, required=True)
args = parser.parse_args()

dataset = config.configs[args.dataset]
learning_rate = args.learning_rate
iterations = args.iterations

X_train, Y_train, X_dev, Y_dev = model.load_and_prepare_data(dataset['images'], dataset['labels'])
W1, b1, W2, b2 = model.train_neural_network(X_train, Y_train, learning_rate, iterations, dataset['model'])
