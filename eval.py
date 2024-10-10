import model
import config
import random
import argparse

parser = argparse.ArgumentParser(prog='eval.py', description='Evaluate neural network models trained using different datasets, learning rates, and iterations', add_help=True)
parser.add_argument('-dataset', dest='dataset', metavar='mnist', help='dataset name', type=str, required=True)
parser.add_argument('-predictions', dest='predictions', metavar='10', help='how many predictions', type=int, required=True)
args = parser.parse_args()

dataset = config.configs[args.dataset]
predictions = args.predictions

X_train, Y_train, X_dev, Y_dev = model.load_and_prepare_data(dataset['images'], dataset['labels'])
W1, b1, W2, b2 = model.load_params(dataset['model'])

dev_predictions = model.make_predictions(X_dev, W1, b1, W2, b2)
print(model.get_accuracy(dev_predictions, Y_dev))

for i in range(predictions):
    a = random.randint(0,1000)    # 10k images in the fashion-mnist
    model.test_prediction(a, X_dev, Y_dev, W1, b1, W2, b2)
