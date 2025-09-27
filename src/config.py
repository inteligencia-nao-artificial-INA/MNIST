configs = {
    'mnist': {
        # interesting index: [546, 94, 708, 233, 244, 846, 181]
        'images': "dataset/mnist/train-images.idx3-ubyte",
        'labels': "dataset/mnist/train-labels.idx1-ubyte",
        'model': "model/mnist/nn_parameters.npz",
    },

    'fashion-mnist': {
        # label_class = {0:'T-shirt', 1:'Trouser', 2: 'Pullover', 3: 'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}
        'images': "dataset/fashion-mnist/train-images-idx3-ubyte",
        'labels': "dataset/fashion-mnist/train-labels-idx1-ubyte",
        'model': "model/fashion-mnist/nn_parameters.npz",
    }
}