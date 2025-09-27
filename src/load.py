import struct
import numpy as np

def load_mnist_images(file_path):
    """
    Loads MNIST images from an ubyte format file.

    Args:
        file_path (str): Path to the ubyte format image file.

    Returns:
        np.ndarray: A NumPy array containing the MNIST images. The array shape is
                    (n_images, n_pixels), where each image is flattened into a
                    vector of size (rows * cols).
    """
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))                       # ignore the header (16 bytes)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)        # read the rest of the file (each image is rows x cols in size)
    return images

def load_mnist_labels(file_path):
    """
    Loads MNIST labels from an ubyte format file.

    Args:
        file_path (str): Path to the ubyte format label file.

    Returns:
        np.ndarray: A NumPy array containing the MNIST labels.
    """
    with open(file_path, 'rb') as f:
        
        magic, num_labels = struct.unpack(">II", f.read(8))                                      # ignore the header (8 bytes)
        labels = np.frombuffer(f.read(), dtype=np.uint8)                                         # read the rest of the file (labels are stored as uint8)
    return labels