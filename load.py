import numpy as np
import struct

# carregar o imagens no formato ubyte 
def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        # ignorar o cabeçalho (16 bytes)
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # lendo o resto do arquivo (cada imagem é de tamanho rows x cols)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
    return images

# carregar os rótulos no formato ubyte 
def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        # ignorar o cabeçalho (8 bytes)
        magic, num_labels = struct.unpack(">II", f.read(8))
        # lendo o resto do arquivo (rótulos são uint8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels