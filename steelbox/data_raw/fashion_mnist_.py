from mnist import MNIST
import numpy as np

import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

def gen_data_h(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def gen_data(path):
    the_path = path + "/fashion-mnist-master/data/fashion/"
    tr_img, tr_lab = gen_data_h(the_path)
    t_img, t_lab = gen_data_h(the_path, 't10k')
    tr_img = tr_img / 255
    t_img = t_img / 255
    return tr_img, tr_lab, t_img, t_lab

if __name__ == "__main__":
    tr_img, tr_lab, t_img, t_lab = gen_data(".")
    print (len(tr_img), len(tr_lab), len(t_img), len(t_lab))
    print (tr_img.shape)
    print (np.max(tr_img))

