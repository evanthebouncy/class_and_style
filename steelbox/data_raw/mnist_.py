from mnist import MNIST
import numpy as np

import random
from scipy import ndarray
# import skimage as sk
# from skimage import transform
# from skimage import util

def gen_data(data_dir):
    the_path = data_dir + "/mnist_data/"
    #print ("loading mnist from ", the_path)
    mndata = MNIST(the_path)
    mndata.gz = True
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    train_images = np.array(train_images) / 255 
    test_images = np.array(test_images) / 255 
    train_labels, test_labels = np.array(train_labels), np.array(test_labels)
    return train_images, train_labels, test_images, test_labels

def gen_data_skew(data_dir, skew_lab=0):
    X_tr, Y_tr, X_t, Y_t = gen_data(data_dir)
    
    skew_idxs = [np.where(Y_tr == lab)[0][::(10 if lab != skew_lab else 1)] for lab in range(10)]
    skew_idxs = np.concatenate(skew_idxs, axis=0)
    
    return X_tr[skew_idxs], Y_tr[skew_idxs], X_t, Y_t


def data_augmentation(X):
    
    # def random_transform(image_array):
    #     scale_x, scale_y = np.random.normal(1, 0.1), np.random.normal(1, 0.1)
    #     rotation = np.random.normal(0, 0.2)
    #     shear = np.random.normal(0, 0.1)
    #     translation_x, translation_y = int(np.random.normal(0, 2)), int(np.random.normal(0, 2))
    #     a_xform = transform.AffineTransform(scale=(scale_x, scale_y), 
    #                                       rotation=rotation, 
    #                                       shear=shear,
    #                                       translation=(translation_x, translation_y))
    #     transformed = transform.warp(image_array, a_xform)
    #     return transformed

    def random_noise(image_array):
        # add random noise to the image
        return image_array + np.random.normal(0, 1, size=image_array.shape)

    ret_X = []
    for x in X:
        x = np.reshape(x, [28,28])
        x = random_noise(x)
        # x = random_transform(random_noise(x))
        ret_X.append(np.reshape(x, 28*28))
    return np.array(ret_X)



if __name__ == "__main__":
    tr_img, tr_lab, t_img, t_lab = gen_data(".")
    print (len(tr_img), len(tr_lab), len(t_img), len(t_lab))
    print (tr_img.shape)

    print (data_augmentation(tr_img[:40]).shape)

    X_tr_skew, Y_tr_skew, X_t, Y_t = gen_data_skew(".")
    import pdb; pdb.set_trace()
