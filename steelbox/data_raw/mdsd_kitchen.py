import numpy as np


def gen_data(path):
    data = np.load(path + "/mdsd/kitchen/data.npz")
    indices = np.random.RandomState(1234).permutation(data['bow'].shape[0])
    train_indices, test_indices = indices[:1600], indices[1600:]
    return data['bow'][train_indices], data['labels'][train_indices], data['bow'][test_indices], data['labels'][test_indices]


if __name__ == "__main__":
    tr_img, tr_lab, t_img, t_lab = gen_data(".")
    print (len(tr_img), len(tr_lab), len(t_img), len(t_lab))
    print (tr_img.shape)
    print (np.max(tr_img))
