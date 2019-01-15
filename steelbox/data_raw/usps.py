import numpy as np
def scale(img):
    img = np.reshape(img, (16, 16))
    img = cv2.resize(img, (28, 28))
    img = np.reshape(img, (28*28,))
    return img

def gen_data(data_dir):
    import pickle
    path = data_dir + "/usps.p"
    return pickle.load(open(path, 'rb'))

def save_data(data_dir):
    path = data_dir + "/usps.h5"
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

    X_tr = np.array([scale(x) for x in X_tr])
    X_te = np.array([scale(x) for x in X_te])

    import pickle

    pickle.dump((X_tr, y_tr, X_te, y_te), open('usps.p', 'wb'))
    print ('data dumpt')


if __name__ == '__main__':
    import h5py
    import cv2

    save_data('.')
    X_tr, Y_tr, X_t, Y_t = gen_data('.') 
    print ('data loaded')
    print (X_tr.shape)

