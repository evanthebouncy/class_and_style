import pickle
import matplotlib.pyplot as plt
import numpy as np


# do a 2D visualization of the space of things
def plot2d(X, Y, name=None):
    cl_colors = np.linspace(0, 1, len(set(Y))) if (Y is not None) else ['blue']

    from sklearn.manifold import TSNE
    X_tsne = TSNE(n_components=2).fit_transform(X)
    x = [x[0] for x in X_tsne]
    y = [x[1] for x in X_tsne]

    colors = [cl_colors[lab] for lab in Y] if (Y is not None) else 'blue'
    plt.scatter(x, y, c=colors, alpha=0.5)
    name = name if name else ""
    plt.savefig('vis_'+name+'.png')
    plt.clf()

if __name__ == '__main__':
    MNIST_X, MNIST_Y = pickle.load(open("mnist_dim32.p", "rb"))
    rand_idx = np.random.choice(range(60000), 1000, replace=False)

    MNIST_X = MNIST_X[rand_idx]
    MNIST_Y = MNIST_Y[rand_idx]
    plot2d(MNIST_X, MNIST_Y, "mnist_32")

    JOIN_X, JOIN_Y = pickle.load(open("mnist_usps_dim32.p", "rb"))
    rand_idx = np.random.choice(range(60000+7291), 1000, replace=False)

    JOIN_X = JOIN_X[rand_idx]
    JOIN_Y = JOIN_Y[rand_idx]
    plot2d(JOIN_X, JOIN_Y, "join_32")
