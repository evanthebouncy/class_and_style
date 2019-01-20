
import pickle
data_embed_path = 'data_embed/mnist_dim2.p'
X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))

BND = 10000

X = X_tr_emb[:BND]
Y = Y_tr[:BND]

mnist_tiers = pickle.load(open("data_sub/mnist_tiers_2d.p", "rb"))

reps = mnist_tiers[:100]

X_centers = X_tr_emb[reps]

# begin the plotting
import matplotlib.pyplot as plt
import numpy as np
plt.figure(dpi=2400)
plt.xlim(-2, 2)
plt.ylim(-2, 2)


# plot the cluster assignments (lines)
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(X_centers)
voronoi_plot_2d(vor, show_points=True, show_vertices=False)

# plot the tsne embedding space
cl_colors = np.linspace(0, 1, len(set(Y_tr)))
x1 = [x[0] for x in X]
x2 = [x[1] for x in X]
colors = [cl_colors[lab] for lab in Y]
plt.scatter(x1, x2, c=colors, alpha=0.5, edgecolors='none')

plt.savefig('experiment_pictures/mnist_embed.png')
plt.clf()


# find out boundary representative sets
from data_raw.mnist_ import gen_data as mnist_gen_data
MNIST_X_tr, MNIST_Y_tr, MNIST_X_t, MNIST_Y_t = mnist_gen_data("./data_raw")
X_X = []
for r1 in reps:
    for r2 in reps:
        if r1 < r2:
            d_r1r2 = np.sum((X_tr_emb[r1] - X_tr_emb[r2]) ** 2)
            X_X.append((r1,r2,d_r1r2))

X_X = sorted(X_X, key=lambda x: x[2])

top_k = X_X[:50]

for idx, r1_r2_blah in enumerate(top_k):
    r1,r2,blah = r1_r2_blah
    print (" hey ", r1, r2, blah)
    confuse1 = np.reshape(MNIST_X_tr[r1], (28,28))
    confuse2 = np.reshape(MNIST_X_tr[r2], (28,28))
    lab1, lab2 = MNIST_Y_tr[r1], MNIST_Y_tr[r2]
    plt.imshow(confuse1, cmap='gray')
    plt.savefig('experiment_pictures/confuse/{}_{}.png'.format(idx, r1))
    plt.imshow(confuse2, cmap='gray')
    plt.savefig('experiment_pictures/confuse/{}_{}.png'.format(idx, r2))




