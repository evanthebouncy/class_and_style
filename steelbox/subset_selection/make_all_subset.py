import numpy as np
import random
import time
import math
from copy import deepcopy
from sklearn.cluster import KMeans
from .knn import score_subset, update_weight
from sklearn.neighbors import kneighbors_graph
from collections import Counter

from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial import cKDTree
from .rec_annealing import recover_index
from .condense import condense_once
import argparse


def condensor(X_tr_emb,Y_tr,name,final_size=100):
    import time
    import pickle
    from tqdm import tqdm

    #print("loading them pickle . . . ")
    #data_embed_path = 'data_embed/mnist_dim32.p'
    #X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
    # X_tr_emb, Y_tr = X_tr_emb[:100],Y_tr[:100]
    #print("loaded ")
    N = X_tr_emb.shape[0]

    remove_orders = []

    X, Y = X_tr_emb, Y_tr
    for i in tqdm(range(1000)):
        X, Y, rm_idx = condense_once(X, Y, X_tr_emb, Y_tr, 0.1)
        print("iteration ", i, " cur size ", len(Y))

        remove_orders.append(rm_idx)
        if X.shape[0] < final_size:
            break
        index = np.arange(X_tr_emb.shape[0])
        remove_orders_h = np.hstack(remove_orders)
        # print(np.delete(index,remove_orders))
        index = np.delete(index, remove_orders_h)
        index = np.append(index, remove_orders_h[::-1])
        # print(index)
        data_tier_path = 'data_sub/' + name + '_tiers.p'
        pickle.dump(index, open(data_tier_path, "wb"))
        print("saved . . .", data_tier_path)

def kmeans(X_tr_emb,Y_tr,name,start_size = 100):
    import time
    import pickle
    from tqdm import tqdm

    #print("loading them pickle . . . ")
    #data_embed_path = 'data_embed/mnist_dim32.p'
    #X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
    # X_tr_emb, Y_tr = X_tr_emb[:1000],Y_tr[:1000]
    print("loaded ")
    ans = []
    #size = 100
    fract = 1.2
    while (size < X_tr_emb.shape[0]):
        print(size)
        ans.append(k_means_idx(X_tr_emb, int(size)))
        size = size * fract
        data_tier_path = 'data_sub/'+name+'_kmeans.p'
        pickle.dump(np.array(ans), open(data_tier_path, "wb"))
        print("saved . . .", data_tier_path)
        # print(ans)

def flat_anneal(X_tr_emb,Y_tr,idx_condensor,name,size = 100):
    import pickle
    #data_path = 'data_sub/mnist_tiers.p'
    #idx_condensor = index
    #print(idx_condensor.shape)
    X, Y = X_tr_emb, Y_tr

    #size = 100
    frac = 1.1
    cond_anneal = []

    while size < idx_condensor.shape[0]:
        index = idx_condensor[:size]
        tl = 60
        #if size == 100:
        #    tl = 300
        cond_anneal.append(anneal_optimize(index, X, Y, tl))
        pickle.dump(cond_anneal, open('data_sub/'+name+'tiers_anneal.p', 'wb'))
        size = int(size * frac)
        print('saved', size)



def kmeans_anneal(X,Y,idx_condensor,name):
    import time
    import pickle
    import numpy as np
    from subset_selection.rec_annealing import anneal_optimize


    cond_anneal = []

    for index in idx_condensor:
        cond_anneal.append(anneal_optimize(index, X, Y))
        pickle.dump(cond_anneal, open('data_sub/'+name+'kmeans_anneal.p', 'wb'))
        print('saved', len(index))


def random_anneal(X,Y,name,size=100,frac=1.1,tl=60,trial=4):
    import time
    import pickle
    import numpy as np
    from subset_selection.rec_annealing import anneal_optimize

    #X, Y = pickle.load(open('data_embed/mnist_dim32.p', 'rb'))


    cond_anneal = []

    while size < len(Y):

        for i in range(trial):
            index = np.random.choice(list(range(len(Y))), size, replace=False)
            cond_anneal.append(anneal_optimize(index, X, Y, tl))
        pickle.dump(cond_anneal, open('data_sub/'+name+'random_anneal.p', 'wb'))
        size = int(size * frac)
        print('saved', size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='subset generating parameters')
    parser.add_argument('--dataset_name', type=str, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

