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
    losses = []
    X, Y = X_tr_emb, Y_tr
    for i in tqdm(range(1000)):
        X, Y, rm_idx, loss = condense_once(X, Y, X_tr_emb, Y_tr, 0.1,True)
        print("iteration ", i, " cur size ", len(Y), 'loss ',loss)
        losses.append(loss)
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
        data_tier_path = 'data_sub/' + name + '_tiers_loss.p'
        pickle.dump(losses,open(data_tier_path,'wb'))
        print("saved . . .", data_tier_path)

def kmeans(X_tr_emb,Y_tr,name,size = 100):
    import time
    import pickle
    from tqdm import tqdm
    from subset_selection.rec_annealing import k_means_idx

    #print("loading them pickle . . . ")
    #data_embed_path = 'data_embed/mnist_dim32.p'
    #X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
    # X_tr_emb, Y_tr = X_tr_emb[:1000],Y_tr[:1000]
    print("loaded ")
    ans = []
    #size = 100
    fract = 1.2
    while (size < 15000):
        print(size)
        ans.append(k_means_idx(X_tr_emb, int(size)))
        size = size * fract
        data_tier_path = 'data_sub/'+name+'_kmeans.p'
        pickle.dump(np.array(ans), open(data_tier_path, "wb"))
        print("saved . . .", data_tier_path)
        # print(ans)

def flat_anneal(X_tr_emb,Y_tr,idx_condensor,name,size = 100):
    import pickle
    from subset_selection.rec_annealing import anneal_optimize
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
        pickle.dump(cond_anneal, open('data_sub/'+name+'_tiers_anneal.p', 'wb'))
        size = int(size * frac)
        print('saved_anneal', size)



def kmeans_anneal(X,Y,idx_condensor,name):
    import time
    import pickle
    import numpy as np
    from subset_selection.rec_annealing import anneal_optimize


    cond_anneal = []

    for index in idx_condensor:
        cond_anneal.append(anneal_optimize(index, X, Y))
        pickle.dump(cond_anneal, open('data_sub/'+name+'_kmeans_anneal.p', 'wb'))
        print('saved_kmeans_anneal', len(index))


def random_anneal(X,Y,name,size=100,frac=1.1,tl=60,trial=1):
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
        pickle.dump(cond_anneal, open('data_sub/'+name+'_random_anneal.p', 'wb'))
        size = int(size * frac)
        print('saved_random_anneal', size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("type")
    args = parser.parse_args()
    import pickle
    X,Y = pickle.load(open('data_embed/relation_dim32.p','rb'))
    name = 'relation'
    print('Let the games begin:',args.type)

    if args.type == 'tiers':
        condensor(X,Y,name)
        data_tier_path = 'data_sub/' + name + '_tiers.p'
        index = pickle.load(open(data_tier_path, "rb"))
        flat_anneal(X,Y,index,name)
    if args.type == 'kmeans':
        kmeans(X,Y,name)
        data_tier_path = 'data_sub/' + name + '_kmeans.p'
        index = pickle.load(open(data_tier_path, "rb"))
        kmeans_anneal(X, Y, index, name)
    if args.type == 'random':
        random_anneal(X,Y,name)
