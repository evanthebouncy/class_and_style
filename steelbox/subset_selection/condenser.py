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

# condense out the redundant vectors
def condense_once(X, Y, X_orig, Y_orig):
    W_one = np.ones(len(Y_orig))

    A = kneighbors_graph(X, 1, mode='connectivity', n_jobs=4)
    idx1s, idx2s = A.nonzero()

    dependency = []

    # idx1 closest neighbor is idx2
    for idx1, idx2 in zip(idx1s, idx2s):
        # idx1 would not be affected if it were to depend on idx2
        if Y[idx1] == Y[idx2]:
            # find out if other points of same label depends on idx1
            dependency.append((idx1, np.sum(A[:,idx1])))

    dependency = sorted(dependency, key=lambda x:x[1])
    to_remove = [x[0] for x in dependency if x[1] == 0]
    if len(to_remove) == 0:
        to_remove = [x[0] for x in dependency[:int(len(Y) * 0.1)]]


    to_remove = sorted(list(to_remove))
    X_sub, Y_sub = np.delete(X, to_remove, 0), np.delete(Y, to_remove) 
    
    score_after = score_subset(X_sub, Y_sub, X_orig, Y_orig, W_one, one_near=True)
    print (" score ", score_after, " size ", len(Y_sub))
    return X_sub, Y_sub

# condense out the redundant vectors
def condense_once_2(X, Y, X_orig, Y_orig):
    if len(Y) == len(Y_orig):
        return condense_once(X, Y, X_orig, Y_orig)
    W_one = np.ones(len(Y_orig))

    # closest maps X_orig to X (may be identity map in case elements in both)
    closest, _ = pairwise_distances_argmin_min(X_orig, X)
    # A maps X to X in such a way that is not self-maping
    A = kneighbors_graph(X, 1, mode='connectivity', n_jobs=4)

    count = dict()
    for idx in range(X.shape[0]):
        count[idx] = 0.0
    for idx_start, idx in enumerate(closest):
        if idx_start != idx:
            count[idx] += 1.0 if Y_orig[idx_start] == Y[idx] else -1.0
        else:
            idx_end = np.argmax(A[idx])
            count[idx_end] += 1.0 if Y_orig[idx_start] == Y[idx_end] else -1.0

    count = sorted(list(count.items()), key=lambda x:x[1])
    to_remove = [x[0] for x in count[:int(len(Y) * 0.1)]]

    X_sub, Y_sub = np.delete(X, to_remove, 0), np.delete(Y, to_remove) 
    
    score_after = score_subset(X_sub, Y_sub, X_orig, Y_orig, W_one, one_near=True)
    print (" score ", score_after, " size ", len(Y_sub))
    return X_sub, Y_sub

def condense_once_3(X, Y, X_orig, Y_orig):
    tree = cKDTree(X)
    _, top2_idx = tree.query(X_orig, 2)

    def get_remove_cost(idx_rep):
        nearests = np.where(top2_idx[:,0] == idx_rep)
        nearests = nearests[0]

        existing_acc = 0
        removed_acc = 0
        for idx_villager in nearests:
            if Y_orig[idx_villager] == Y[idx_rep]:
                existing_acc += 1

            other_rep = top2_idx[idx_villager][1]
            if Y_orig[idx_villager] == Y[other_rep]:
                removed_acc += 1

        return existing_acc - removed_acc

    remove_costs = sorted([(get_remove_cost(idx),idx) for idx in range(X.shape[0])])

    throw_amount = max(1, int(0.01 * len(Y)))
    idx_to_keep = [x[1] for x in remove_costs][throw_amount:]
    
    X_sub, Y_sub = X[idx_to_keep,:], Y[idx_to_keep]
    W_one = np.ones(len(Y_orig))
    score_after = score_subset(X_sub, Y_sub, X_orig, Y_orig, W_one, one_near=True)
    print (" score ", score_after, " size ", len(Y_sub))
    return X_sub, Y_sub

def condense_once_4(X, Y, X_orig, Y_orig, throw_frac=0.1):
    tree = cKDTree(X)
    _, top2_idx = tree.query(X_orig, 2)

    rep_buddy = {}

    def get_remove_cost(idx_rep):
        nearests = np.where(top2_idx[:,0] == idx_rep)
        nearests = nearests[0]
        
        rep_buddy[idx_rep] = set()

        existing_acc = 0
        removed_acc = 0
        for idx_villager in nearests:
            if Y_orig[idx_villager] == Y[idx_rep]:
                existing_acc += 1

            other_rep = top2_idx[idx_villager][1]
            if Y_orig[idx_villager] == Y[other_rep]:
                removed_acc += 1

            # the safe removal of idx_rep is now dependent on other_rep as a buddy
            if Y[idx_rep] == Y_orig[idx_villager] == Y[other_rep]:
                rep_buddy[idx_rep].add(other_rep)

        return existing_acc - removed_acc

    remove_costs = sorted([(get_remove_cost(idx),idx) for idx in range(X.shape[0])])

    throw_amt = int(max(1, throw_frac * len(Y)))
    ban_list = set()
    to_remove = []
    for i in range(throw_amt):
        throw_idx = remove_costs[i][1]
        if throw_idx in ban_list:
            continue
        else:
            to_remove.append(throw_idx)
            ban_list.update(rep_buddy[throw_idx])

    X_sub, Y_sub = np.delete(X, to_remove, 0), np.delete(Y, to_remove) 
    
    W_one = np.ones(len(Y_orig))
    score_after = score_subset(X_sub, Y_sub, X_orig, Y_orig, W_one, one_near=True)
    print (" score ", score_after, " size ", len(Y_sub))
    return X_sub, Y_sub


if __name__ == '__main__':

    def test1():

        from data_raw.artificial import gen_data
        X, Y, X_t, Y_t = gen_data(2000)
        A = kneighbors_graph(X, 3, mode='connectivity', include_self=True)
        A = A.toarray()
        print (A)

    def test2():
        import time
        import pickle
        print ("loading them pickle . . . ")
        data_embed_path = 'data_embed/mnist_dim32.p'
        X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
        N = X_tr_emb.shape[0]

        n_jobs = 4
        start_time = time.time()
        print ("sweating them neighbors finding ")
        A = kneighbors_graph(X_tr_emb, 2, mode='connectivity', n_jobs=n_jobs)
        print (A)
        print (" time ", time.time() - start_time, " njobs ", n_jobs)

    def test3():
        import time
        import pickle
        print ("loading them pickle . . . ")
        data_embed_path = 'data_embed/mnist_dim2.p'
        X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
        print ("loaded ")
        N = X_tr_emb.shape[0]

        X, Y = X_tr_emb, Y_tr
        for i in range(50):
            print ("iteration ", i)
            X, Y = condense_once(X, Y, X_tr_emb, Y_tr)

    def test4():
        import time
        import pickle
        print ("loading them pickle . . . ")
        data_embed_path = 'data_embed/mnist_dim32.p'
        X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
        print ("loaded ")
        N = X_tr_emb.shape[0]

        X, Y = X_tr_emb, Y_tr
        for i in range(70):
            print ("iteration ", i)
            X, Y = condense_once_2(X, Y, X_tr_emb, Y_tr)

    def test5():
        import time
        import pickle
        print ("loading them pickle . . . ")
        data_embed_path = 'data_embed/mnist_dim32.p'
        X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
        print ("loaded ")
        N = X_tr_emb.shape[0]

        X, Y = X_tr_emb, Y_tr
        for i in range(700):
            print ("iteration ", i)
            X, Y = condense_once_4(X, Y, X_tr_emb, Y_tr)

    test5()

