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

def condense_once(X, Y, X_orig, Y_orig, throw_frac=0.01):
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

        # a small random amount to break ties
        smol_random = random.random() * 0.1
        
        return existing_acc - removed_acc + smol_random

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

    # recover the index in the original X where the things are removed
    X_rm, Y_rm = X[to_remove, :], Y[to_remove]
    rm_idx = recover_index(X_rm, X_orig)
    
    return X_sub, Y_sub, rm_idx


if __name__ == '__main__':

    def test5():
        import time
        import pickle
        from tqdm import tqdm

        print ("loading them pickle . . . ")
        data_embed_path = 'data_embed/mnist_dim32.p'
        X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
        print ("loaded ")
        N = X_tr_emb.shape[0]

        remove_orders = []

        X, Y = X_tr_emb, Y_tr
        for i in tqdm(range(700)):
            X, Y, rm_idx = condense_once(X, Y, X_tr_emb, Y_tr)
            print ("iteration ", i, " cur size ", len(Y))

            remove_orders.append(rm_idx)

            # saving stuff
            data_tier_path = 'data_sub/mnist_tiers.p'
            pickle.dump(remove_orders, open(data_tier_path, "wb"))
            print ("saved . . . ", data_tier_path)

    test5()

