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

if __name__ == '__main__':
    def condensor():
        import time
        import pickle
        from tqdm import tqdm

        print("loading them pickle . . . ")
        data_embed_path = 'data_embed/mnist_dim32.p'
        X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
        #X_tr_emb, Y_tr = X_tr_emb[:100],Y_tr[:100]
        print ("loaded ")
        N = X_tr_emb.shape[0]

        remove_orders = []

        X, Y = X_tr_emb, Y_tr
        for i in tqdm(range(1000)):
            X, Y, rm_idx = condense_once(X, Y, X_tr_emb, Y_tr,0.1)
            print ("iteration ", i, " cur size ", len(Y))

            remove_orders.append(rm_idx)
            if X.shape[0] < 100:
                break

            # saving stuff
            # data_tier_path = 'data_sub/mnist_tiers.p'
            # pickle.dump(remove_orders, open(data_tier_path, "wb"))
            # print ("saved . . . ", data_tier_path)
        index = np.arange(X_tr_emb.shape[0])
        remove_orders = np.hstack(remove_orders)
        #print(np.delete(index,remove_orders))
        index = np.delete(index,remove_orders)
        index = np.append(index,remove_orders[::-1])
        #print(index)
        data_tier_path = 'data_sub/mnist_tiers.p'
        pickle.dump(index,open(data_tier_path,"wb"))
        print ("saved . . .", data_tier_path)
    def condensor2d():
        import time
        import pickle
        from tqdm import tqdm

        print("loading them pickle . . . ")
        data_embed_path = 'data_embed/mnist_dim2.p'
        X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
        #X_tr_emb, Y_tr = X_tr_emb[:100],Y_tr[:100]
        print ("loaded ")
        N = X_tr_emb.shape[0]

        remove_orders = []

        X, Y = X_tr_emb, Y_tr
        for i in tqdm(range(1000)):
            X, Y, rm_idx = condense_once(X, Y, X_tr_emb, Y_tr,0.1)
            print ("iteration ", i, " cur size ", len(Y))

            remove_orders.append(rm_idx)
            if X.shape[0] < 100:
                break

            # saving stuff
            # data_tier_path = 'data_sub/mnist_tiers.p'
            # pickle.dump(remove_orders, open(data_tier_path, "wb"))
            # print ("saved . . . ", data_tier_path)
        index = np.arange(X_tr_emb.shape[0])
        remove_orders = np.hstack(remove_orders)
        #print(np.delete(index,remove_orders))
        index = np.delete(index,remove_orders)
        index = np.append(index,remove_orders[::-1])
        #print(index)
        data_tier_path = 'data_sub/mnist_tiers_2d.p'
        pickle.dump(index,open(data_tier_path,"wb"))
        print ("saved . . .", data_tier_path)





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
            X, Y, rm_idx = condense_once(X, Y, X_tr_emb, Y_tr, loss='regression')
            print ("iteration ", i, " cur size ", len(Y))

            remove_orders.append(rm_idx)

            # saving stuff
            # data_tier_path = 'data_sub/mnist_tiers.p'
            # pickle.dump(remove_orders, open(data_tier_path, "wb"))
            # print ("saved . . . ", data_tier_path)

    # test5()

    def test6():
        import time
        import pickle
        from tqdm import tqdm

        print ("loading them pickle . . . ")
        data_embed_path = 'data_embed/fashion_dim32.p'
        X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
        print ("loaded ")
        N = X_tr_emb.shape[0]

        remove_orders = []

        X, Y = X_tr_emb, Y_tr
        for i in tqdm(range(800)):
            X, Y, rm_idx = condense_once(X, Y, X_tr_emb, Y_tr)
            print ("iteration ", i, " cur size ", len(Y))

            remove_orders.append(rm_idx)

            # saving stuff
            data_tier_path = 'data_sub/fashion_tiers.p'
            pickle.dump(remove_orders, open(data_tier_path, "wb"))
            print ("saved . . . ", data_tier_path)

    # test6()

    def test7():
        import time
        import pickle
        from tqdm import tqdm

        print ("loading them pickle . . . ")
        data_embed_path = 'data_embed/mnist_usps_dim32.p'
        print ("condensing ", data_embed_path)
        X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
        print ("loaded ")
        N = X_tr_emb.shape[0]

        remove_orders = []

        X, Y = X_tr_emb, Y_tr
        for i in tqdm(range(800)):
            X, Y, rm_idx = condense_once(X, Y, X_tr_emb, Y_tr)
            print ("iteration ", i, " cur size ", len(Y))

            remove_orders.append(rm_idx)

            # saving stuff
            data_tier_path = 'data_sub/mnist_usps_tiers.p'
            pickle.dump(remove_orders, open(data_tier_path, "wb"))
            print ("saved . . . ", data_tier_path)

    # test7()

    def test8():
        import time
        import pickle
        from tqdm import tqdm

        print ("loading them pickle . . . ")
        data_embed_path = 'data_embed/mnist_skew_dim32.p'
        print ("condensing ", data_embed_path)
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
            data_tier_path = 'data_sub/mnist_skew_tiers.p'
            pickle.dump(remove_orders, open(data_tier_path, "wb"))
            print ("saved . . . ", data_tier_path)

    condensor2d()
