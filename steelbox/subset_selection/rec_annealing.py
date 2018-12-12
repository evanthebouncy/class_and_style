import numpy as np
import random
import time
import math
from copy import deepcopy
from sklearn.cluster import KMeans
from .knn import score_subset, update_weight

# the base annealing algorithm with weights
def sub_select_knn_wei(X, Y, W, n_samples):
    from copy import deepcopy
    data_size = len(Y)

    # return a index _NOT_ present in the sub_idxs, not really efficient but ok . . . 
    def random_other_idx(sub_idxs):
        ret = random.randint(0, data_size - 1)
        if ret in sub_idxs:
            return random_other_idx(sub_idxs)
        return ret

    # swap out an existing index with another one not in it
    def swap_one(sub_idxs, swap_idx):
        sub_idxs = deepcopy(sub_idxs)
        sub_idxs[swap_idx] = random_other_idx(sub_idxs)
        return sub_idxs

    # initialize the sub idxs and the corresponding subsets
    sub_idxs = np.random.choice(list(range(len(Y))), n_samples, replace=False)
    X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]
    score_old = score_subset(X_sub, Y_sub, X, Y, W)
    stuck_cnt = 0

    # iterate for 100 times the n_sample size
    for i in range(n_samples * 100):
        stuck_cnt += 1
        # break if local maximum is reached . . . 
        if stuck_cnt > n_samples:
            break
        swap_idx = i % n_samples
        new_sub_idxs = swap_one(sub_idxs, swap_idx)
        new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]

        score_new = score_subset(new_X_sub, new_Y_sub, X, Y, W)

        # if found a better subset, reset the counters and update the current best
        if score_new > score_old:
            X_sub, Y_sub = new_X_sub, new_Y_sub
            sub_idxs = new_sub_idxs
            score_old = score_new
            stuck_cnt = 0

    return X_sub, Y_sub, update_weight(X_sub, Y_sub, X, Y, W)

def subluangao(X, Y, W, bin_size, ratio):
    """
        bin_size : if size_of_X less than bin_size, do annealing,
                   where n_sample is ratio * size_of_X
                   otherwise, use kmeans to split into 2 clusters and recurse

        Once every bin is finished with annealing, aggregate all the 'elders'
        together with their appropriate weights

        This is a 1-shot elder selection process
    """
    # if recursive set size less than bin perform the usual annealing
    if (X.shape[0] <= bin_size):
        return sub_select_knn_wei(X, Y, W, int(X.shape[0]*ratio))

    # otherwise recurse with 2 clusters in a binary fashion
    else:
        kmeans = KMeans(n_clusters=2)
        kmeans = kmeans.fit(X)
        cluster_labels = kmeans.predict(X)
        left_idxs = np.where(cluster_labels == 0)
        right_idxs = np.where(cluster_labels == 1)

        X_left, Y_left, W_left = X[left_idxs], Y[left_idxs], W[left_idxs]
        X_right, Y_right, W_right = X[right_idxs], Y[right_idxs], W[right_idxs]

        X_left_sub, Y_left_sub, W_left_sub = subluangao(X_left, Y_left, W_left, 
                                                        bin_size, ratio)
        X_right_sub, Y_right_sub, W_right_sub = subluangao(X_right, Y_right, W_right, 
                                                        bin_size, ratio)
        X_sub = np.append(X_left_sub, X_right_sub, axis=0)
        Y_sub = np.append(Y_left_sub, Y_right_sub, axis=0)
        W_sub = np.append(W_left_sub, W_right_sub, axis=0)
        return X_sub, Y_sub, W_sub

# def luangao2(leng, ratio, chosepoint, X, Y, KNNp, t=10000):
def luangao2(X, Y, n_samples, bin_size, ratio, t=10000):

    # hace mucho tiempo, los todo W estaban uno
    W = np.ones(Y.shape, float)

    for i in range(t):
        X,  Y,  W = subluangao(X, Y, W, bin_size, ratio)
        if (X.shape[0] * ratio * 0.5 < n_samples):
            break

    xx, yy, ww = sub_select_knn_wei(X, Y, W, n_samples)
    return xx, yy, ww

def rec_select(leng,ratio,chosepoint,X,Y,t=10000):
    dic = {}
    for i in range(X.shape[0]):
        dic[X[i].tostring()]=i
    xx,yy,ww = luangao2(leng,ratio,chosepoint,X,Y,t)
    result = np.zeros(chosepoint)
    for i in range(xx.shape[0]):
        result[i]=dic[xx[i].tostring()]
    return xx,yy,ww,result

if __name__ == '__main__':

    def test1():
        from data_raw.artificial import gen_data
        X, Y, X_t, Y_t = gen_data(2000)
        W = np.ones(1000)
        X_rsub, Y_rsub = X[:100, :], Y[:100]
        X_sub, Y_sub, W_sub = sub_select_knn_wei(X, Y, W, 100)
        print ("score of rand subset\n", score_subset(X_rsub, Y_rsub, X, Y, W))
        print ("score of anneal subset\n", score_subset(X_sub, Y_sub, X, Y, W))

    # test1()

    def test2():
        from data_raw.artificial import gen_data
        X, Y, X_t, Y_t = gen_data(2000)
        W = np.ones(1000)

        # random subset
        X_rsub, Y_rsub = X[:100, :], Y[:100]
        # normal subset
        X_sub, Y_sub, W_sub = sub_select_knn_wei(X, Y, W, 100)
        # cell partition subset
        X_sub_cell, Y_sub_cell, W_sub_cell = subluangao(X, Y, W, 100, 0.1)
        print ("score of rand\n", score_subset(X_rsub, Y_rsub, X, Y, W))
        print ("score of anneal\n", score_subset(X_sub, Y_sub, X, Y, W))
        print ("score of cell-anneal\n", score_subset(X_sub_cell, Y_sub_cell, X, Y, W))

        # recursive subset
        bin_sizes = [50, 100, 200]
        ratios = [0.5, 0.7]
        for bin_size in bin_sizes:
            for ratio in ratios:
                X_sub_rec, Y_sub_rec, W_sub_rec = luangao2(X, Y, 100, bin_size, ratio)
                print ("score of rec-anneal bin_size {} ratio {}\n".format(bin_size, ratio), score_subset(X_sub_rec, Y_sub_rec, X, Y, W))

    # test2()

    def test3():
        from data_raw.artificial import gen_data
        X, Y, X_t, Y_t = gen_data(2000)
        W = np.ones(1000)

        n_sample = 100

        # random subset
        X_rsub, Y_rsub = X[:n_sample, :], Y[:n_sample]
        # normal subset
        X_sub, Y_sub, W_sub = sub_select_knn_wei(X, Y, W, n_sample)
        # cell partition subset
        X_sub_cell, Y_sub_cell, W_sub_cell = subluangao(X, Y, W, n_sample, 0.1)
        print ("\nscore of rand\n", score_subset(X_rsub, Y_rsub, X, Y, W))
        print ("\nscore of anneal\n", score_subset(X_sub, Y_sub, X, Y, W))
        print ("\nscore of cell-anneal\n", score_subset(X_sub_cell, Y_sub_cell, X, Y, W))

        # recursive subset
        print ("\nscore of rec-anneal\n")
        bin_sizes = [50, 100, 200]
        ratios = [0.1, 0.2, 0.3, 0.5, 0.7]
        for bin_size in bin_sizes:
            for ratio in ratios:
                try:
                    X_sub_, Y_sub_, W_sub_ = luangao2(X, Y, n_sample, bin_size, ratio)
                    msg = "bin_size {} ratio {}\n".format(bin_size, ratio)
                    print (msg, score_subset(X_sub_, Y_sub_, X, Y, W))
                except:
                    pass
    test3()

    # K = pd.read_pickle('mnist_dim32.p')
    # print(K.shape)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print(y_train.shape,y_test.shape)
    # X=K
    # Y=y_train

    # XX=deepcopy(X)
    # YY=deepcopy(Y)
    # together = zip(X,Y)
    # chosepoint=100
    # KNNpoint=1
    # batchsize = 10
    # ratio = 0.5
    # dat_size = X.shape[0]

    # starttime = time.time()

    # xt1,yt1=luangao2(batchsize,ratio,chosepoint,X,Y)
    # print(time.time()-starttime)

    # print("to beat updated", evalModel(xt1, yt1, XX, YY, KNNpoint))
