
from evaluation.data_sampler import WSampler, make_tier_idx
from .knn import score_subset
import json
import numpy as np



def get_size_from_index(size):
    import math
    return int(math.pow(1.1, size) * 100)


def get_idx(subset_name, subset_size_index):
    import pickle
    if subset_name == 'random':
        subset_size = get_size_from_index(subset_size_index)
        if subset_size > 60000:
            return 'bad_size'
        return np.random.choice(range(60000), subset_size, replace=False)
    data_path = 'data_sub/mnist_' + subset_name + '.p'
    mnist_idx = pickle.load(open(data_path, 'rb'))
    if subset_name == 'tiers':
        subset_size = get_size_from_index(subset_size_index)
        if subset_size > 60000:
            return 'bad_size'
        return mnist_idx[:subset_size]
    else:
        if subset_size_index >= len(mnist_idx):
            return 'bad_size'
        return mnist_idx[subset_size_index]



def eval_model(subset_name, subset_size_index):
    import pickle
    mnist_idx = get_idx(subset_name, subset_size_index)
    if type(mnist_idx) == str:
        return 'bad_size'


    from data_raw.mnist_ import gen_data as mnist_gen_data
    MNIST_X_tr, MNIST_Y_tr, MNIST_X_t, MNIST_Y_t = mnist_gen_data("./data_raw")

    X_tr, Y_tr = MNIST_X_tr[mnist_idx], MNIST_Y_tr[mnist_idx]
    W=np.ones(MNIST_Y_tr.shape)


    score_m_m = score_subset(X_tr,Y_tr,MNIST_X_tr,MNIST_Y_tr,W)/W.shape[0]
    return score_m_m,Y_tr.shape[0]



def test():
    # models = ['DTREE','SVMrbf','SVMLin','EKNN','RFOREST']
    # subset_names = ['random','tiers','kmeans','tiers_anneal','kmeans_anneal','random_anneal']
    subset_names = ['random', 'tiers', 'kmeans', 'tiers_anneal', 'kmeans_anneal', 'random_anneal']
    min_siz_95=1e9
    min_siz_99=1e9
    min_subset = (0,0)
    for subset_name in subset_names:
        for i in range(1000):
            score,siz = eval_model(subset_name,i)
            print(score,siz)
            if score>0.95:
                if siz<min_siz_95:
                    min_siz_95 = siz
                    min_subset = (subset_name,i)
            if score>0.99:
                if siz<min_siz_99:
                    min_siz_99 = siz
                    min_subset = (subset_name,i)
    print(min_siz_95,' 95')
    print(min_siz_99,' 99')






if __name__ == '__main__':
    test()
