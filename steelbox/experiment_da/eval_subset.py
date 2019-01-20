from evaluation.classifiers import FCNet, LRegr, EKNN, CNN1, \
                                   SVM, SGD, DTREE, EKNN, QDA, \
                                   RFOREST, GP
from evaluation.data_sampler import WSampler, make_tier_idx
import numpy as np
import pickle
import random

def get_size_from_index(size):
    import math
    return int(math.pow(1.1,size)*100)

def get_idx(subset_name,subset_size_index):
    import pickle
    if subset_name == 'random':
        subset_size = get_size_from_index(subset_size_index)
        if subset_size > 60000:
            return 'bad_size'
        return np.random.choice(range(60000), subset_size, replace=False)
    data_path = 'data_sub/mnist_' + subset_name + '.p'
    mnist_idx = pickle.load(open(data_path,'rb'))
    if subset_name == 'tiers':
        subset_size = get_size_from_index(subset_size_index)
        if subset_size > 60000:
            return 'bad_size'
        return mnist_idx[:subset_size]
    else:
        if subset_size_index>=len(mnist_idx):
            return 'bad_size'
        return mnist_idx[subset_size_index]

def eval_model(model_name,subset_name,subset_size_index):
    import pickle
    mnist_idx = get_idx(subset_name,subset_size_index)
    if type(mnist_idx)==str:
        return 'bad_size'

    models = {'FC': lambda : FCNet(X_dim, Y_dim).cuda(),
              'CNN': lambda : CNN1((1, 28, 28), 10).cuda(),
              'SVMrbf': lambda : SVM('rbf'),
              'SVMLin': lambda : SVM('linear'),
              'DTREE' : lambda : DTREE(),
              'EKNN' : lambda : EKNN(),
              'RFOREST' : lambda : RFOREST(),
    }
    if model_name not in models.keys():
        print('Wrong Model Name, has to be one of FC, CNN, SVMrbf, SVMlin, DTREE, EKNN, RFOREST')
        assert 0
    if subset_name not in ['random','tiers','kmeans','tiers_anneal','kmeans_anneal','random_anneal']:
        print('Wrong subset name, has to be one of [random,tiers,kmeans,tiers_anneal,kmeans_anneal,random_anneal]')

    model = models[model_name]()
    from data_raw.mnist_ import gen_data as mnist_gen_data
    MNIST_X_tr, MNIST_Y_tr, MNIST_X_t, MNIST_Y_t = mnist_gen_data("./data_raw")


    X_tr,Y_tr = MNIST_X_tr[mnist_idx],MNIST_Y_tr[mnist_idx]

    sampler = WSampler(X_tr, Y_tr, np.ones([len(Y_tr)], ))

    model.learn(sampler)

    score_m_m = model.evaluate((MNIST_X_t, MNIST_Y_t))

    return_string = model_name+'_'+subset_name+'_'+str(len(Y_tr))+'_'+str(score_m_m)
    return return_string

def test():
    models = ['DTREE','SVMrbf','SVMLin','EKNN','RFOREST']
    subset_names = ['random','tiers','kmeans','tiers_anneal','kmeans_anneal','random_anneal']
    for model in models:
        for subset_name in subset_names:
            print(eval_model(model,subset_name,2))
            print(eval_model(model, subset_name, 1000))

def example():
    ans = []
    models = ['FC','CNN','DTREE','SVMrbf','SVMLin','EKNN','RFOREST']
    subset_names = ['random','tiers','kmeans','tiers_anneal','kmeans_anneal','random_anneal']
    for model in models:
        for subset_name in subset_names:
            for subset_size_index in range(0,1000):
                str = eval_model(model,subset_name,subset_size_index)
                if str == 'bad_size':
                    break
                ans.append(str)
    return ans



if __name__ == '__main__':
    test()
