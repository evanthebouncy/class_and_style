from evaluation.classifiers import FCNet, LRegr, EKNN, CNN1, \
                                   SVM, SGD, DTREE, EKNN, QDA, \
                                   RFOREST, GP
from evaluation.data_sampler import WSampler, make_tier_idx
import numpy as np
import pickle
import random

if __name__ == '__main__':
    from data_raw.mnist_ import gen_data as mnist_gen_data
    MNIST_X_tr, MNIST_Y_tr, MNIST_X_t, MNIST_Y_t = mnist_gen_data("./data_raw")

    mnist_tiers = pickle.load(open('data_sub/{}_tiers.p'.format('mnist'), "rb"))
    mnist_prefix_tiers = mnist_tiers
    # representative subset MNIST
    MNIST_X_repr = MNIST_X_tr[mnist_prefix_tiers]
    MNIST_Y_repr = MNIST_Y_tr[mnist_prefix_tiers]

    # random subset of MNIST
    mnist_rand_idx = np.random.choice(range(60000), len(mnist_prefix_tiers), replace=False)
    MNIST_X_rand = MNIST_X_tr[mnist_rand_idx]
    MNIST_Y_rand = MNIST_Y_tr[mnist_rand_idx]


    # train_data = 'mnist+usps'
    # train_data = '(mnist+usps)_repr'

    X_Y = {
            'mnist' : (MNIST_X_tr, MNIST_Y_tr),
            'mnist_repr'  : (MNIST_X_repr, MNIST_Y_repr),
            'mnist_rand'  : (MNIST_X_rand, MNIST_Y_rand),
            # 'usps'  : (USPS_X_tr, USPS_Y_tr),
            # 'mnist+usps'  : (JOIN_X_tr, JOIN_Y_tr),
            # 'mnist+usps)_repr'  : (JOIN_X_repr, JOIN_Y_repr),
            # 'mnist+usps)_rand'  : (JOIN_X_rand, JOIN_Y_rand),
            # 'skew' : (SKEW_X_tr, SKEW_Y_tr),
            # 'skew_rand' : (SKEW_X_rand, SKEW_Y_rand),
            # 'skew_repr' : (SKEW_X_repr, SKEW_Y_repr),
          }

    for train_data in sorted(list(X_Y.keys())):
        X_tr, Y_tr = X_Y[train_data]
        size = 200
        sampler = WSampler(X_tr[:size], Y_tr[:size], np.ones([len(Y_tr[:size])],))
        X_dim = X_tr.shape[1]
        Y_dim = len(np.unique(Y_tr))
        print ("===================================================== ON ", train_data)

        for model in [
                      #FCNet(X_dim, Y_dim).cuda(), 
                      CNN1((1, 28, 28), 10).cuda(), 
                      SVM('rbf'),
                      SVM('linear'),
                      LRegr(),
                      SGD(),
                      DTREE(),
                      EKNN(),
                      QDA(),
                      RFOREST(),
                      # GP(),
                      ]:
            model.learn(sampler)

            score_m_m = model.evaluate((MNIST_X_t, MNIST_Y_t))

            print ("    ------ ", model.name)
            # print ("    ", train_data, " -> MNIST has accuracy ", score_m_m)
            print ("    ", train_data, " -> USPS  has accuracy ", score_m_m)


