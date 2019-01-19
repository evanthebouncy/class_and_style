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
    from data_raw.usps import gen_data as usps_gen_data
    USPS_X_tr, USPS_Y_tr, USPS_X_t, USPS_Y_t = usps_gen_data("./data_raw")
    from data_raw.mnist_ import gen_data_skew as skew_gen_data
    SKEW_X_tr, SKEW_Y_tr, SKEW_X_t, SKEW_Y_t = skew_gen_data("./data_raw")

    mnist_tiers = pickle.load(open('data_sub/{}_tiers.p'.format('mnist'), "rb"))
    mnist_tiers = make_tier_idx(mnist_tiers, 60000)

    skew_tiers = pickle.load(open('data_sub/{}_tiers.p'.format('mnist_skew'), "rb"))
    skew_tiers = make_tier_idx(skew_tiers, 11336)

    mnist_usps_tiers = pickle.load(open('data_sub/mnist_usps_tiers.p', "rb"))
    mnist_usps_tiers = make_tier_idx(mnist_usps_tiers, 60000 + 7291)

    mnist_prefix_tiers = sum([mnist_tiers[i] for i in range(300)], [])
    print ("lenth of mnist_prefix ", len(mnist_prefix_tiers))

    mnist_usps_prefix_tiers = sum([mnist_usps_tiers[i] for i in range(100)], [])
    print ("lenth of mnist_usps_prefix ", len(mnist_usps_prefix_tiers))

    skew_prefix_tiers = sum([skew_tiers[i] for i in range(100)], [])
    print ("lenth of skew_prefix ", len(skew_prefix_tiers))

    # measure how many USPS digits we were able to retain
    before = len([x for x in mnist_usps_prefix_tiers if x < 60000])
    after = len(mnist_usps_prefix_tiers) - before
    print ("usps-repr weight : ", after / (before + after), " avg-usps weight ", 7291 / (60000 + 7291))

    # representative subset MNIST
    MNIST_X_repr = MNIST_X_tr[mnist_prefix_tiers]
    MNIST_Y_repr = MNIST_Y_tr[mnist_prefix_tiers]

    # random subset of MNIST
    mnist_rand_idx = np.random.choice(range(60000), len(mnist_prefix_tiers), replace=False)
    MNIST_X_rand = MNIST_X_tr[mnist_rand_idx]
    MNIST_Y_rand = MNIST_Y_tr[mnist_rand_idx]

    # joint set mnist + usps
    JOIN_X_tr = np.concatenate((MNIST_X_tr, USPS_X_tr), axis=0)
    JOIN_Y_tr = np.concatenate((MNIST_Y_tr, USPS_Y_tr), axis=0)

    JOIN_X_t = np.concatenate((MNIST_X_t, USPS_X_t), axis=0)
    JOIN_Y_t = np.concatenate((MNIST_Y_t, USPS_Y_t), axis=0)

    # joint repr set mnist + usps
    JOIN_X_repr = JOIN_X_tr[mnist_usps_prefix_tiers]
    JOIN_Y_repr = JOIN_Y_tr[mnist_usps_prefix_tiers]

    # random subset of join
    mnist_usps_rand_idx = np.random.choice(range(60000 + 7291), len(mnist_usps_prefix_tiers), replace=False)
    JOIN_X_rand = JOIN_X_tr[mnist_usps_rand_idx]
    JOIN_Y_rand = JOIN_Y_tr[mnist_usps_rand_idx]

    # repr set SKEW
    SKEW_X_repr = SKEW_X_tr[skew_prefix_tiers]
    SKEW_Y_repr = SKEW_Y_tr[skew_prefix_tiers]

    # random subset of SKEW
    skew_rand_idx = np.random.choice(range(11336), len(skew_prefix_tiers), replace=False)
    SKEW_X_rand = SKEW_X_tr[skew_rand_idx]
    SKEW_Y_rand = SKEW_Y_tr[skew_rand_idx]


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

        sampler = WSampler(X_tr, Y_tr, np.ones([len(Y_tr)],))
        X_dim = X_tr.shape[1]
        Y_dim = len(np.unique(Y_tr))
        print ("===================================================== ON ", train_data)

        for model in [
                      FCNet(X_dim, Y_dim).cuda(), 
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

            # score_m_m = model.evaluate((MNIST_X_t, MNIST_Y_t))
            score_m_u = model.evaluate((USPS_X_t, USPS_Y_t))

            print ("    ------ ", model.name)
            # print ("    ", train_data, " -> MNIST has accuracy ", score_m_m)
            print ("    ", train_data, " -> USPS  has accuracy ", score_m_u)


