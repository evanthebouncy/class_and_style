from .data_sampler import WSampler, make_tier_idx 
from .classifiers import FCNet
import numpy as np
import pickle
import random

def eval_fc(dataset, tiers):

    if dataset == 'artificial':
        X_tr, Y_tr, X_t, Y_t = pickle.load(open('data_raw/artificial/artificial.p', 'rb'))
    if dataset == 'mnist':
        from data_raw.mnist_ import gen_data
        X_tr, Y_tr, X_t, Y_t = gen_data("./data_raw")

    tier_results = []

    for i in range(1, len(tiers)):
        sub_idx = sum(tiers[:i], [])
        sub_wei = np.ones([len(sub_idx)],)

        X_tr_sub = X_tr[sub_idx]
        Y_tr_sub = Y_tr[sub_idx]

        sampler = WSampler(X_tr_sub, Y_tr_sub, sub_wei)
        X_dim = X_tr.shape[1]
        Y_dim = len(np.unique(Y_tr))
        fc = FCNet(X_dim, Y_dim).cuda()
        fc.learn(sampler)
        fc_score = fc.evaluate((X_t, Y_t))

        print ("data len ", len(sub_idx), " has accuracy ", fc_score)
        tier_results.append( (len(sub_idx), fc_score) )

        # result_path = 'results/mnist_tiers.p'
        # pickle.dump(tier_results, open(result_path, "wb"))
        # print ('result saved')

def point_eval_fc(dataset, tiers, i, is_rand, is_aug):

    if dataset == 'artificial':
        X_tr, Y_tr, X_t, Y_t = pickle.load(open('data_raw/artificial/artificial.p', 'rb'))
    if dataset == 'mnist':
        from data_raw.mnist_ import gen_data
        X_tr, Y_tr, X_t, Y_t = gen_data("./data_raw")


    sub_idx = sum(tiers[:i], [])
    sub_wei = np.ones([len(sub_idx)],)

    if is_rand:
        print ("using random index ")
        sub_idx = np.random.choice(list(range(X_tr.shape[0])), len(sub_idx), replace=False)

    X_tr_sub = X_tr[sub_idx]
    Y_tr_sub = Y_tr[sub_idx]

    def aug(X):
        return X + np.random.normal(0, 1, size=X.shape)

    aug = aug if is_aug else None

    sampler = WSampler(X_tr_sub, Y_tr_sub, sub_wei, aug)
    X_dim = X_tr.shape[1]
    Y_dim = len(np.unique(Y_tr))
    fc = FCNet(X_dim, Y_dim).cuda()
    fc.learn(sampler)
    fc_score = fc.evaluate((X_t, Y_t))

    print ("\n\n data len ", len(sub_idx), " has accuracy ", fc_score, "\n\n")

# see if training in/out of order of tiers has an effect
def eval_fc_order(dataset, tiers, in_order = True):
    if dataset == 'artificial':
        X_tr, Y_tr, X_t, Y_t = pickle.load(open('data_raw/artificial/artificial.p', 'rb'))
    if dataset == 'mnist':
        from data_raw.mnist_ import gen_data
        X_tr, Y_tr, X_t, Y_t = gen_data("./data_raw")


    X_dim = X_tr.shape[1]
    Y_dim = len(np.unique(Y_tr))
    fc = FCNet(X_dim, Y_dim).cuda()

    ordered_idx = sum(tiers, [])

    ordered_idx = list(reversed(ordered_idx))

    if not in_order:
        random.shuffle(ordered_idx)

    num_chunks = len(ordered_idx) // 40
    for i in range(num_chunks):
        sub_idx = ordered_idx[40 * i : 40 * (i + 1)]
        X_tr_sub = X_tr[sub_idx]
        Y_tr_sub = Y_tr[sub_idx]

        fc.learn_once(X_tr_sub, Y_tr_sub)
    fc_score = fc.evaluate((X_t, Y_t))

    print ("data len ", len(ordered_idx), " has accuracy ", fc_score)



if __name__ == '__main__':
    print ("hi")
    dataset = 'mnist'

    data_tier_path = 'data_sub/mnist_tiers.p'
    mnist_tiers = pickle.load(open(data_tier_path, "rb"))
    tiers = make_tier_idx(mnist_tiers, 60000)

    # eval_fc(dataset, tiers)
    def test_tiers_point():
        while True:
            i = int(input("enter i\n"))
            is_rand = eval(input("enter is rand\n"))
            is_aug = eval(input("enter is aug\n"))
            point_eval_fc(dataset, tiers, i, is_rand, is_aug)

    # eval_fc_order(dataset, tiers, True)
    test_tiers_point()
