from evaluation.classifiers import FCNet, LGR, EKNN, CNN1, \
    SVM, SGD, DTREE, EKNN, QDA, \
    RFOREST, GP
from evaluation.data_sampler import WSampler, make_tier_idx
import json
import numpy as np
import pickle
import random
import itertools
import tqdm



def get_idx(subset_name, subset_size):
    # realistically this should not happen anymore
    if subset_size > 60000:
        return 'bad_size'
    import pickle
    if subset_name == 'random':
        return np.random.choice(range(60000), subset_size, replace=False)

    data_path = 'data_sub/mnist_' + subset_name + '.p'
    mnist_idx = pickle.load(open(data_path, 'rb'))

    # tiers is stored as a flattened list, so we can use the same logic as random
    if subset_name == 'tiers':
        return mnist_idx[:subset_size]

    # these sets are stored as a 2d array of list of lists
    else:
        # get the length for each of the elements in mnist_idx
        all_sizes = [len(x) for x in mnist_idx]
        # get the best matching size to the size requested
        closest_idx = np.argmin([abs(size - subset_size) for size in all_sizes])
        return mnist_idx[closest_idx]


def eval_model(model_name, subset_name, subset_size):
    result = {}
    import pickle
    mnist_idx = get_idx(subset_name, subset_size)
    if type(mnist_idx) == str:
        result['error'] = 'bad_size'
        return result

    # timeout in seconds for the model LGR / FC / CNN which are implemented in pytorch
    neural_timeout = 300
    stop_criteria = (0.01, 1000, neural_timeout)

    models = {

        # 'LGR': lambda : LGR(28 * 28, 10, stop_criteria).cuda(),
        # 'FC': lambda :  FCNet(28 * 28, 10, stop_criteria).cuda(),
        # 'CNN': lambda : CNN1((1, 28, 28), 10, stop_criteria).cuda(),
        'LGR': lambda: LGR(28 * 28, 10, stop_criteria),
        'FC': lambda: FCNet(28 * 28, 10, stop_criteria),
        'CNN': lambda: CNN1((1, 28, 28), 10, stop_criteria),
        'SVMrbf': lambda: SVM('rbf'),
        'SVMLin': lambda: SVM('linear'),
        'DTREE': lambda: DTREE(),
        'EKNN': lambda: EKNN(),
        'RFOREST': lambda: RFOREST(),
    }
    if model_name not in models.keys():
        print('Wrong Model Name, has to be one of FC, CNN, SVMrbf, SVMlin, DTREE, EKNN, RFOREST')
        assert 0
    if subset_name not in ['random', 'tiers', 'kmeans', 'tiers_anneal', 'kmeans_anneal', 'random_anneal']:
        print('Wrong subset name, has to be one of [random,tiers,kmeans,tiers_anneal,kmeans_anneal,random_anneal]')

    model = models[model_name]()
    from data_raw.mnist_ import gen_data as mnist_gen_data
    MNIST_X_tr, MNIST_Y_tr, MNIST_X_t, MNIST_Y_t = mnist_gen_data("./data_raw")

    X_tr, Y_tr = MNIST_X_tr[mnist_idx], MNIST_Y_tr[mnist_idx]

    sampler = WSampler(X_tr, Y_tr, np.ones([len(Y_tr)], ))

    import time
    t_start = time.time()
    model.learn(sampler)
    score_m_m = model.evaluate((MNIST_X_t, MNIST_Y_t))
    time_spent = time.time() - t_start

    terminate_cause = model.term if hasattr(model, 'term') else None

    result['num_samples'] = len(Y_tr)
    result['time'] = time_spent
    result['term'] = terminate_cause
    result['score_m_m'] = score_m_m
    result['model']
    return result


def test():
    models = ['LGR','FC','CNN','DTREE','SVMrbf','SVMLin','EKNN','RFOREST']
    subset_names = [('random',60000),('tiers_anneal',10000)]
    # subset_names = ['random','tiers','kmeans','tiers_anneal','kmeans_anneal','random_anneal']
    #models = ['LGR', 'FC', 'CNN']
    #subset_names = ['random', 'tiers', 'kmeans', 'tiers_anneal', 'kmeans_anneal', 'random_anneal']
    res = []
    for model in models:
        for subset_name,siz in subset_names:
            k=eval_model(model, subset_name, siz)
            print(k)
            res.append(k)
            import pickle
            pickle.dump(res,open('results/mnist_barchart.p','wb'))
test()






