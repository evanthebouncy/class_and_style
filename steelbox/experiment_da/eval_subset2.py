from evaluation.classifiers import FCNet, LGR, EKNN, CNN1, \
    SVM, SGD, DTREE, EKNN, QDA, \
    RFOREST, GP
from evaluation.data_sampler import WSampler, make_tier_idx
import json
import numpy as np
import pickle
import random
import itertools
# import ray
import tqdm

neural_timeout = 60
stop_criteria = (0.01, 1000, neural_timeout)
model1 = LGR(1024, 2, stop_criteria).cuda()
model2 = FCNet(1024, 2, stop_criteria).cuda()


# subset_size is an integer number
def get_idx(subset_name, subset_size):
    # realistically this should not happen anymore
    if subset_size > 67340:
        return 'bad_size'
    import pickle
    if subset_name == 'random':
        return np.random.choice(range(67340), subset_size, replace=False)

    data_path = 'data_sub/sentiment_' + subset_name + '.p'
    sentiment_idx = pickle.load(open(data_path, 'rb'))

    # tiers is stored as a flattened list, so we can use the same logic as random
    if subset_name == 'tiers':
        return sentiment_idx[:subset_size]

    # these sets are stored as a 2d array of list of lists
    else:
        # get the length for each of the elements in mnist_idx
        all_sizes = [len(x) for x in sentiment_idx]
        # get the best matching size to the size requested
        closest_idx = np.argmin([abs(size - subset_size) for size in all_sizes])
        return sentiment_idx[closest_idx]


# @ray.remote
def eval_model(model_name, subset_name, subset_size):
    import pickle
    sentiment_idx = get_idx(subset_name, subset_size)
    if type(sentiment_idx) == str:
        return 'bad_size'

    # timeout in seconds for the model LGR / FC / CNN which are implemented in pytorch
    neural_timeout = 60
    stop_criteria = (0.01, 1000, neural_timeout)

    models = {

        'LGR': lambda: LGR(1024, 2, stop_criteria).cuda(),
        'FC': lambda: FCNet(1024, 2, stop_criteria).cuda()

    }
    if model_name not in models.keys():
        print('Wrong Model Name, has to be one of FC, CNN, ')
        assert 0
    if subset_name not in ['random', 'tiers', 'kmeans', 'tiers_anneal', 'kmeans_anneal', 'random_anneal']:
        print('Wrong subset name, has to be one of [random,tiers,kmeans,tiers_anneal,kmeans_anneal,random_anneal]')
        assert 0

    model = models[model_name]()
    # from data_raw.mnist_ import gen_data as mnist_gen_data
    # MNIST_X_tr, MNIST_Y_tr, MNIST_X_t, MNIST_Y_t = mnist_gen_data("./data_raw")

    from data_raw.sentiment_ import gen_data as sentiment_gen_data
    Sentiment_X_tr, Sentiment_Y_tr, X_t, Y_t = sentiment_gen_data("./data_raw")

    X_tr, Y_tr = Sentiment_X_tr[sentiment_idx], Sentiment_Y_tr[sentiment_idx]

    sampler = WSampler(X_tr, Y_tr, np.ones([len(Y_tr)], ))

    import time
    t_start = time.time()
    model.learn(sampler)
    score_m_m = model.evaluate((X_t, Y_t))
    time_spent = time.time() - t_start

    terminate_cause = model.term if hasattr(model, 'term') else None

    return_string = json.dumps(
        {'model_name': model_name,
         'subset_name': subset_name,
         'num_samples': len(Y_tr),
         'time': time_spent,
         'term': terminate_cause,
         'score_m_m': score_m_m})

    return return_string


def test():
    models = ['LGR', 'FC']
    subset_names = ['random', 'tiers', 'kmeans', 'tiers_anneal', 'kmeans_anneal', 'random_anneal']
    # in this order, smallest, biggest, and in-between
    subset_sizes = [100, 60000, 250, 500, 1000, 2000, 4000, 8000, 15000, 30000]
    all_experiments = list(itertools.product(subset_sizes, subset_names, models))

    print(len(all_experiments))

    random_exp = random.choice(all_experiments)
    print(random_exp)
    subset_size, subset_name, model = random_exp
    print(eval_model(model, subset_name, subset_size))


# RICHARD PARALLELIZE THIS GIANT LOOP THINGIE THIS THING THIS THIS ! ! !
def example(output_path):
    output = open(output_path, 'w')

    ans = []
    models = ['LGR', 'FC']
    subset_names = ['random', 'tiers', 'kmeans', 'tiers_anneal', 'kmeans_anneal', 'random_anneal']
    all_experiments = list(itertools.product(models,
                                             subset_names, range(0, 1000)))

    pbar = tqdm.tqdm(total=len(all_experiments), dynamic_ncols=True)
    experiment_iter = iter(all_experiments)
    ready, futures = [], []

    while True:
        while len(futures) < 1000:
            try:
                model, subset_name, subset_size_index = next(experiment_iter)
                f = eval_model.remote(model, subset_name, subset_size_index)
                futures.append(f)
            except Stopiteration:
                break
        ready, futures = ray.wait(futures)
        for f in ready:
            result = ray.get(f)
            output.write(result + '\n')
            output.flush()
            pbar.update(1)
        if not futures:
            break
    output.close()


if __name__ == '__main__':
    # ray.init(num_cpus=64)
    # example('eval_subset_results.jsonl')
    test()
