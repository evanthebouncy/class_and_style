from .data_sampler import WSampler
from .classifiers import FCNet
import numpy as np

def eval_fc(dataset, n_sample, emb_method, subset_method):
    import pickle

    if dataset == 'artificial':
        X_tr, Y_tr, X_t, Y_t = pickle.load(open('data_raw/artificial/artificial.p', 'rb'))
    if dataset == 'mnist':
        from data_raw.mnist_ import gen_data
        X_tr, Y_tr, X_t, Y_t = gen_data("./data_raw")
    n_sample = str(n_sample)
    sub_path = 'data_sub/'+dataset+'_'+n_sample+'_'+emb_method+'_'+subset_method+'.p'
    
    print ("data from ", sub_path)
    sub_idx, sub_wei = pickle.load(open(sub_path, "rb"))

    # print ("using uniform weight for everything")
    # sub_wei = np.ones([len(sub_wei)],)

    X_tr_sub = X_tr[sub_idx]
    Y_tr_sub = Y_tr[sub_idx]

    sampler = WSampler(X_tr_sub, Y_tr_sub, sub_wei)
    X_dim = X_tr.shape[1]
    Y_dim = len(np.unique(Y_tr))
    fc = FCNet(X_dim, Y_dim).cuda()
    fc.learn(sampler)
    print ("has accuracy ", fc.evaluate((X_t, Y_t)))


if __name__ == '__main__':
    print ("hi")
    dataset = 'mnist'
    n_sample = 100
    emb_method = 'vae'
    # subset_methods = ['rsub', 'anneal', 'cellsub', 'recsub']
    subset_methods = ['rsub', 'cluster', 'approx', 'recsub']
    for subset_method in subset_methods: 
        eval_fc(dataset, n_sample, emb_method, subset_method)
