import numpy as np
from .rec_annealing import sub_select_anneal_wei,\
                          sub_select_cell,\
                          sub_select_rec,\
                          score_subset,\
                          recover_index


def select_artificial(n_samples = [10, 20, 50, 100, 200]):
    import pickle
    data_embed_path = 'data_embed/artificial_dim10.p'
    X_tr_emb, Y_tr = pickle.load(open(data_embed_path, "rb"))
    N = X_tr_emb.shape[0]

    print ("selecting subset for : ", data_embed_path)

    def save_index_weight(idxes, wei,name):
        data_subset_path = 'data_sub/'+name
        pickle.dump((idxes, wei), open(data_subset_path, "wb"))
        print ("saved . . . ", data_subset_path)

    # DO THE WORK
    W = np.ones(N)
    for n_sample in n_samples:

        # random subset
        X_rsub, Y_rsub = X_tr_emb[:n_sample, :], Y_tr[:n_sample]
        print ("\nscore of rand\n", score_subset(X_rsub, Y_rsub, X_tr_emb, Y_tr, W, sub_bad = False))
        rsub_idxes = recover_index(X_rsub, X_tr_emb)
        rsub_wei = np.ones(n_sample)
        save_index_weight(rsub_idxes, rsub_wei, 'artificial_{}_vae_rsub.p'.format(n_sample))

        # normal subset
        X_anneal, Y_anneal, W_anneal = sub_select_anneal_wei(X_tr_emb, Y_tr, W, n_sample)
        print ("\nscore of anneal\n", score_subset(X_anneal, Y_anneal, X_tr_emb, Y_tr, W, sub_bad = False))
        anneal_idxes = recover_index(X_anneal, X_tr_emb)
        save_index_weight(anneal_idxes, W_anneal, 'artificial_{}_vae_anneal.p'.format(n_sample))

        # cell partition subset
        ratio = n_sample / N
        X_cell, Y_cell, W_cell = sub_select_cell(X_tr_emb, Y_tr, W, 50, ratio)
        print ("\nscore of cell-anneal\n", score_subset(X_cell, Y_cell, X_tr_emb, Y_tr, W, sub_bad = False))
        cell_idxes = recover_index(X_cell, X_tr_emb)
        save_index_weight(cell_idxes, W_cell, 'artificial_{}_vae_cellsub.p'.format(n_sample))

        # recursive subset
        best_score = 0
        best_param = None
        best_idx_wei = None
        print ("\nscore of rec-anneal\n")
        bin_sizes = [20, 50, 100]
        ratios = [0.1, 0.2, 0.3, 0.5, 0.7]
        for bin_size in bin_sizes:
            for ratio in ratios:
                try:
                    X_sub_, Y_sub_, W_sub_ = sub_select_rec(X_tr_emb, Y_tr, n_sample, bin_size, ratio)
                    param = "bin{}_ratio{}\n".format(bin_size, ratio)
                    rec_score = score_subset(X_sub_, Y_sub_, X_tr_emb, Y_tr, W, sub_bad = False)
                    if rec_score > best_score:
                        print ("new best score ", rec_score, " with params ", param)
                        idxes = recover_index(X_sub_, X_tr_emb)
                        best_score = rec_score
                        best_param = param
                        best_idx_wei = idxes, W_sub_
                except:
                    pass
        best_score
        save_index_weight(*best_idx_wei, 'artificial_{}_vae_recsub.p'.format(n_sample))

if __name__ == '__main__':
    select_artificial()
