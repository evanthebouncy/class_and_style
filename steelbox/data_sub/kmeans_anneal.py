import time
import pickle
import numpy as np
from subset_selection.rec_annealing import anneal_optimize

data_path = 'data_sub/mnist_tiers_kmeans.p'
idx_condensor = pickle.load(open(data_path,"rb"))
print(idx_condensor.shape)
X,Y=pickle.load(open('data_embed/mnist_dim32.p','rb'))

cond_anneal = []

for index in idx_condensor:
    cond_anneal.append(anneal_optimize(index,X,Y))
    pickle.dump(cond_anneal,open('data_sub/kmeans_anneal.p','wb'))
    print('saved',len(index))
