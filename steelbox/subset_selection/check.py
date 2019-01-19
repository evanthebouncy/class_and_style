import time
import pickle
import numpy as np
from subset_selection.rec_annealing import anneal_optimize

data_path = 'data_sub/mnist_tiers.p'
idx_condensor = pickle.load(open(data_path,"rb"))
print(idx_condensor.shape)
X,Y=pickle.load(open('data_embed/mnist_dim32.p','rb'))

size = 100
frac = 1.1
cond_anneal = []

while size<idx_condensor.shape[0]:
    index = idx_condensor[:size]
    tl = 60
    if size==100:
        tl = 300
    cond_anneal.append(anneal_optimize(index,X,Y,tl))
    pickle.dump(cond_anneal,open('data_sub/condensor_anneal.p','wb'))
    size = int(size*frac)
    print('saved',size)
