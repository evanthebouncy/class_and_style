import time
import pickle
import numpy as np
from subset_selection.rec_annealing import anneal_optimize

X,Y=pickle.load(open('data_embed/mnist_dim32.p','rb'))

size = 100
frac = 1.1
cond_anneal = []

while size<len(Y):
    tl = 60
    for i in range(4):
         
        index = np.random.choice(list(range(len(Y))),size,replace = False)
        cond_anneal.append(anneal_optimize(index,X,Y,tl))
    pickle.dump(cond_anneal,open('data_sub/random_anneal.p','wb'))
    size = int(size*frac)
    print('saved',size)
