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

from data_raw.sentiment_ import gen_data as sentiment_gen_data
X_tr, Y_tr, X_t, Y_t = sentiment_gen_data("./data_raw")

sampler = WSampler(X_tr, Y_tr, np.ones([len(Y_tr)], ))

model1.learn(sampler)
model2.learn(sampler)
score1 = model1.evaluate((X_t, Y_t))
score2 = model2.evaluate((X_t, Y_t))

print (score1, model1.term)
print (score2, model2.term)

