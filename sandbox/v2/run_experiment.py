# from sklearn.linear_model import LogisticRegression
# from sklearn import neighbors
# import numpy as np
from data import make_dataset, L

import random
import numpy as np

from classifiers import DataHolder
from classifiers import FCNet, LRegr, KNN 
from utils import *

# ========================== SUBSET SELECTION SCHEMES =======================

# ----------------- cluster in ( raw_input / embedded class ) space ----------------
def sub_select_cluster(X, Y, n_samples, embed=False, weighted=True):
  
  # either embed the X into class space or leave it as is
  X_emb = [get_class(x) for x in X] if embed else X
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=n_samples)
  kmeans = kmeans.fit(X_emb)
  
  cluster_labels = list(kmeans.predict(X_emb))
  # print (cluster_labels[:100])
  from sklearn.metrics import pairwise_distances_argmin_min
  closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_emb)
  counts = [cluster_labels.count(i) for i in range(n_samples)] if weighted\
           else [1.0] * n_samples
  
  # return [ (X[closest[i]], counts[i]) for i in range(n_samples) ], kmeans.score(X_emb)
  X_sub = []
  Y_sub = []
  for i in range(n_samples):
    X_sub.append(X[closest[i]])
    Y_sub.append(Y[closest[i]])
  return DataHolder(np.array(X_sub), np.array(Y_sub), counts) , closest

# ----------------------- optimize knn by annealing ---------------------
def sub_select_knn_anneal(X, Y, n_samples, weighted=True):
  def swap_one(sub_idxs, swap_idx):
    sub_idxs = [x for x in sub_idxs]
    sub_size = len(sub_idxs)
    all_rem_idxs = [iii for iii in range(len(X)) if iii not in sub_idxs]
    sub_idxs[swap_idx] = random.choice(all_rem_idxs)
    return sub_idxs
  
  _, sub_idxs = sub_select_cluster(X, Y, n_samples)
  X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs] 
  score_old = knn_loss(make_knn(X_sub, Y_sub), X, Y)
  stuck_cnt = 0
  for i in range(n_samples * 100000):
    stuck_cnt += 1
    if stuck_cnt > n_samples:
      break
    swap_idx = i % n_samples
    new_sub_idxs = swap_one(sub_idxs, swap_idx)
    new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
    score_new = knn_loss(make_knn(new_X_sub, new_Y_sub), X, Y)
    if score_new < score_old:
      X_sub, Y_sub = new_X_sub, new_Y_sub
      sub_idxs = new_sub_idxs
      score_old = score_new
      stuck_cnt = 0
  # print ("score ", score_old)

  # bit of redundant code but i need to find the right cluster counts for weight
  from sklearn import neighbors
  # fit with X_sub Y_sub, and see which elements in X gets assigned to which
  clf = neighbors.KNeighborsClassifier(1+int(np.log(len(Y))))
  knn_cl = clf.fit(X_sub, Y_sub)
  # only assign weights for correctly classified Xs
  correct_idxs = knn_cl.predict(X) == Y
  correct_X = X[correct_idxs, :]

  neighbors = knn_cl.kneighbors(correct_X, return_distance=False)
  neighbors_flat = np.reshape(neighbors, (-1,))
  from collections import Counter
  cnt = Counter(neighbors_flat)

  frequency = [cnt[i] for i in range(n_samples)] if weighted else n_samples * [1.0]
  return DataHolder(X_sub, Y_sub, frequency)

# ======================== RUNNING THE EXPERIMENT ===========================
def run_subset_size(clf, X_tr, Y_tr, X_t, Y_t, n_sub):
  ret_dict = dict()

  print ("================ running for ", n_sub, clf().name, "====================")
  data_holder = DataHolder(X_tr, Y_tr, [1.0] * len(X_tr))
  all_result = get_acc(clf(), data_holder, X_t, Y_t)
  print ( "all result : ", all_result)
  ret_dict['all'] = all_result

  data_holder = DataHolder(X_tr[:n_sub], Y_tr[:n_sub], [1.0] * n_sub)
  rand_result = get_acc(clf(), data_holder, X_t, Y_t)
  print ( "rand result : ", rand_result)
  ret_dict['rand'] = rand_result

  data_holder, _ = sub_select_cluster(X_tr, Y_tr, n_sub, False, False)
  clus_raw = get_acc(clf(), data_holder, X_t,Y_t)
  print ( "cluster all result unif : ", clus_raw )
  ret_dict['clus_all'] = clus_raw

  data_holder, _ = sub_select_cluster(X_tr, Y_tr, n_sub, False, True)
  clus_raw = get_acc(clf(), data_holder, X_t,Y_t)
  print ( "cluster all result weighted : ", clus_raw )
  ret_dict['clus_all_weighted'] = clus_raw

  data_holder, _ = sub_select_cluster(X_tr, Y_tr, n_sub, True, weighted=True)
  clus_cls = get_acc(clf(), data_holder, X_t,Y_t)
  print ( "cluster class result : ", clus_cls )
  ret_dict['clus_cls'] = clus_cls

  knn_w_data_holder = sub_select_knn_anneal(X, Y, n_sub, weighted=True)
  knn_weighted = get_acc(clf(), knn_w_data_holder, X_t, Y_t)
  print ( "knn_weighted : ", knn_weighted )
  ret_dict['knn_weighted'] = knn_weighted

  # modify the weights to be uniform and try again
  knn_unif_data_holder = knn_w_data_holder
  knn_unif_data_holder.weights = np.array([1.0 / n_sub] * n_sub)
  knn_unif = get_acc(clf(), knn_unif_data_holder, X_t, Y_t)
  print ( "knn_unif result : ", knn_unif )
  ret_dict['knn_unif'] = knn_unif

  return ret_dict

# ================ THE EVALUAATION CLSSIFIERS ==========
CLFS = {
    'lrgr' : LRegr,
    'knn' : KNN,
    'fcnet' : lambda : FCNet().cuda(),
    }

if __name__ == "__main__":
  n = 2000
  n_tr = n // 2

  X, Y = make_dataset(n)
  X_tr, Y_tr = X[:n_tr], Y[:n_tr]
  X_t, Y_t = X[n_tr:], Y[n_tr:]

  # some TSNE print for  all features and only class relevant features
  # print ("doing a quick tsne visualization of the data . . . ")
  # tsne(X_tr, Y_tr, "all_feature")
  # X_emb = [get_class(x) for x in X]
  # tsne(X_emb, Y, "class_feature")
  # X_common = [get_common(x) for x in X]
  # tsne(X_common, Y, "style_feature")

  clf_name = 'lrgr'
  clf = CLFS[clf_name]
  
  sub_sizes = [10 * i for i in range(1, 31)]
  run_dicts = []
  for sub_size in sub_sizes:
    run_dict = run_subset_size(clf, X_tr, Y_tr, X_t, Y_t, sub_size)
    run_dicts.append(run_dict)

  sub_scores = []
  keys = run_dicts[0].keys()
  for key in keys:
    key_vals = []
    for run_dict in run_dicts:
      key_vals.append(run_dict[key])
    sub_scores.append( (key, key_vals) )

  plot_progress(clf_name, sub_sizes, sub_scores)








