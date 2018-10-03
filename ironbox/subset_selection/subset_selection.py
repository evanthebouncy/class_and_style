import random
import numpy as np
from sklearn import neighbors
from copy import deepcopy

# ======================= SUBSET SELECTION SCHEMES =======================
# just random
def sub_select_random(X, Y, n_samples, ae=None):
  r_idx = np.random.choice(range(len(X)), n_samples)
  return r_idx

# clustering in the latent space, without regarding of different labels
def sub_select_cluster(X, Y, n_samples, ae):
  
  X_emb = ae.embed(X)
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=n_samples)
  kmeans = kmeans.fit(X_emb)
  
  cluster_labels = list(kmeans.predict(X_emb))
  # print (cluster_labels[:100])
  from sklearn.metrics import pairwise_distances_argmin_min
  closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_emb)
  
  return closest

# optimize knn by annealing, respecting the labels
def sub_select_knn(X, Y, n_samples, ae, kind='classification', approximate_size=1000):
  assert kind in ['classification', 'regression']
  make_knn = make_cl_knn if kind == 'classification' else make_reg_knn
  knn_loss = knn_cl_loss if kind == 'classification' else knn_reg_loss

  data_size = len(X)

  def random_other_idx(sub_idxs):
    ret = random.randint(0, data_size-1)
    if ret in sub_idxs:
      return random_other_idx(sub_idxs)
    return ret

  def swap_one(sub_idxs, swap_idx):
    sub_idxs = deepcopy(sub_idxs)
    # all_rem_idxs = [iii for iii in range(len(X)) if iii not in sub_idxs]
    # sub_idxs[swap_idx] = random.choice(all_rem_idxs)
    # return sub_idxs
    sub_idxs[swap_idx] = random_other_idx(sub_idxs)
    return sub_idxs

  def get_subset_score(X_sub, Y_sub):
    # when X and Y are H U G E, we approximate
    if len(Y) > approximate_size:
      approximate_idxs = sub_select_random(X, Y, approximate_size)
      X_approx, Y_approx = X[approximate_idxs, :], Y[approximate_idxs]
      return knn_loss(make_knn(X_sub, Y_sub), X_approx, Y_approx)
    else:
      return knn_loss(make_knn(X_sub, Y_sub), X, Y)

  sub_idxs = sub_select_cluster(X, Y, n_samples, ae)
  X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]

  score_old = get_subset_score(X_sub, Y_sub)
  stuck_cnt = 0
  for i in range(n_samples * 100000):
    print ("i ", i)
    stuck_cnt += 1
    if stuck_cnt > n_samples:
      break
    swap_idx = i % n_samples
    new_sub_idxs = swap_one(sub_idxs, swap_idx)
    new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]

    score_new = get_subset_score(new_X_sub, new_Y_sub)
    
    if score_new < score_old:
      print ("new score ! ", score_new)
      X_sub, Y_sub = new_X_sub, new_Y_sub
      sub_idxs = new_sub_idxs
      score_old = score_new
      stuck_cnt = 0

  return sub_idxs

# =========================== HELPERS ==========================
def make_cl_knn(X, Y):
  clf = neighbors.KNeighborsClassifier(1+int(np.log(len(Y))))
  return clf.fit(X, Y)

def make_reg_knn(X, Y):
  reg = neighbors.KNeighborsRegressor(1+int(np.log(len(Y))))
  return reg.fit(X, Y)

def knn_cl_loss(clf, X, Y):
  pred = clf.predict(X)
  incorrect = (pred != Y)
  return sum(incorrect)

def knn_reg_loss(reg, X, Y):
  pred = reg.predict(X)
  return np.sum((pred - Y) ** 2)

