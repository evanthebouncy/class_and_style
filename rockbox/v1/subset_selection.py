from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import neighbors
import numpy as np
import random
from copy import deepcopy

# ========================= CLASSIFICATION ==============================
def make_cl_knn(X, Y):
  clf = neighbors.KNeighborsClassifier(1+int(np.log(len(Y))))
  knn_cl = clf.fit(X, Y)
  return knn_cl

def knn_cl_loss(clf, X, Y):
  pred = clf.predict(X)
  incorrect = (pred != Y)
  return sum(incorrect)

# ========================= REGRESSION ===============================

def make_reg_knn(X, Y):
  reg_class = neighbors.KNeighborsRegressor(1+int(np.log(len(Y))))
  knn_reg = neighbors.KNeighborsRegressor(1+int(np.log(len(Y)))).fit(X, Y)
  return knn_reg

def knn_reg_loss(reg, X, Y):
  pred = reg.predict(X)
  return np.sum((pred - Y) ** 2)

# =================== DATA HOLDER WITH PORPORTIONAL SAMPLING =============
class DataHolder:
  def __init__(self, X, Y, weights):
    self.X = np.array(X)
    self.Y = np.array(Y)
    weights = np.array(weights)
    weights = weights / np.sum(weights) # normalized weights D: :D :D
    self.weights = weights

  # return the unweighted dataset, uniform weighted
  def get_all_set(self):
    return self.X, self.Y

  # return the entire dataset, properly weighted by duplications
  def get_all_set_weighted(self):
    min_prob = np.min(self.weights)
    multiple = (1 / min_prob)
    repeats = (multiple * self.weights).astype(int)
    retX = np.repeat(self.X, repeats, axis=0)
    retY = np.repeat(self.Y, repeats, axis=0)
    return retX, retY

  # return a sample of dataset smapled from the weight distribution
  def sample(self, n):
    sample_idx = np.random.choice(range(len(self.weights)),
                                     size=n,
                                     p=self.weights)
    return self.X[sample_idx, :], self.Y[sample_idx]


# ===================== SUBSET SELECTION ALGORITHMS ===================

# ----------------- cluster in ( raw_input / embedded class ) space ----------------
def sub_select_cluster(X, Y, n_samples, embed=False, weighted=True):
  
  # either embed the X into class space or leave it as is
  X_emb = [get_class(x) for x in X] if embed else X
  kmeans = KMeans(n_clusters=n_samples)
  kmeans = kmeans.fit(X_emb)
  
  cluster_labels = list(kmeans.predict(X_emb))
  # print (cluster_labels[:100])
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
def sub_select_knn_anneal(kind, X, Y, n_samples, weighted=True):
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
      stuck_cnt = stuck_cnt // 2
  return sub_idxs
  # # print ("score ", score_old)

  # # -------------- get the proper weight distribution ---------------
  # knn = make_knn(X_sub, Y_sub)
  # neighbors = knn.kneighbors(X, return_distance=False)                              
  # neighbors_flat = np.reshape(neighbors, (-1,))
  # from collections import Counter
  # cnt = Counter(neighbors_flat)
  # frequency = [cnt[i] for i in range(n_samples)] if weighted else n_samples * [1.0]
  # return DataHolder(X_sub, Y_sub, frequency)

# =================== WRAPPER SELECTOR =======================
# this is simply a wrapper around an auto_encoder
class Selector:
  def __init__(self, auto_encoder):
    self.auto_encoder = auto_encoder
    assert 'embed' in dir(auto_encoder)

  def select_subset(self, kind, X, Y, n_samples, method='knn'):
    X_emb = self.auto_encoder.embed(X)
    if method == 'knn':
      return sub_select_knn_anneal(kind, X_emb, Y, n_samples, True)




