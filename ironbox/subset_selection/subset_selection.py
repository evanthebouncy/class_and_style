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
def sub_select_cluster(X, n_samples, ae):
  
  X_emb = ae.embed(X)
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=n_samples)
  kmeans = kmeans.fit(X_emb)
  
  cluster_labels = list(kmeans.predict(X_emb))
  # print (cluster_labels[:100])
  from sklearn.metrics import pairwise_distances_argmin_min
  closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_emb)
  return closest

# clustering in the latent space, equally distribute samples per label
def sub_select_label_cluster(X, Y, n_samples, ae):
  all_label_kinds = sorted(list(set(list(Y)))) # WTF IS THIS CODE LMAO
  label_idxs = [np.reshape(np.argwhere(Y == lab), (-1,)) for lab in all_label_kinds]

  ret_idxs = []
  for label_idx in label_idxs:
    X_lab, Y_lab = X[label_idx, :], Y[label_idx]
    sub_lab_idx = sub_select_cluster(X_lab, n_samples // len(all_label_kinds), ae)
    ret_idxs.append(label_idx[sub_lab_idx])

  ret_idxs = np.concatenate(ret_idxs)
  return ret_idxs

# optimize knn by annealing, respecting the labels
def sub_select_knn(X, Y, n_samples, ae, kind='classification', approximate_size=10000):
  assert kind in ['classification', 'regression']
  make_knn = make_cl_knn if kind == 'classification' else make_reg_knn
  knn_loss = knn_cl_loss if kind == 'classification' else knn_reg_loss

  data_size = len(X)

  # If things get too huge we use a subset-approximation of the whole set to be faster
  approximate_idxs = sub_select_random(X, Y, approximate_size)
  X_approx, Y_approx = X[approximate_idxs, :], Y[approximate_idxs]

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

  # when X and Y are H U G E, we approximate
  def get_subset_score(X_sub, Y_sub):
    if len(Y) > approximate_size:
      return knn_loss(make_knn(X_sub, Y_sub), X_approx, Y_approx)
    else:
      return knn_loss(make_knn(X_sub, Y_sub), X, Y)

  sub_idxs = sub_select_label_cluster(X, Y, n_samples, ae)
  X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]

  score_old = get_subset_score(X_sub, Y_sub)
  stuck_cnt = 0
  for i in range(n_samples * 100000):
    stuck_cnt += 1
    if stuck_cnt > n_samples:
      break
    swap_idx = i % n_samples
    new_sub_idxs = swap_one(sub_idxs, swap_idx)
    new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]

    score_new = get_subset_score(new_X_sub, new_Y_sub)
    
    if score_new < score_old:
      print ("[knn_anneal] iteration ", i, "stuck ", stuck_cnt, " new score! : ", score_new)
      X_sub, Y_sub = new_X_sub, new_Y_sub
      sub_idxs = new_sub_idxs
      score_old = score_new
      stuck_cnt = 0

  return sub_idxs



# optimize knn by DUELING
def sub_select_knn_dueling(X, Y, n_samples, ae, 
                           kind='classification', adv_size=None):
  # --------------- helper functions ----------------

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

  # when X and Y are H U G E, we approximate
  def get_subset_score(X_sub, Y_sub, X_adv, Y_adv):
    return knn_loss(make_knn(X_sub, Y_sub), X_adv, Y_adv)


  def minimize_sub(sub_idxs, X_adv, Y_adv):
    X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]
    score_old = get_subset_score(X_sub, Y_sub, X_adv, Y_adv)
    stuck_cnt = 0
    for i in range(n_samples * 100000):
      stuck_cnt += 1
      if stuck_cnt > n_samples:
        break
      swap_idx = i % n_samples
      new_sub_idxs = swap_one(sub_idxs, swap_idx)
      new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]

      score_new = get_subset_score(new_X_sub, new_Y_sub, X_adv, Y_adv)
      
      if score_new < score_old:
        X_sub, Y_sub = new_X_sub, new_Y_sub
        sub_idxs = new_sub_idxs
        score_old = score_new
        stuck_cnt = 0

    return sub_idxs

  def maximize_adv(adv_idxs, X_sub, Y_sub):
    X_adv, Y_adv = X[sub_idxs, :], Y[sub_idxs]
    score_old = get_subset_score(X_sub, Y_sub, X_adv, Y_adv)
    stuck_cnt = 0
    for i in range(n_samples * 100000):
      stuck_cnt += 1
      if stuck_cnt > n_samples:
        break
      swap_idx = i % n_samples
      new_adv_idxs = swap_one(adv_idxs, swap_idx)
      new_X_adv, new_Y_adv = X[new_adv_idxs, :], Y[new_adv_idxs]

      score_new = get_subset_score(X_sub, Y_sub, new_X_adv, new_Y_adv)
      
      if score_new > score_old:
        X_adv, Y_adv = new_X_adv, new_Y_adv
        adv_idxs = new_adv_idxs
        score_old = score_new
        stuck_cnt = 0

    return adv_idxs

  # ------------------- some initializations --------------------
  assert kind in ['classification', 'regression']
  make_knn = make_cl_knn if kind == 'classification' else make_reg_knn
  knn_loss = knn_cl_loss if kind == 'classification' else knn_reg_loss

  # set of adversaries
  data_size = len(Y)
  adv_size = adv_size if adv_size else min(n_samples, data_size)
  adv_idxs = sub_select_random(X, Y, adv_size)
  X_adv, Y_adv = X[adv_idxs, :], Y[adv_idxs]
  # set of candidates
  sub_idxs = sub_select_label_cluster(X, Y, n_samples, ae)

  # ---------------- run the min / max optimization ----------------
  all_scores = []
  all_sub_idxs = []
  all_adv_idxs = []
  while True:
    print ("[dueling] minimizing ...")
    sub_idxs = minimize_sub(sub_idxs, X_adv, Y_adv)
    X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs]
    print ("[dueling] maximizing ...")
    adv_idxs = maximize_adv(adv_idxs, X_sub, Y_sub)
    X_adv, Y_adv = X[adv_idxs, :], Y[adv_idxs]

    all_scores.append(get_subset_score(X_sub, Y_sub, X_adv, Y_adv))
    print ("scores ", len(all_scores), all_scores)
    all_sub_idxs.append(sub_idxs)
    all_adv_idxs.append(adv_idxs)

    # break if fixed point is reached
    if len(all_sub_idxs) > 2:
      last1 = all_sub_idxs[-1]
      last2 = all_sub_idxs[-2]
      n_updated_sub = sum(last1 != last2)
      print('updated sub idx ', n_updated_sub)
      if n_updated_sub == 0:
        break

    # break is variance dies down
    if len(all_scores) > 20:
      far_variance = np.var(all_scores[-20:-10])
      near_variance = np.var(all_scores[-10:])
      if near_variance > 0.98 * far_variance:
        break

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

