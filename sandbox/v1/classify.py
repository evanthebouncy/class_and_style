from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
import numpy as np
from data import make_dataset, L
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

# =========================== UTILITIES =============================
# feature extraction made easy when u can cheat :D
def get_class(x):
  return x[:L]

def get_common(x):
  return x[L:2*L] 

def get_style(x):
  return x[L:] 

# do a 2D visualization of the space of things
def tsne(embedded_X, labels=None, name=None):
  cl_colors = np.linspace(0, 1, len(set(labels))) if (labels is not None) else ['blue']
  from sklearn.manifold import TSNE
  X_tsne = TSNE(n_components=2).fit_transform(embedded_X)
  # import matplotlib
  # matplotlib.use("svg")
  x = [x[0] for x in X_tsne]
  y = [x[1] for x in X_tsne]
  colors = [cl_colors[lab] for lab in labels] if (labels is not None) else 'blue'
  plt.scatter(x, y, c=colors, alpha=0.5)
  name = name if name else ""
  plt.savefig('tsne_'+name+'.png')
  plt.clf()
  return X_tsne

# plot the progress of accuracy over time
def plot_progress(clf_name, x_vals, diff_y_name_vals):
  cmap = plt.cm.get_cmap('hsv', 1+len(diff_y_name_vals))

  for ii, y_name_vals in enumerate(diff_y_name_vals):
    y_name, y_vals = y_name_vals
    plt.plot(x_vals, y_vals, color=cmap(ii), label=y_name)
  
  plt.legend()
  plt.savefig(clf_name + '_acc_progress.png')
  plt.clf()

# ========================== SUBSET SELECTION SCHEMES =======================

# ------------------------------------- greedy hash --------------------------------
# greedily hash the class feature space and try to get "distinct" elements
def sub_select_hash_class(X_tr, Y_tr, n_sub):
  seen = set()
  X_sub, Y_sub = [], []
  for i in range(len(X_tr)):
    x,y = X_tr[i], Y_tr[i]
    feat = str(x)
    if feat not in seen:
      X_sub.append(x)
      Y_sub.append(y)
      seen.add(feat)
  return X_sub[:n_sub], Y_sub[:n_sub]

# ----------------- cluster in ( raw_input / embedded class ) space ----------------
def sub_select_cluster(X, Y, n_samples, embed=False):
  
  # either embed the X into class space or leave it as is
  X_emb = [get_class(x) for x in X] if embed else X
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=n_samples)
  kmeans = kmeans.fit(X_emb)
  
  cluster_labels = list(kmeans.predict(X_emb))
  # print (cluster_labels[:100])
  from sklearn.metrics import pairwise_distances_argmin_min
  closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_emb)
  counts = [cluster_labels.count(i) for i in range(n_samples)]
  
  # return [ (X[closest[i]], counts[i]) for i in range(n_samples) ], kmeans.score(X_emb)
  X_sub = []
  Y_sub = []
  for i in range(n_samples):
    X_sub.append(X[closest[i]])
    Y_sub.append(Y[closest[i]])
  return np.array(X_sub), np.array(Y_sub), closest

# ------------------------- do a knn classification -----------------------------------
def make_knn(X, Y):
  clf = neighbors.KNeighborsClassifier(1+int(np.log(len(Y))))
  knn_cl = clf.fit(X, Y)
  def classify(X_new):
    return knn_cl.predict(X_new)
  return classify

def knn_loss(clf, X, Y):
  pred = clf(X)
  incorrect = (pred != Y)
  return sum(incorrect)

def sub_select_knn_sample(X, Y, n_samples, embed):
  X_emb = np.array([get_class(x) for x in X]) if embed else X
  # make the sub and preserve the original information
  def make_sub():
    r_idxs = np.random.choice(np.arange(len(X)), n_samples, replace=False)
    X_emb_sub = X_emb[r_idxs, :]
    X_sub = X[r_idxs, :]
    return X_emb_sub, X_sub, Y[r_idxs]
  diff_subs = [make_sub() for _ in range(1000)]
  diff_knns = [make_knn(sub[0], sub[2]) for sub in diff_subs]
  diff_loss = [knn_loss(knn, X_emb, Y) for knn in diff_knns]
  best_knn_id = np.argmin(diff_loss)
  return diff_subs[best_knn_id][1], diff_subs[best_knn_id][2]

# ----------------------- optimize knn by annealing ---------------------
def sub_select_knn_anneal(X, Y, n_samples):
  def swap_one(sub_idxs):
    sub_idxs = [x for x in sub_idxs]
    sub_size = len(sub_idxs)
    all_rem_idxs = [iii for iii in range(len(X)) if iii not in sub_idxs]
    sub_idxs[random.choice(range(sub_size))] = random.choice(all_rem_idxs)
    return sub_idxs
  
  _, _, sub_idxs = sub_select_cluster(X, Y, n_samples)
  X_sub, Y_sub = X[sub_idxs, :], Y[sub_idxs] 
  score_old = knn_loss(make_knn(X_sub, Y_sub), X, Y)
  for i in range(1000):
    new_sub_idxs = swap_one(sub_idxs)
    new_X_sub, new_Y_sub = X[new_sub_idxs, :], Y[new_sub_idxs]
    score_new = knn_loss(make_knn(new_X_sub, new_Y_sub), X, Y)
    if score_new < score_old:
      X_sub, Y_sub = new_X_sub, new_Y_sub
      sub_idxs = new_sub_idxs
      score_old = score_new
  print ("score ", score_old)

  return X_sub, Y_sub

# ------------------------- maximize confusion in  -----------------------------------
def sub_select_knn_minmax_sample(X, Y, n_samples):
  X_emb = np.array([get_class(x) for x in X])
  
  # make the sub and preserve the original information
  def make_sub():
    r_idxs = np.random.choice(np.arange(len(X)), n_samples, replace=False)
    X_sub = X[r_idxs, :]
    Y_sub = Y[r_idxs]
    return X_sub, Y_sub

  # make a knn on the class space, and get its loss on whole data
  def class_loss (X_sub, Y_sub):
    X_sub_emb = np.array([get_class(x) for x in X_sub])
    knn = make_knn(X_sub_emb, Y_sub)
    return knn_loss(knn, X_emb, Y)

  # calculate the knn training loss on the sub data in the style space
  def style_loss(X_sub, Y_sub):
    X_sub_sty = np.array([get_style(x) for x in X_sub])
    sub_sub_idxs = np.random.choice(np.arange(len(X_sub)), len(X_sub)//2, replace=False)
    if (len(sub_sub_idxs) == 0): return 0
    sub_sub_sty  = X_sub_sty[sub_sub_idxs]
    sub_sub_Y    = Y_sub[sub_sub_idxs]
    knn_sub_sub = make_knn(sub_sub_sty, sub_sub_Y)
    return knn_loss(knn_sub_sub, X_sub_sty, Y_sub)

  # find the "best classifier" loss on the style space for a given subset
  def style_superstition(X_sub, Y_sub):
    best_train_loss = min([style_loss(X_sub, Y_sub) for _ in range(20)])
    superstition = abs(len(X_sub) // 2 - best_train_loss)
    return superstition

  subs = [make_sub() for _ in range(1000)]
  losses = [class_loss(*sub) + style_superstition(*sub) for sub in subs]
  best_id = np.argmin(losses)
  return subs[best_id]

# ===================== FC NN CLASSIFIER =====================
def to_torch(x, dtype, req = False):
  tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x

class FCNet(nn.Module):

  def __init__(self):
    super(FCNet, self).__init__()
    self.name = "FCNet"
    self.fc = nn.Linear(L * 2, 16)
    self.pred = nn.Linear(16, 2)
    self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

  def predict(self, x):
    x = F.relu(self.fc(x))
    x = F.log_softmax(self.pred(x), dim=1)
    return x

  def learn(self, train_corpus):
    X, Y = train_corpus
    X, Y = np.array(X), np.array(Y)

    losses = []
    while True:
      # load in the datas
      b_size = min(40, len(X) // 2)
      indices = sorted( random.sample(range(len(X)), b_size) )
      X_sub = X[indices]
      Y_sub = Y[indices]
      # convert to proper torch forms
      X_sub = to_torch(X_sub, "float")
      Y_sub = to_torch(Y_sub, "int")

      # optimize 
      self.opt.zero_grad()
      output = self.predict(X_sub)
      loss = F.nll_loss(output, Y_sub)
      losses.append( loss.data.cpu().numpy() )
      # terminate if no improvement
      if loss < 1e-3 or min(losses) < min(losses[-1000:]):
        break
      loss.backward()
      self.opt.step()

  def evaluate(self, test_corpus):
    test_data, test_label = test_corpus
    test_data = to_torch(test_data, "float")
    label_pred = np.argmax(self.predict(test_data).data.cpu().numpy(), axis=1)
    return np.sum(label_pred == test_label) / len(test_label)

# ========================= LOGISTIC REGRESSION CLASSIFIER ========================
class LRegr:

  def __init__(self):
    self.name = "LRegr"

  def learn(self, train_corpus):
    train_data, train_label = train_corpus
    logisticRegr = LogisticRegression(solver='sag')
    logisticRegr.fit(train_data, train_label)
    self.logisticRegr = logisticRegr

  def evaluate(self, test_corpus):
    test_data, test_label = test_corpus
    label_pred = self.logisticRegr.predict(test_data)
    return np.sum(label_pred == test_label) / len(test_label)

# ========================= KNN CLASSIFIER ========================
class KNN:

  def __init__(self):
    self.name = "KNN"

  def learn(self, train_corpus):
    train_data, train_label = train_corpus
    self.knn = make_knn(train_data, train_label) 

  def evaluate(self, test_corpus):
    test_data, test_label = test_corpus
    return 1.0 - (knn_loss(self.knn, test_data, test_label) / len(test_label))

def get_acc(clf_name, X_sub, Y_sub, X_t, Y_t, use_feat=False):
  X_sub = [get_class(x) for x in X_sub] if use_feat else X_sub
  X_t = [get_class(x) for x in X_t] if use_feat else X_t
  clfs = {
      'lrgr' : LRegr,
      'knn' : KNN,
      'fcnet' : lambda : FCNet().cuda(),
      }
  clf = clfs[clf_name]()
  clf.learn((X_sub, Y_sub))
  return clf.evaluate((X_t,Y_t)) 

def get_acc_feat(X_sub, Y_sub, X_t, Y_t):
  X_sub_feat = [get_class(x) for x in X_sub]
  X_t_feat = [get_class(x) for x in X_t]
  lregr = LRegr()
  lregr.learn((X_sub_feat, Y_sub))
  return lregr.evaluate((X_t_feat,Y_t)) 

# ======================== RUNNING THE EXPERIMENT ===========================
def run_subset_size(clf, X_tr, Y_tr, X_t, Y_t, n_sub):
  ret_dict = dict()

  print ("================ running for ", n_sub, clf, "====================")
  X_sub, Y_sub = X_tr, Y_tr
  all_result = get_acc(clf, X_sub, Y_sub, X_t, Y_t)
  print ( "all result : ", all_result)
  ret_dict['all'] = all_result

  X_sub, Y_sub = X_tr[:n_sub], Y_tr[:n_sub]
  rand_result = get_acc(clf, X_sub, Y_sub, X_t, Y_t)
  print ( "rand result : ", rand_result)
  ret_dict['rand'] = rand_result

  X_sub, Y_sub = sub_select_hash_class (X_tr, Y_tr, n_sub)
  hash_result = get_acc(clf, X_sub, Y_sub, X_t,Y_t)
  print ( "hash result : ", hash_result )
  ret_dict['hash'] = hash_result

  X_sub, Y_sub, _ = sub_select_cluster(X_tr, Y_tr, n_sub, False)
  clus_raw = get_acc(clf, X_sub, Y_sub, X_t,Y_t)
  print ( "cluster all result : ", clus_raw )
  ret_dict['clus_all'] = clus_raw

  X_sub, Y_sub, _ = sub_select_cluster(X_tr, Y_tr, n_sub, True)
  clus_cls = get_acc(clf, X_sub, Y_sub, X_t,Y_t)
  print ( "cluster class result : ", clus_cls )
  ret_dict['clus_cls'] = clus_cls

  X_sub, Y_sub = sub_select_knn_sample(X_tr, Y_tr, n_sub, False)
  knn_all = get_acc(clf, X_sub, Y_sub, X_t,Y_t)
  print ( "knn_all result : ", knn_all )
  ret_dict['knn_all'] = knn_all

  X_sub, Y_sub = sub_select_knn_sample(X_tr, Y_tr, n_sub, True)
  knn_cls = get_acc(clf, X_sub, Y_sub, X_t, Y_t)
  print ( "knn_cls result : ", knn_cls )
  ret_dict['knn_cls'] = knn_cls

  X_sub, Y_sub = sub_select_knn_anneal(X, Y, n_sub)
  knn_anneal = get_acc(clf, X_sub, Y_sub, X_t, Y_t)
  print ( "knn_anneal result : ", knn_anneal )
  ret_dict['knn_anneal'] = knn_anneal

  return ret_dict

if __name__ == "__main__":
  n = 2000
  n_tr = n // 2

  X, Y = make_dataset(n)
  X_tr, Y_tr = X[:n_tr], Y[:n_tr]
  X_t, Y_t = X[n_tr:], Y[n_tr:]

  # some TSNE print for  all features and only class relevant features
  tsne(X_tr, Y_tr, "all_feature")
  X_emb = [get_class(x) for x in X]
  tsne(X_emb, Y, "class_feature")
  X_style = [get_style(x) for x in X]
  tsne(X_style, Y, "style_feature")
  X_common = [get_common(x) for x in X]
  tsne(X_common, Y, "common_feature")

  clf_name = 'fcnet'
  
  sub_sizes = [10 * i for i in range(1, 21)]
  run_dicts = []
  for sub_size in sub_sizes:
    run_dict = run_subset_size(clf_name, X_tr, Y_tr, X_t, Y_t, sub_size)
    run_dicts.append(run_dict)

  sub_scores = []
  keys = run_dicts[0].keys()
  for key in keys:
    key_vals = []
    for run_dict in run_dicts:
      key_vals.append(run_dict[key])
    sub_scores.append( (key, key_vals) )

  plot_progress(clf_name, sub_sizes, sub_scores)








