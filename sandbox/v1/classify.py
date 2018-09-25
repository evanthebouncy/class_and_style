from sklearn.linear_model import LogisticRegression
import numpy as np
from data import make_dataset, L
import matplotlib.pyplot as plt


# =========================== UTILITIES =============================
# feature extraction made easy when u can cheat :D
def feature(x):
  return x[:L]

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
def plot_progress(x_vals, diff_y_name_vals):
  cmap = plt.cm.get_cmap('hsv', 1+len(diff_y_name_vals))

  for ii, y_name_vals in enumerate(diff_y_name_vals):
    y_name, y_vals = y_name_vals
    plt.plot(x_vals, y_vals, color=cmap(ii), label=y_name)
  
  plt.legend()
  plt.savefig('acc_progress.png')
  plt.clf()

# ========================== SUBSET SELECTION SCHEMES =======================

# ------------------------------------- greedy hash --------------------------------
# greedily hash the feature space and try to get "distinct" elements
def sub_select_hash_feature(X_tr, Y_tr, n_sub):
  seen = set()
  X_sub, Y_sub = [], []
  for i in range(len(X_tr)):
    x,y = X_tr[i], Y_tr[i]
    feat = str(feature(x))
    if feat not in seen:
      X_sub.append(x)
      Y_sub.append(y)
      seen.add(feat)
  return X_sub[:n_sub], Y_sub[:n_sub]

# ----------------- cluster in ( raw_input / embedded feature ) space ----------------
def sub_select_cluster(X, Y, n_samples, embed):
  
  # either embed the X into feature space or leave it as is
  X_emb = [feature(x) for x in X] if embed else X
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
  return np.array(X_sub), np.array(Y_sub)

def make_knn(self, X, Y, embed=True):
  X_emb = self.embed(X) if embed else X
  from sklearn import neighbors
  # I HRD SOMEWHERE THAT USING k = LOG(len(DATA)) is good
  clf = neighbors.KNeighborsClassifier(1+int(np.log(len(Y))))
  # clf = neighbors.KNeighborsClassifier(1+int(np.log(len(Y))), weights='distance')
  knn_cl = clf.fit(X_emb, Y)
  def classify(X_new):
    X_new_emb = self.embed(X_new) if embed else X_new
    return knn_cl.predict(X_new_emb)
  return classify

def sub_select_knn_feature_sample(X, Y, n_samples):
  X_emb = [feature(x) for x in X]


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

# ======================== RUNNING THE EXPERIMENT ===========================
def run_subset_size(X_tr, Y_tr, X_t, Y_t, n_sub):
  ret_dict = dict()

  print ("===================== running for ", n_sub, "==========================")
  lregr = LRegr()
  lregr.learn((X_tr, Y_tr))
  all_result = lregr.evaluate((X_t,Y_t))
  print ( "all result : ", all_result)
  ret_dict['all'] = all_result

  X_rand, Y_rand = X_tr[:n_sub], Y_tr[:n_sub]
  lregr = LRegr()
  lregr.learn((X_rand, Y_rand))
  rand_result = lregr.evaluate((X_t,Y_t)) 
  print ( "rand_subset result : ", rand_result)
  ret_dict['rand'] = rand_result

  X_sub, Y_sub = sub_select_hash_feature(X_tr, Y_tr, n_sub)
  lregr = LRegr()
  lregr.learn((X_sub, Y_sub))
  hash_result = lregr.evaluate((X_t,Y_t))
  print ( "hash_subset result : ", hash_result )
  ret_dict['hash'] = hash_result

  X_sub, Y_sub = sub_select_cluster(X_tr, Y_tr, n_sub, False)
  lregr = LRegr()
  lregr.learn((X_sub, Y_sub))
  clus_raw = lregr.evaluate((X_t,Y_t))
  print ( "raw_cluster_subset result : ", clus_raw )
  ret_dict['clus_raw'] = clus_raw

  X_sub, Y_sub = sub_select_cluster(X_tr, Y_tr, n_sub, True)
  lregr = LRegr()
  lregr.learn((X_sub, Y_sub))
  clus_feat = lregr.evaluate((X_t,Y_t)) 
  print ( "feature_cluster_subset result : ", clus_feat )
  ret_dict['clus_feat'] = clus_feat

  return ret_dict

if __name__ == "__main__":
  n = 2000
  n_tr = n // 2

  X, Y = make_dataset(n)
  X_tr, Y_tr = X[:n_tr], Y[:n_tr]
  X_t, Y_t = X[n_tr:], Y[n_tr:]

  # some TSNE print for  all features and only class relevant features
  # tsne(X_tr, Y_tr, "all_feature")
  # X_emb = [feature(x) for x in X]
  # tsne(X_emb, Y, "klas_feature")
  
  sub_sizes = [10 * i for i in range(1, 31)]
  run_dicts = []
  for sub_size in sub_sizes:
    run_dict = run_subset_size(X_tr, Y_tr, X_t, Y_t, sub_size)
    run_dicts.append(run_dict)

  sub_scores = []
  keys = run_dicts[0].keys()
  for key in keys:
    key_vals = []
    for run_dict in run_dicts:
      key_vals.append(run_dict[key])
    sub_scores.append( (key, key_vals) )

  plot_progress(sub_sizes, sub_scores)








