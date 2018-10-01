from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

# =========================== UTILITIES =============================
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

# compute the accuracy of using a classifier trained on the subset
def get_acc(clf, data_holder, X_t, Y_t):
  clf.learn(data_holder)
  return clf.evaluate((X_t,Y_t)) 


# plot the progress of accuracy over time
def plot_progress(clf_name, x_vals, diff_y_name_vals):
  cmap = plt.cm.get_cmap('hsv', 1+len(diff_y_name_vals))

  for ii, y_name_vals in enumerate(diff_y_name_vals):
    y_name, y_vals = y_name_vals
    plt.plot(x_vals, y_vals, color=cmap(ii), label=y_name)
  
  plt.legend()
  plt.savefig(clf_name + '_acc_progress.png')
  plt.clf()


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

def make_knn_regression(X, Y):
  reg_class = neighbors.KNeighborsRegressor(1+int(np.log(len(Y))))
  knn_reg = neighbors.KNeighborsRegressor(1+int(np.log(len(Y)))).fit(X, Y)
  def regress(X_new):
    return knn_cl.predict(X_new)
  return regress

def knn_loss(clf, X, Y):
  pred = clf(X)
  return np.sum((pred - Y) ** 2)

# ============= torch related utils =============

import torch
from torch.autograd import Variable

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = str(device) == "cuda"

def to_torch(x, req = False, cuda=use_cuda):
  dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
  x = Variable(torch.from_numpy(x).type(dtype), requires_grad = req)
  return x

def to_torch_int(x, req = False, cuda=use_cuda):
  dtype = torch.cuda.LongTensor if cuda else torch.LongTensor
  x = Variable(torch.from_numpy(x).type(dtype), requires_grad = req)
  return x

def to_torch_byte(x, req = False, cuda=use_cuda):
  dtype = torch.cuda.ByteTensor if cuda else torch.ByteTensor
  x = Variable(torch.from_numpy(x).type(dtype), requires_grad = req)
  return x
