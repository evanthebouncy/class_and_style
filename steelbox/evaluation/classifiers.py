from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

# ===================== FC NN CLASSIFIER =====================
def to_torch(x, dtype, req = False):
  tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x

def learn_loop(self, train_corpus):

    loss_th, loss_iter_bnd, stop_time = self.stop_criteria

    losses = []
    time_s = time.time()

    while True:
        # break on time 
        if time.time() - time_s > stop_time:
            self.term = 'timeout'
            break

        X_sub, Y_sub = train_corpus.get_sample(40)
        loss = self.learn_once(X_sub, Y_sub)
        losses.append( loss.data.cpu().numpy() + 1e-5 )
        # terminate if no improvement
        if len(losses) > 2 * loss_iter_bnd:
            last_loss =            np.mean(losses[-loss_iter_bnd:])
            last_last_loss = np.mean(losses[-2 * loss_iter_bnd:-loss_iter_bnd])
            if abs(last_loss - last_last_loss) / last_last_loss < loss_th:
                self.term = 'saturation'
                break


class FCNet(nn.Module):

    def __init__(self, in_dim, out_dim, stop_criteria = (0.01, 1000, 120)):
        super(FCNet, self).__init__()
        self.name = "FCNet"
        h_dim = (in_dim + out_dim) // 2
        self.fc = nn.Linear(in_dim, h_dim)
        self.pred = nn.Linear(h_dim, out_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

        self.stop_criteria = stop_criteria

    def predict(self, x):
        x = F.relu(self.fc(x))
        x = F.log_softmax(self.pred(x), dim=1)
        return x

    def learn_once(self, X_sub, Y_sub):
        X_sub = to_torch(X_sub, "float")
        Y_sub = to_torch(Y_sub, "int")

        # optimize 
        self.opt.zero_grad()
        output = self.predict(X_sub)
        loss = F.nll_loss(output, Y_sub)
        loss.backward()
        self.opt.step()

        return loss

    def learn(self, train_corpus):

        learn_loop(self, train_corpus)

    def evaluate(self, test_corpus):
        test_data, test_label = test_corpus
        test_data = to_torch(test_data, "float")
        label_pred = np.argmax(self.predict(test_data).data.cpu().numpy(), axis=1)
        return np.sum(label_pred == test_label) / len(test_label)

# ============== Convulutional Nural Network For Image Classification ===============
class CNN1(nn.Module):

    # init with (channel, height, width) and out_dim for classiication
    def __init__(self, ch_h_w, out_dim, stop_criteria = (0.01, 1000, 120)):
        super(CNN1, self).__init__()
        self.name = "CNN1"

        self.ch, self.h, self.w = ch_h_w

        self.conv1 = nn.Conv2d(self.ch, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, out_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

        self.stop_criteria = stop_criteria
  
    def predict(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def learn_once(self, X_sub, Y_sub):
        X_sub = to_torch(X_sub, "float").view(-1, self.ch, self.h, self.w)
        Y_sub = to_torch(Y_sub, "int")
  
        # optimize 
        self.opt.zero_grad()
        output = self.predict(X_sub)
        loss = F.nll_loss(output, Y_sub)
        loss.backward()
        self.opt.step()
  
        return loss
  
    def learn(self, train_corpus):
        learn_loop(self, train_corpus)
  
    def evaluate(self, test_corpus):
        test_data, test_label = test_corpus
        test_data = to_torch(test_data, "float").view(-1, self.ch, self.h, self.w)
        label_pred = np.argmax(self.predict(test_data).data.cpu().numpy(), axis=1)
        return np.sum(label_pred == test_label) / len(test_label)
  


# ========================= LOGISTIC REGRESSION CLASSIFIER ========================
class LGR(nn.Module):

  def __init__(self, in_dim, out_dim, stop_criteria = (0.01, 1000, 120)):
      super(LGR, self).__init__()
      self.name = "LGR"
      self.pred = nn.Linear(in_dim, out_dim)
      self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

      self.stop_criteria = stop_criteria

  def predict(self, x):
      x = F.log_softmax(self.pred(x), dim=1)
      return x

  def learn_once(self, X_sub, Y_sub):
      X_sub = to_torch(X_sub, "float")
      Y_sub = to_torch(Y_sub, "int")

      # optimize 
      self.opt.zero_grad()
      output = self.predict(X_sub)
      loss = F.nll_loss(output, Y_sub)
      loss.backward()
      self.opt.step()

      return loss

  def learn(self, train_corpus):
      learn_loop(self, train_corpus)

  def evaluate(self, test_corpus):
      test_data, test_label = test_corpus
      test_data = to_torch(test_data, "float")
      label_pred = np.argmax(self.predict(test_data).data.cpu().numpy(), axis=1)
      return np.sum(label_pred == test_label) / len(test_label)


# ======================== K NAEREST NEIGBHR CLASSIFIER ==========================
class EKNN:

    def __init__(self):
        self.name = "KNN"

    def learn(self, train_corpus):
        X, Y = train_corpus.X, train_corpus.Y
        K = 1
        clf = neighbors.KNeighborsClassifier(K)
        self.knn = clf.fit(X, Y)

    def evaluate(self, test_corpus):
        X_t, Y_t = test_corpus
        Y_pred = self.knn.predict(X_t)
        return np.sum(Y_pred == Y_t) / len(Y_t)

# =================== SVM CLASSIFIER =====================
class SVM:

    def __init__(self, kern):
        self.kern = kern
        self.name = 'SVM_' + kern

    def learn(self, train_corpus):
        from sklearn.svm import SVC
        X, Y = train_corpus.X, train_corpus.Y
        clf = SVC(kernel = self.kern, gamma='auto')
        self.svm = clf.fit(X, Y) 

    def evaluate(self, test_corpus):
        X_t, Y_t = test_corpus
        Y_pred = self.svm.predict(X_t)
        return np.sum(Y_pred == Y_t) / len(Y_t)

# ==================== LINEAR SGD CLASSIFIER ======================
class SGD:

    def __init__(self):
        self.name = 'Linear_SGD'

    def learn(self, train_corpus):
        from sklearn import linear_model
        X, Y = train_corpus.X, train_corpus.Y
        clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
        self.sgd = clf.fit(X, Y)

    def evaluate(self, test_corpus):
        X_t, Y_t = test_corpus
        Y_pred = self.sgd.predict(X_t)
        return np.sum(Y_pred == Y_t) / len(Y_t)

# ================ DECISION TREE CLASSIFIER ==================
class DTREE:

    def __init__(self):
        self.name = 'DTREE'

    def learn(self, train_corpus):
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        X, Y = train_corpus.X, train_corpus.Y
        self.dtree = clf.fit(X, Y)

    def evaluate(self, test_corpus):
        X_t, Y_t = test_corpus
        Y_pred = self.dtree.predict(X_t)
        return np.sum(Y_pred == Y_t) / len(Y_t)

# ================= QDA ===============
class QDA:

    def __init__(self):
        self.name = 'QDA'

    def learn(self, train_corpus):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        clf = QuadraticDiscriminantAnalysis()
        X, Y = train_corpus.X, train_corpus.Y
        self.qda = clf.fit(X, Y)

    def evaluate(self, test_corpus):
        X_t, Y_t = test_corpus
        Y_pred = self.qda.predict(X_t)
        return np.sum(Y_pred == Y_t) / len(Y_t)

# ============ RANDOM FOREST ===============
class RFOREST:

    def __init__(self):
        self.name = 'RFOREST'

    def learn(self, train_corpus):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100)
        X, Y = train_corpus.X, train_corpus.Y
        self.forest = clf.fit(X, Y)

    def evaluate(self, test_corpus):
        X_t, Y_t = test_corpus
        Y_pred = self.forest.predict(X_t)
        return np.sum(Y_pred == Y_t) / len(Y_t)

# ============= GP ===============
class GP:

    def __init__(self):
        self.name = 'GP'

    def learn(self, train_corpus):
        from sklearn.gaussian_process import GaussianProcessClassifier

        clf = GaussianProcessClassifier()
        X, Y = train_corpus.X, train_corpus.Y
        self.gp = clf.fit(X, Y)

    def evaluate(self, test_corpus):
        X_t, Y_t = test_corpus
        Y_pred = self.gp.predict(X_t)
        return np.sum(Y_pred == Y_t) / len(Y_t)
