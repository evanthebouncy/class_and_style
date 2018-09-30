from sklearn import neighbors
import numpy as np
from data import make_dataset, L

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import make_knn, knn_loss

# ====================== DATA HOLDER WITH PORPORTIONAL SAMPLING =============
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
  def get_sample(self, n):
    sample_idx = np.random.choice(range(len(self.weights)),
                                     size=n,
                                     p=self.weights)
    return self.X[sample_idx, :], self.Y[sample_idx]

# ===================== FC NN REGRESSOR =====================
def to_torch(x, dtype, req = False):
  tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x

class FCNet(nn.Module):

  def __init__(self):
    super(FCNet, self).__init__()
    self.name = "FCNet"
    self.fc = nn.Linear(L * 2, 16)
    self.pred = nn.Linear(16, 1)
    self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

  def predict(self, x):
    x = F.relu(self.fc(x))
    x = self.pred(x).view((-1,))
    return x

  def learn(self, train_corpus):

    def stop_criteria(losses):
      if len(losses) < 3000:
        return False
      else:
        loss_recent = losses[-1000:]
        loss_faraway = losses[-2000:-1000]
        assert len(loss_recent) == len(loss_faraway)
        return sum(loss_recent) < 0.97 * sum(loss_faraway)


    losses = []
    while True:
      X_sub, Y_sub = train_corpus.get_sample(40)
      # convert to proper torch forms
      X_sub = to_torch(X_sub, "float")
      Y_sub = to_torch(Y_sub, "float")

      # optimize 
      self.opt.zero_grad()
      output = self.predict(X_sub)
      loss = F.mse_loss(output, Y_sub)
      losses.append(loss)
      if stop_criteria(losses):
        break
      loss.backward()
      self.opt.step()

  def evaluate(self, test_corpus):
    test_data, test_label = test_corpus
    test_data = to_torch(test_data, "float")
    pred = self.predict(test_data).data.cpu().numpy()
    print ("evaluating")
    return np.mean((pred - test_label) ** 2)

# ========================= KNN CLASSIFIER ========================
class KNN:

  def __init__(self):
    self.name = "KNN"

  def learn(self, train_corpus):
    train_data, train_label = train_corpus.get_all_set()
    self.knn = make_knn(train_data, train_label) 

  def evaluate(self, test_corpus):
    test_data, test_label = test_corpus
    return 1.0 - (knn_loss(self.knn, test_data, test_label) / len(test_label))


