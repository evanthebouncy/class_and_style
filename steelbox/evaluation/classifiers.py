from sklearn.linear_model import LogisticRegression
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# ===================== FC NN CLASSIFIER =====================
def to_torch(x, dtype, req = False):
  tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x

class FCNet(nn.Module):

  def __init__(self, in_dim, out_dim):
    super(FCNet, self).__init__()
    self.name = "FCNet"
    h_dim = (in_dim + out_dim) // 2
    self.fc = nn.Linear(in_dim, h_dim)
    self.pred = nn.Linear(h_dim, out_dim)
    self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

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

    losses = []
    while True:
      X_sub, Y_sub = train_corpus.get_sample(40)
      loss = self.learn_once(X_sub, Y_sub)
      losses.append( loss.data.cpu().numpy() )
      # terminate if no improvement
      if loss < 1e-3 or min(losses) < min(losses[-1000:]):
        print (len(losses), " learn iter ")
        break

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
    train_data, train_label = train_corpus.get_all_set_weighted()
    logisticRegr = LogisticRegression(solver='sag')
    logisticRegr.fit(train_data, train_label)
    self.logisticRegr = logisticRegr

  def evaluate(self, test_corpus):
    test_data, test_label = test_corpus
    label_pred = self.logisticRegr.predict(test_data)
    return np.sum(label_pred == test_label) / len(test_label)


