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

# A simple fully connected neural nework representing a function that maps
# A input_dim input to k classes
class FCNet(nn.Module):

  # require input dimension, k_classes outputs
  # input-layer, input_dim // 2 hidden layer, k_class output layer
  def __init__(self, input_dim, k_classes):
    super(FCNet, self).__init__()
    self.name = "FCNet"
    self.fc = nn.Linear(input_dim, input_dim // 2)
    self.pred = nn.Linear(input_dim // 2, k_classes)
    self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

  def predict(self, x):
    x = F.relu(self.fc(x))
    x = F.log_softmax(self.pred(x), dim=1)
    return x

  # train until saturation
  # assume train_corpus is a data_holder that supports a 'get_sample(n_batch)'
  # function which return some samples
  def learn(self, train_corpus):

    losses = []
    while True:
      # terminate if no improvement
      if len(losses) > 2100:
        if losses[-1] < 1e-3:
          break
        near_data_loss = np.mean(losses[-1000:])
        far_data_loss  = np.mean(losses[-2000:-1000])
        # if average loss of last 1k iteration is greater than 99% of the
        # last last 1k iterations, stop too
        if near_data_loss > 0.99 * far_data_loss:
          break

      # randomly sample a batch of data
      X_batch, Y_batch = train_corpus.get_sample(40)
      # convert to proper torch forms
      X_batch = to_torch(X_batch, "float")
      Y_batch = to_torch(Y_batch, "int")

      # optimize 
      self.opt.zero_grad()
      output = self.predict(X_batch)
      loss = F.nll_loss(output, Y_batch)
      losses.append( loss.data.cpu().numpy() )
      loss.backward()
      self.opt.step()

  # evaluate the model on the test_corpus, here test_corpus is assumed to be simply
  # a pair of X, Y 
  def evaluate(self, test_corpus):
    X, Y = test_corpus
    X = to_torch(X, "float")
    label_pred = np.argmax(self.predict(X).data.cpu().numpy(), axis=1)
    return np.sum(label_pred != Y) / len(Y)

if __name__ == '__main__':
  fcnet = FCNet(100, 4)
  print ("hi")
