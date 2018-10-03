import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

# for drawing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def to_torch(x, dtype, req = False):
  tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x

class AE(nn.Module):
  def __init__(self):
    super(AE, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
      nn.ReLU(True),
      nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
      nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
      nn.ReLU(True),
      nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
      nn.Sigmoid(),
    )
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
      nn.ReLU(True),
      nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
      nn.Tanh()
    )
    self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

  def save(self, loc):
    torch.save(net.state_dict(), loc) 

class CnnAE():
  # takes in channel variable n and width of image
  def __init__(self, n_channel, w_img):
    self.name = "CnnAE"
    self.n_channel, self.w_img = n_channel, w_img

  def save(self, loc):
    torch.save(self.ae.state_dict(), loc)

  def load(self, loc):
    ae = AE().cuda()
    ae.load_state_dict(torch.load(loc))
    self.ae = ae

  def torchify(self, X):
    return to_torch(X, "float").view(-1, self.n_channel, self.w_img, self.w_img)

  def learn(self, X):
    ae = AE().cuda()
    losses = []
    for i in range(99999999999):
      # load in the datas
      indices = sorted( random.sample(range(len(X)), 40) )
      X_sub = np.array([X[i] for i in indices])
      # convert to proper torch forms
      X_sub = self.torchify(X_sub)

      # optimize 
      ae.opt.zero_grad()
      output = ae(X_sub)
      loss_fun = nn.MSELoss()
      loss = loss_fun(output, X_sub)
      loss.backward()
      ae.opt.step()

      losses.append(loss.data.cpu().numpy())

      # some breaking conditions
      if len(losses) > 20000:
        first_loss = losses[0]
        last_loss = losses[-1]
        if last_loss * 10000 < first_loss:
          break

        far_loss = sum(losses[-20000:-10000]) 
        near_loss = sum(losses[-10000:])
        if near_loss > 0.98 * far_loss:
          break

      if len(losses) % len(X) == 0:
        print ("loss ", loss)



      if i % 4000 == 0:
        print (X_sub.size())
        print (output.size())
        print (far_loss, near_loss)

        X_sub_np = X_sub[0].data.cpu().view(28,28).numpy()
        output_np = output[0].data.cpu().view(28,28).numpy()
        plt.imsave('test1.png', X_sub_np)
        plt.imsave('test2.png', output_np)

    self.ae = ae
    return ae

  def embed(self, X):
    X = np.array(X)
    X = self.torchify(X)
    encoded = self.ae.encoder(X).view(-1, 8*2*2)
    return encoded.data.cpu().numpy()

