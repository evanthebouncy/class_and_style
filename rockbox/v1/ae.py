import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random

class AE(nn.Module):

  def __init__(self, raw_data_dim):
    super(AE, self).__init__()

    self.raw_data_dim = raw_data_dim
    n_hidden_big = raw_data_dim // 4
    n_hidden = raw_data_dim // 8
    # for encding
    self.enc_fc1 = nn.Linear(raw_data_dim, n_hidden_big)
    self.enc_fc2 = nn.Linear(n_hidden_big, n_hidden)
    # for decoding
    self.dec_fc1 = nn.Linear(n_hidden, n_hidden_big)
    self.dec_fc2 = nn.Linear(n_hidden_big, raw_data_dim)

    self.opt = torch.optim.Adam(self.parameters(), lr=0.01)

  def enc(self, x):
    '''
    the x input here is encoded as batch x channel x L x L
    '''
    x = x.view(-1, self.raw_data_dim)
    x = F.relu(self.enc_fc1(x))
    x = F.sigmoid(self.enc_fc2(x))
    return x

  def dec(self, x):
    x = F.relu(self.dec_fc1(x))
    x = F.sigmoid(self.dec_fc2(x))
    x = x.view(-1, self.raw_data_dim)
    return x

  def forward(self, x):
    x = self.enc(x)
    x = self.dec(x)
    return x

  def xentropy_loss(self, target, prediction):
    prediction = torch.clamp(prediction, 1e-5, 1.0-1e-5)
    target_prob1 = target
    target_prob0 = 1.0 - target 
    log_prediction_prob1 = torch.log(prediction)
    log_prediction_prob0 = torch.log(1.0 - prediction)

    loss_value = -torch.sum(target_prob1 * log_prediction_prob1 + \
                            target_prob0 * log_prediction_prob0)
    return loss_value

class AEnet():
  # takes in channel variable n and width of image
  def __init__(self, sa_xform):
    self.name = "AEnet"
    self.sa_xform = sa_xform

  def torchify(self, x):
    x = Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor), 
        requires_grad = False)
    return x

  def learn_ae(self, X):
    ae = AE(self.sa_xform.length).cuda()
    X = np.array([self.sa_xform.sa_to_np(x) for x in X])
    
    losses = []

    while True:
      # load in the datas
      indices = sorted( random.sample(range(len(X)), 40) )
      X_sub = self.torchify(X[indices])

      X_emb = ae.enc(X_sub)
      X_rec = ae.dec(X_emb)
      loss = ae.xentropy_loss(X_sub, X_rec)

      # optimize 
      ae.opt.zero_grad()
      loss.backward()
      ae.opt.step()

      losses.append(loss.data.cpu().numpy())
      if len(losses) > 2000:
        far_loss = sum(losses[-2000:-1000]) 
        near_loss = sum(losses[-1000:])
        if near_loss > 0.98 * far_loss:
          break

      if len(losses) % len(X) == 0:
        print ("loss ", loss)

        # X_sub_np =  X_sub[0].data.cpu().view(10,10).numpy()
        # output_np = X_rec[0].data.cpu().view(10,10).numpy()
        # render_pic(X_sub_np, 'drawings/art_sam.png')
        # render_pic(output_np, 'drawings/art_rec.png')

    self.ae = ae
    return ae

  def embed(self, X):
    X = np.array([self.sa_xform.sa_to_np(x) for x in X])
    print (X)
    X = self.torchify(X)
    encoded = self.ae.enc(X)
    return encoded.data.cpu().numpy()

