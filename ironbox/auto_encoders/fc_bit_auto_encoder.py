import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


class AE(nn.Module):

  def __init__(self, n_feature, n_hidden):
    super(AE, self).__init__()
    n_hidden_big  = (n_feature + n_hidden) // 2
    self.n_feature = n_feature
    self.n_hidden = n_hidden
    # for encding
    self.enc_fc1 = nn.Linear(n_feature, n_hidden_big)
    self.enc_fc2 = nn.Linear(n_hidden_big, n_hidden)
    # for decoding
    self.dec_fc1 = nn.Linear(n_hidden, n_hidden_big)
    self.dec_fc2 = nn.Linear(n_hidden_big, n_feature)

    self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

  def enc(self, x):
    '''
    the x input here is encoded as batch x channel x L x L
    '''
    x = x.view(-1, self.n_feature)
    x = F.relu(self.enc_fc1(x))
    x = F.relu(self.enc_fc2(x))
    return x

  def dec(self, x):
    x = F.relu(self.dec_fc1(x))
    x = F.sigmoid(self.dec_fc2(x))
    # add a smol constant because I am paranoid
    x = x.view(-1, self.n_feature)
    return x

  def forward(self, x):
    x = self.enc(x)
    x = self.dec(x)
    return x

  def auto_enc_cost(self, x_tgt, x_rec):
    x_rec = torch.clamp(x_rec, 1e-5, 1.0-1e-5)
    x_tgt_prob1 = x_tgt
    x_tgt_prob0 = 1.0 - x_tgt 
    log_x_rec_prob1 = torch.log(x_rec)
    log_x_rec_prob0 = torch.log(1.0 - x_rec)

    cost_value = -torch.sum(x_tgt_prob1 * log_x_rec_prob1 + \
                            x_tgt_prob0 * log_x_rec_prob0)
    return cost_value

class FcAE():
  # takes in channel variable n and width of image
  def __init__(self, n_feature, n_hidden):
    self.name = "AEnet"
    self.n_feature = n_feature
    self.n_hidden = n_hidden

  def save(self, loc):
    torch.save(self.ae.state_dict(), loc)

  def load(self, loc):
    ae = AE(self.n_feature, self.n_hidden).cuda()
    ae.load_state_dict(torch.load(loc))
    self.ae = ae
    return ae

  def torchify(self, x):
    x = Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor), 
        requires_grad = False)
    return x

  def learn(self, X):
    ae = AE(self.n_feature, self.n_hidden).cuda()
    losses = []
    for i in range(99999999999):
      # load in the datas
      indices = sorted( random.sample(range(len(X)), 40) )
      X_sub = X[indices]
      # convert to proper torch forms
      X_sub = self.torchify(X_sub)

      # optimize 
      ae.opt.zero_grad()
      output = ae(X_sub)
      loss = ae.auto_enc_cost(X_sub, output)
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
        print ("loss ", len(losses), loss)

      #if i % 4000 == 0:
      #  print (i, loss)

      #  X_sub_np = X_sub[0].data.cpu().view(10,10).numpy()
      #  output_np = output[0].data.cpu().view(10,10).numpy()
      #  render_pic(X_sub_np, 'drawings/art_sam.png')
      #  render_pic(output_np, 'drawings/art_rec.png')

    self.ae = ae
    return ae

  def embed(self, X):
    X = np.array(X)
    X = self.torchify(X)
    encoded = self.ae.enc(X)
    return encoded.data.cpu().numpy()

