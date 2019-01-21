import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
from visualizer import getdata

Conv_W = 3
CC, LL, WW = 8, 60, 60

def to_torch(x, dtype="float", req = False):
  tor_type = torch.cuda.LongTensor if dtype == "int" else torch.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x

class CNN(nn.Module):
    def __init__(self, n_chan):
        super(CNN, self).__init__()
        # 1 channel input to 2 channel output of first time print and written
        self.conv1 = nn.Conv2d(n_chan, 8, Conv_W)
        self.conv2 = nn.Conv2d(8, 8, Conv_W)
        # self.conv3 = nn.Conv2d(8, 32, Conv_W)

        self.dense_enc = nn.Linear(CC*LL*WW, 32)
        self.dense_dec = nn.Linear(32, CC*LL*WW)

        # self.deconv3 = torch.nn.ConvTranspose2d(32, 8, Conv_W)
        self.deconv2 = torch.nn.ConvTranspose2d(8, 8, Conv_W)
        self.deconv1 = torch.nn.ConvTranspose2d(8, n_chan, Conv_W)

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        # conv1
        x = F.relu(self.conv1(x))
        size1 = x.size()
        # x, idx1 = self.pool(x)

        # conv2
        x = F.relu(self.conv2(x))
        size2 = x.size()
        # x, idx2 = self.pool(x)
        #print('size2=',size2)

        # reached the middle layer, some dense
        x = x.view(-1, CC*LL*WW)
        #x = x.view(-1,8*60*60)
        x = torch.tanh(self.dense_enc(x))

        x = F.relu(self.dense_dec(x))
        x = x.view(-1, CC, LL, WW)

        # deconv2
        # x = self.unpool(x, idx2, size2)
        x = F.relu(self.deconv2(x))

        # deconv1
        # x = self.unpool(x, idx1, size1)
        x = torch.tanh(self.deconv1(x))
        return x

    #def decode(self, x):


    def embed(self, x):
        x = to_torch(x)
        x = F.relu(self.conv1(x))
        size1 = x.size()
        x = F.relu(self.conv2(x))
        size2 = x.size()
        x = x.view(-1, CC*LL*WW)
        x = torch.tanh(self.dense_enc(x))
        return x

    def learn_once(self, imgs):
        img_rec = self(imgs)

        self.opt.zero_grad()
        # compute all the cost LOL
        loss = ((img_rec - imgs) ** 2).mean()
        loss.backward()

        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)

        self.opt.step()
        return loss

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))

    def learn(self,X,learn_iter = 1000):

        losses = []
        # for i in range(99999999999):
        for i in range(learn_iter):
            # load in the datas
            indices = sorted(random.sample(range(len(X)), 40))
            X_sub = X[indices]
            # convert to proper torch forms
            X_sub = to_torch(X_sub)

            # optimize

            losses.append(self.learn_once(X_sub).data.cpu().numpy())

            # # some breaking conditions
            # if len(losses) > 20000:
            #     first_loss = losses[0]
            #     last_loss = losses[-1]
            #     if last_loss * 10000 < first_loss:
            #         break

            #     far_loss = sum(losses[-20000:-10000])
            #     near_loss = sum(losses[-10000:])
            #     if near_loss > 0.98 * far_loss:
            #         break

            # if len(losses) % len(X) == 0:
            #     print ("loss ", len(losses), loss)
            if i % 4 == 0:
                print(i, losses[len(losses)-1])



