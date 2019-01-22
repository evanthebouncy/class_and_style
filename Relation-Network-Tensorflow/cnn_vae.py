import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
from visualizer import getdata

Conv_W = 3
CC, LL, WW = 32, 6, 6

def to_torch(x, dtype="float", req = False):
  tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x

class ECLVR(nn.Module):
    def __init__(self, n_chan):
        super(ECLVR, self).__init__()
        # 1 channel input to 2 channel output of first time print and written
        self.conv1 = nn.Conv2d(n_chan, 8, Conv_W)
        self.conv2 = nn.Conv2d(8, 16, Conv_W)
        self.conv3 = nn.Conv2d(16, 32, Conv_W)

        self.dense_enc = nn.Linear(CC*LL*WW + 11, 100)

        # variational bits
        self.fc_mu = nn.Linear(100, 32)
        self.fc_logvar = nn.Linear(100, 32)
        self.fc_dec = nn.Linear(32, 100)
        
        self.dense_dec = nn.Linear(100, CC*LL*WW)
        self.dense_dec_qry1 = nn.Linear(100, 6)
        self.dense_dec_qry2 = nn.Linear(100, 5)

        self.deconv3 = torch.nn.ConvTranspose2d(32, 16, Conv_W)
        self.deconv2 = torch.nn.ConvTranspose2d(16, 8, Conv_W)
        self.deconv1 = torch.nn.ConvTranspose2d(8, n_chan, Conv_W)

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x, x_qry):
        # conv1
        x = F.relu(self.conv1(x))
        size1 = x.size()
        x, idx1 = self.pool(x)

        # conv2
        x = F.relu(self.conv2(x))
        size2 = x.size()
        x, idx2 = self.pool(x)
        #print('size2=',size2)

        # conv3
        x = F.relu(self.conv3(x))
        size3 = x.size()
        x, idx3 = self.pool(x)

        # =================================================
        # reached the middle layer, some dense
        x = x.view(-1, CC*LL*WW)
        # add in the qury bits
        x = torch.cat((x, x_qry), dim=1)

        x = torch.relu(self.dense_enc(x))

        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        x = self.reparameterize(mu, logvar)    
        x = F.relu(self.fc_dec(x))

        x_qry_rec1 = torch.softmax(self.dense_dec_qry1(x), dim=1)
        x_qry_rec2 = torch.softmax(self.dense_dec_qry2(x), dim=1)
        x_qry_rec = torch.cat((x_qry_rec1, x_qry_rec2), dim=1)

        x = F.relu(self.dense_dec(x))
        x = x.view(-1, CC, LL, WW)
        # ================================================= 

        # deconv3
        x = self.unpool(x, idx3, size3)
        x = F.relu(self.deconv3(x))

        # deconv2
        x = self.unpool(x, idx2, size2)
        x = F.relu(self.deconv2(x))

        # deconv1
        x = self.unpool(x, idx1, size1)
        x = torch.sigmoid(self.deconv1(x))
        return x, x_qry_rec, mu, logvar

    def embed(self, x, x_qry):
        x_rec, x_qry_rec, mu, logvar = self(x, x_qry)
        return mu

    # ================== VAE MAGIC =================
 
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
 
    def kld_loss(self, mu, logvar):

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return KLD



    def learn_once(self, imgs, qrys):
        img_rec, qry_rec, mu, logvar = self(imgs, qrys)
        
        self.opt.zero_grad()

        # compute all the cost LOL
        # rec_loss = ((img_rec - imgs) ** 2).mean()
        L2L = torch.sum((imgs - img_rec) ** 2)
        KLD = self.kld_loss(mu, logvar)
        BCE = F.binary_cross_entropy(qry_rec, qrys)

        # if (np.random.random() < 0.01) :
        #     print ("LOSSSSSSSSSSSSS")
        #     print (L2L, "IMG")
        #     print (KLD, "REGULARIZE")
        #     print (BCE, "BIT ERROR")

        # loss = L2L + BCE + 0.1 * KLD
        loss = L2L + 500 * BCE + 0.1 * KLD

        loss.backward()

        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)

        self.opt.step()
        return loss

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))

    def learn(self, X, Q, learn_iter = 1000):

        losses = []
        # for i in range(99999999999):
        for i in range(learn_iter):
            # load in the datas
            indices = sorted(random.sample(range(len(X)), 40))
            # indices = list(range(40))
            X_sub = X[indices]
            # convert to proper torch forms
            Q_sub = Q[indices]

            X_sub = to_torch(X_sub)
            Q_sub = to_torch(Q_sub)

            # optimize
            losses.append(self.learn_once(X_sub, Q_sub).data.cpu().numpy())

            if i % 1000 == 0:
                print(i, losses[len(losses)-1])
                img_orig = X_sub[0].detach().cpu().numpy()
                img_rec = self(X_sub, Q_sub)[0][0].detach().cpu().numpy()
                qry_rec = self(X_sub, Q_sub)[1][0].detach().cpu().numpy()

                print ("qry first part ")
                print (Q_sub[0][:6])
                print (qry_rec [:6])
                print ("qry second part ")
                print (Q_sub[0][6:])
                print (qry_rec [6:])
                draw(img_orig, 'orig_img.png')
                draw(img_rec, 'rec_img.png')

def draw(WxHxC, name):
    to_draw = np.transpose(WxHxC, (1,2,0))

    import matplotlib.pyplot as plt
    plt.imshow(to_draw)
    plt.savefig(name)

