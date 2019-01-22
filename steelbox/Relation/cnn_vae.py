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

class CNN(nn.Module):
    def __init__(self, n_chan):
        super(CNN, self).__init__()
        # 1 channel input to 2 channel output of first time print and written
        self.conv1 = nn.Conv2d(n_chan, 8, Conv_W)
        self.conv2 = nn.Conv2d(8, 16, Conv_W)
        self.conv3 = nn.Conv2d(16, 32, Conv_W)

        self.dense_enc = nn.Linear(CC*LL*WW, 100)

        # variational bits
        self.fc_mu = nn.Linear(100, 32)
        self.fc_logvar = nn.Linear(100, 32)

        self.dense_dec = nn.Linear(100, CC*LL*WW)

        self.deconv3 = torch.nn.ConvTranspose2d(32, 16, Conv_W)
        self.deconv2 = torch.nn.ConvTranspose2d(16, 8, Conv_W)
        self.deconv1 = torch.nn.ConvTranspose2d(8, n_chan, Conv_W)

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
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
        #x = x.view(-1,8*60*60)
        x = torch.relu(self.dense_enc(x))

        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        x = self.reparameterize(mu, logvar)    
        

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

    # VAE MAGIC =================
 
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
 
 45 
 46     def forward(self, x):
 47         mu, logvar = self.encode(x)
 48         z = self.reparameterize(mu, logvar)
 49         return self.decode(z), mu, logvar
 50 
 51     def ae(self, x):
 52         mu, logvar = self.encode(x)
 53         return self.decode(mu)
 54 
 55     # Reconstruction + KL divergence losses summed over all elements and batch
 56     def loss_function(self, recon_x, x, mu, logvar):
 57         BCE = lambda : F.binary_cross_entropy(recon_x, x)
 58         L2L = lambda : torch.sum((recon_x - x) ** 2)
 59 
 60         REC = BCE() if self.loss_type == 'xentropy' else L2L()
 61 
 62 
 63         # see Appendix B from VAE paper:
 64         # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
 65         # https://arxiv.org/abs/1312.6114
 66         # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
 67         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
 68 
 69         return REC + KLD



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
            # indices = list(range(40))
            X_sub = X[indices]
            # convert to proper torch forms
            X_sub = to_torch(X_sub)

            # optimize
            losses.append(self.learn_once(X_sub).data.cpu().numpy())

            if i % 1000 == 0:
                print(i, losses[len(losses)-1])
                img_orig = X_sub[0].detach().cpu().numpy()
                img_rec  = self(X_sub)[0].detach().cpu().numpy()

                draw(img_orig, 'orig_img.png')
                draw(img_rec, 'rec_img.png')

def draw(WxHxC, name):
    to_draw = np.transpose(WxHxC, (1,2,0))

    import matplotlib.pyplot as plt
    plt.imshow(to_draw)
    plt.savefig(name)

